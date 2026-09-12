import json
import contextlib
import io
import os
import signal
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from mlbench.apple import probe_apple_runtime, run_apple_isolated, unified_memory_budget
from mlbench.cli import _parser, _selected_backends, _validate_arguments
from mlbench.coreml import _placement
from mlbench.gpu import _matmul_mode, _time_mps_operation
from mlbench.isolation import run_gpu_benchmarks_isolated
from mlbench.mlx_backend import _validate_model_code
from mlbench.report import _placement_note


GIB = 1024**3


class AppleTests(unittest.TestCase):
    def test_memory_budget_scales_beyond_24gb_without_exceeding_metal(self):
        self.assertEqual(unified_memory_budget(128 * GIB, 108 * GIB), 96 * GIB)
        self.assertEqual(unified_memory_budget(16 * GIB, 10 * GIB), 10 * GIB)
        self.assertEqual(unified_memory_budget(128 * GIB, 108 * GIB, 32), 32 * GIB)
        with self.assertRaises(ValueError):
            unified_memory_budget(4 * GIB, 3 * GIB)

    def test_backends_and_llm_validation(self):
        self.assertEqual(_selected_backends("all", ["mps", "mlx", "coreml", "cpu"]), ["mps", "mlx", "coreml"])
        args = _parser().parse_args(["run", "--backend", "mlx", "--profile", "llm"])
        _validate_arguments(args, {"llm"})
        for backend in ("mps", "coreml"):
            args.backend = backend
            with self.assertRaisesRegex(ValueError, "MLX"):
                _validate_arguments(args, {"llm"})

    def test_invalid_memory_budget_and_coreml_suites(self):
        args = _parser().parse_args(["run", "--backend", "coreml"])
        with self.assertRaisesRegex(ValueError, "inference"):
            _validate_arguments(args, {"compute"})
        args.apple_memory_limit_gib = float("nan")
        with self.assertRaisesRegex(ValueError, "有限"):
            _validate_arguments(args, {"inference"})

    def test_mps_matmul_mode_never_uses_cuda(self):
        with _matmul_mode(SimpleNamespace(), "mps", False):
            pass

    def test_mps_timer_synchronizes_work_and_counts_operations(self):
        clock = [0.0]
        queued = [0]
        total = [0]
        def operation():
            queued[0] += 1
            total[0] += 1
        def sync():
            clock[0] += queued[0] * 0.01
            queued[0] = 0
        torch = SimpleNamespace(mps=SimpleNamespace(synchronize=sync))
        with patch("mlbench.gpu.time.perf_counter", side_effect=lambda: clock[0]):
            elapsed, runs, _ = _time_mps_operation(torch, operation, 0.2, 2, None, 0.1)
        self.assertGreaterEqual(elapsed, 0.2)
        self.assertAlmostEqual(elapsed, runs * 0.01)
        self.assertEqual(total[0], runs + 3)  # Warmup and pilot excluded.
        self.assertEqual(queued[0], 0)

    def test_mps_worker_disables_cpu_fallback_and_selects_kernel(self):
        completed = subprocess.CompletedProcess([], 0, "__MLBENCH_GPU_RESULT__=[]\n", "")
        with patch.dict(os.environ, {"PYTORCH_ENABLE_MPS_FALLBACK": "1"}), patch(
            "mlbench.isolation.subprocess.run", return_value=completed
        ) as run:
            run_gpu_benchmarks_isolated("mps", [0], {"compute"}, "quick", None,
                                        0, 0, 1, False, 0.1, mps_matmul="metal")
        self.assertEqual(run.call_args.kwargs["env"]["PYTORCH_ENABLE_MPS_FALLBACK"], "0")
        self.assertEqual(run.call_args.kwargs["env"]["PYTORCH_MPS_PREFER_METAL"], "1")

    def test_native_apple_crash_becomes_reportable_failure(self):
        completed = subprocess.CompletedProcess([], -signal.SIGABRT, "", "Metal failed")
        with patch("mlbench.apple.subprocess.run", return_value=completed):
            result = run_apple_isolated("mlx", {})[0]
        self.assertTrue(result["details"]["worker_failure"])
        self.assertIn("SIGABRT", result["details"]["reason"])

    def test_probe_requires_successful_process_and_handles_missing_dependency(self):
        completed = subprocess.CompletedProcess([], 0, '__MLBENCH_APPLE_PROBE__={"installed":false,"available":false}\n', "")
        with patch("mlbench.apple.platform.system", return_value="Darwin"), patch(
            "mlbench.apple.platform.machine", return_value="arm64"
        ), patch("mlbench.apple.subprocess.run", return_value=completed):
            self.assertFalse(probe_apple_runtime("mlx")["available"])

    def test_custom_model_code_requires_trust_and_stays_in_model_directory(self):
        root = Path("/tmp/model")
        with self.assertRaisesRegex(ValueError, "trust-remote-code"):
            _validate_model_code({"model_file": "model.py"}, root, False)
        with self.assertRaisesRegex(ValueError, "目录内"):
            _validate_model_code({"model_file": "../outside.py"}, root, True)
        _validate_model_code({"model_file": "model.py"}, root, True)

    def test_coreml_plan_distinguishes_cpu_fallback_and_unknown(self):
        operation = SimpleNamespace(blocks=[])
        plan = Mock()
        plan.model_structure.program.functions = {"main": SimpleNamespace(block=SimpleNamespace(operations=[operation]))}
        plan.get_compute_device_usage_for_mlprogram_operation.return_value = SimpleNamespace(
            preferred_compute_device=type("MLCPUComputeDevice", (), {})()
        )
        ct = Mock()
        ct.models.compute_plan.MLComputePlan.load_from_path.return_value = plan
        result = _placement(ct, Path("cache.mlmodelc"), "CPU_AND_NE")
        self.assertFalse(result["neural_engine_planned"])
        self.assertIn("未分配", _placement_note(result))
        ct.models.compute_plan.MLComputePlan.load_from_path.side_effect = RuntimeError("unsupported")
        result = _placement(ct, Path("cache.mlmodelc"), "CPU_AND_NE")
        self.assertIsNone(result["neural_engine_planned"])
        self.assertIn("无法确认", _placement_note(result))


@unittest.skipUnless(os.environ.get("MLBENCH_TEST_APPLE") == "1", "set MLBENCH_TEST_APPLE=1 on Apple Silicon")
class AppleIntegrationTests(unittest.TestCase):
    def test_local_mlx_llm_fixed_length_quantized_and_batched(self):
        """Small random model tests the full loader/cache/timer without network weights."""
        import mlx.core as mx
        from mlx.utils import tree_flatten
        from mlx_lm.models.llama import Model, ModelArgs
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast
        from mlbench.mlx_backend import run_mlx_llm

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = dict(model_type="llama", hidden_size=64, num_hidden_layers=1,
                          intermediate_size=128, num_attention_heads=4, num_key_value_heads=2,
                          rms_norm_eps=1e-5, vocab_size=64, max_position_embeddings=4096,
                          tie_word_embeddings=True)
            (root / "config.json").write_text(json.dumps(config))
            model = Model(ModelArgs(**config))
            model.set_dtype(mx.float16)
            mx.save_safetensors(str(root / "model.safetensors"), dict(tree_flatten(model.parameters())))
            raw = Tokenizer(WordLevel({"[UNK]": 0, **{f"t{i}": i for i in range(1, 64)}}, unk_token="[UNK]"))
            raw.pre_tokenizer = Whitespace()
            PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]").save_pretrained(str(root))
            for precision in ("fp32", "fp16", "bf16", "8bit", "4bit"):
                with self.subTest(precision=precision):
                    results = run_mlx_llm("qwen", str(root), precision, 16, 8, 2, 2, 1,
                                          False, None, True, False, 2)
                    self.assertEqual(results[0]["precision"], precision)
                    if precision in {"fp32", "fp16", "bf16"}:
                        expected = {"fp32": "mlx.core.float32", "fp16": "mlx.core.float16", "bf16": "mlx.core.bfloat16"}[precision]
                        self.assertEqual(results[0]["details"]["parameter_dtypes"], [expected])
            # Exercise the real CLI/workers/checkpoints across all 25 dtype/length pairs.
            from mlbench.cli import main
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = main(["--profile", "llm", "--llm-model", str(root),
                               "--llm-new-tokens", "4", "--llm-runs", "1", "--warmup", "0",
                               "--no-power", "--json-only", "--llm-local-files-only",
                               "--output-dir", str(root / "reports")])
            report = json.loads(output.getvalue())
            self.assertEqual(status, 0, [item["details"].get("reason") for item in report["results"] if item["status"] != "ok"])
            self.assertEqual(len(report["results"]), 150)
            self.assertTrue(report["complete"])
            self.assertEqual({item["details"]["prompt_tokens_per_request"] for item in report["results"]},
                             {128, 256, 512, 1024, 2048})
        self.assertEqual(len(results), 6)
        self.assertTrue(all(item["status"] == "ok" and item["value"] > 0 for item in results))
        self.assertEqual(results[0]["details"]["total_output_tokens"], 32)
        self.assertEqual(results[0]["precision"], "4bit")

    def test_coreml_cache_reuse_concurrency_and_cpu_reference(self):
        from mlbench.coreml import run_coreml_benchmarks
        with tempfile.TemporaryDirectory() as directory:
            config = dict(output_dir=directory, profile="quick", requested_duration=0.1,
                          batch_size=1, warmup=1, streams=2, compute_units="cpu_only", power_enabled=False)
            first = run_coreml_benchmarks(**config)
            second = run_coreml_benchmarks(**config)
            self.assertFalse(first[0]["details"]["cache_hit"])
            self.assertTrue(second[0]["details"]["cache_hit"])
            self.assertEqual(second[0]["details"]["compile_seconds"], 0)
            self.assertGreater(second[0]["value"], 0)
            self.assertFalse(second[0]["details"]["placement"]["neural_engine_planned"])
            import coremltools as ct
            import numpy as np
            model_path = str(Path(second[0]["details"]["model"]).with_suffix(".mlmodelc"))
            cpu = ct.models.CompiledMLModel(model_path, compute_units=ct.ComputeUnit.CPU_ONLY)
            ane = ct.models.CompiledMLModel(model_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
            inputs = {"inputs": np.random.default_rng(7).standard_normal((1, 3, 224, 224), dtype=np.float32)}
            np.testing.assert_allclose(cpu.predict(inputs)["output"], ane.predict(inputs)["output"], rtol=0.05, atol=0.002)


if __name__ == "__main__":
    unittest.main()
