import contextlib
import io
import json
import signal
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mlbench.cli import _parser, main
from mlbench.llm import LLM_PRESETS, _selected_dtype, resolve_llm_configuration
from mlbench.llm_sweep import prompt_lengths, selected_models, selected_precisions
from mlbench.report import format_results, save_report


class LLMSweepTests(unittest.TestCase):
    def test_defaults_cover_all_models_precisions_and_prompt_lengths(self):
        args = _parser().parse_args(["run", "--profile", "llm"])
        models = selected_models(args.llm_preset, args.llm_model)
        precisions = selected_precisions(args.llm_quantization)
        lengths = prompt_lengths(args.llm_prompt_tokens)
        self.assertEqual(set(models), {"llama", "qwen", "deepseek", "kimi"})
        self.assertEqual(precisions, ["fp32", "fp16", "bf16", "8bit", "4bit"])
        self.assertEqual(lengths, [128, 256, 512, 1024, 2048])
        self.assertEqual(len(models) * len(precisions) * len(lengths), 100)

    def test_explicit_overrides_narrow_the_sweep(self):
        args = _parser().parse_args(["run", "--profile", "llm", "--llm-preset", "deepseek",
                                     "--llm-dtype", "bf16", "--llm-prompt-tokens", "2048"])
        self.assertEqual(selected_models(args.llm_preset, None), ["deepseek"])
        self.assertEqual(selected_precisions(args.llm_quantization), ["bf16"])
        self.assertEqual(prompt_lengths(args.llm_prompt_tokens), [2048])
        self.assertEqual(len(selected_models("all", "/models/local")), 1)
        self.assertEqual(prompt_lengths("2048,128,512,128"), [128, 512, 2048])

    def test_invalid_lengths_rejected(self):
        for raw in ("0", "-1", "128,", "128,x", "1.5"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                prompt_lengths(raw)

    def test_torch_fp32_budget_and_explicit_bf16(self):
        _, _, quantization, estimate = resolve_llm_configuration("qwen", None, "fp32")
        self.assertEqual(quantization, "none")
        self.assertEqual(estimate, 18.0)
        torch = SimpleNamespace(float32="float32", float16="float16", bfloat16="bfloat16",
                                cuda=SimpleNamespace(is_bf16_supported=lambda: False),
                                backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
                                                         cudnn=SimpleNamespace(allow_tf32=True)))
        self.assertEqual(_selected_dtype(torch, "fp32"), ("float32", "fp32"))
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        with self.assertRaisesRegex(RuntimeError, "BF16"):
            _selected_dtype(torch, "bf16")

    @staticmethod
    def environment():
        return {"system": {"platform": "test", "python": "3.11"},
                "hardware": {"apple_silicon": {"detected": True, "name": "Apple Test", "unified_memory_bytes": 128 * 1024**3}},
                "runtime": {}, "available_backends": ["mlx", "cpu"], "warnings": []}

    @staticmethod
    def item(model="Qwen/test", precision="fp16", prompt=128):
        return {"backend": "mlx", "device": "Apple Test", "suite": "llm", "test": "llm_prefill",
                "precision": precision, "value": 100.0, "unit": "tokens/s", "status": "ok",
                "details": {"model": model, "requested_precision": precision, "prompt_tokens_per_request": prompt,
                            "new_tokens_per_request": 128, "batch_size": 1, "runs": 3}}

    def test_default_cli_attempts_100_cases_and_preserves_progress_after_crash(self):
        cases = []
        def execute(command, **kwargs):
            config = json.loads(kwargs["input"])
            cases.append(config)
            if len(cases) == 1:
                return subprocess.CompletedProcess(command, -signal.SIGABRT, "", "GPU failed")
            item = self.item(precision=config["quantization"], prompt=config["prompt_tokens"])
            return subprocess.CompletedProcess(command, 0, "__MLBENCH_GPU_RESULT__=" + json.dumps([item]), "")
        with tempfile.TemporaryDirectory() as directory:
            output = io.StringIO()
            with patch("mlbench.cli.detect_environment", return_value=self.environment()), patch(
                "mlbench.llm_sweep.subprocess.run", side_effect=execute
            ), contextlib.redirect_stdout(output):
                status = main(["--profile", "llm", "--json-only", "--no-power", "--output-dir", directory])
            report = json.loads(output.getvalue())
            self.assertEqual(len(list(Path(directory).glob("*.json"))), 1)
            self.assertEqual(len(list(Path(directory).glob("*.md"))), 1)
        self.assertEqual(status, 3)
        self.assertEqual(len(cases), 100)
        self.assertEqual(len(report["results"]), 100)
        self.assertTrue(report["complete"])
        self.assertIn("SIGABRT", report["results"][0]["details"]["reason"])
        self.assertEqual({item["details"]["model"] for item in report["results"]},
                         {preset.model_id for preset in LLM_PRESETS.values()})
        self.assertEqual(report["results"][-1]["details"]["prompt_tokens_per_request"], 2048)

    def test_partial_report_is_retained_on_interruption(self):
        with tempfile.TemporaryDirectory() as directory:
            environment = self.environment()
            paths = save_report(Path(directory), environment, {}, [self.item()], complete=False)
            self.assertFalse(json.loads(paths[0].read_text())["complete"])
            self.assertIn("Partial report", paths[1].read_text())
            final = save_report(Path(directory), environment, {}, [self.item(), self.item(prompt=2048)], paths=paths)
            self.assertEqual(paths, final)
            self.assertTrue(json.loads(final[0].read_text())["complete"])
            self.assertEqual(len(list(Path(directory).glob("*.tmp"))), 0)
            failed = self.item()
            failed.update(status="skipped", value=None)
            failed["details"]["reason"] = "Cannot load model\nLicense | required"
            save_report(Path(directory), environment, {}, [failed], paths=paths)
            self.assertIn("Cannot load model<br>License \\| required", paths[1].read_text())

    def test_reports_distinguish_models_and_prompt_lengths(self):
        items = [self.item(), self.item(prompt=2048), self.item(model="Llama/test")]
        output = format_results(items, 160)
        self.assertIn("Qwen/test", output)
        self.assertIn("Llama/test", output)
        self.assertIn("2048", output)
        table_lines = [line for line in output.splitlines() if line.startswith(("+", "|"))]
        self.assertGreater(len(table_lines), 0)
        narrow = format_results(items, 50)
        self.assertIn("fp16 / 2048 input tokens", narrow)


if __name__ == "__main__":
    unittest.main()
