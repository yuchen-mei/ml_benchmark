import json
import math
import os
import signal
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from argparse import Namespace

from mlbench.cli import (
    _device_indices,
    _gpu_metadata,
    _parse_suites,
    _resolve_suites,
    _selected_backends,
    _validate_arguments,
)
from mlbench.detect import _npu_runtime_warnings, _ort_runtime, _torch_runtime
from mlbench.gpu import _is_gfx1151 as uses_gfx1151_fallback
from mlbench.isolation import run_gpu_benchmarks_isolated
from mlbench.llm import (
    configure_llm_runtime,
    generation_metrics,
    resolve_llm_configuration,
    vram_budget_gib,
)
from mlbench.npu import _make_feed, _profile_provider_counts, _quicktest_model
from mlbench.npu_isolation import run_npu_benchmarks_isolated
from mlbench.npu_runtime import npu_subprocess_environment
from mlbench.power import _json_power, _text_power, add_power_details
from mlbench.report import _display_width, _pad_display, format_results
from mlbench.stats import percentile


class StatsTests(unittest.TestCase):
    def test_percentile_interpolates(self):
        self.assertEqual(percentile([1.0, 2.0, 3.0, 4.0], 50), 2.5)

    def test_empty_percentile_is_nan(self):
        self.assertTrue(math.isnan(percentile([], 95)))


class SelectionTests(unittest.TestCase):
    def test_all_prefers_accelerators(self):
        self.assertEqual(_selected_backends("all", ["cuda", "npu", "cpu"]), ["cuda", "npu"])

    def test_all_falls_back_to_cpu(self):
        self.assertEqual(_selected_backends("all", ["cpu"]), ["cpu"])

    def test_suite_list(self):
        self.assertEqual(_parse_suites("compute,memory"), {"compute", "memory"})

    def test_llm_profile_selects_only_llm(self):
        self.assertEqual(_resolve_suites("llm", "all"), {"llm"})

    def test_llm_cannot_mix_with_synthetic_suites(self):
        with self.assertRaises(ValueError):
            _resolve_suites("standard", "compute,llm")

    def test_device_selection(self):
        self.assertEqual(_device_indices("0,2", 3), [0, 2])

    def test_invalid_duration(self):
        arguments = Namespace(
            duration=0,
            matrix_size=0,
            batch_size=0,
            warmup=0,
            npu_streams=1,
            power_interval=0.1,
            backend="cpu",
        )
        with self.assertRaises(ValueError):
            _validate_arguments(arguments, {"compute"})

    def test_gpu_metadata_uses_sysfs_architecture_fallback(self):
        environment = {
            "runtime": {"torch": {"devices": [{"index": 0, "name": "AMD Radeon Graphics"}]}},
            "hardware": {"amd_gpus": [{"device_id": "0x1586", "architecture": "gfx1151"}]},
        }
        labels, architectures = _gpu_metadata(environment)
        self.assertEqual(labels, {0: "0: AMD Radeon Graphics"})
        self.assertEqual(architectures, {0: "gfx1151"})


class NativeCrashTests(unittest.TestCase):
    def test_doctor_contains_torch_sigsegv(self):
        partial = (
            '__MLBENCH_TORCH_PROBE__={"installed":true,"available":false,'
            '"version":"2.14.0+rocm7.2","hip_build":"7.2","devices":[],'
            '"probe":{"status":"probing"}}\n'
        )
        completed = subprocess.CompletedProcess([], -signal.SIGSEGV, partial, "")
        with (
            patch("mlbench.detect.importlib.util.find_spec", return_value=object()),
            patch("mlbench.detect.subprocess.run", return_value=completed),
        ):
            runtime = _torch_runtime()
        self.assertFalse(runtime["available"])
        self.assertEqual(runtime["version"], "2.14.0+rocm7.2")
        self.assertEqual(runtime["probe"]["status"], "failed")
        self.assertIn("SIGSEGV", runtime["probe"]["reason"])

    def test_gpu_sigsegv_becomes_diagnostic_result(self):
        completed = subprocess.CompletedProcess([], -signal.SIGSEGV, "", "")
        with patch("mlbench.isolation.subprocess.run", return_value=completed):
            results = run_gpu_benchmarks_isolated(
                "rocm",
                [0],
                {"compute"},
                "quick",
                None,
                0,
                0,
                0,
                False,
                0.1,
                {0: "0: AMD Radeon Graphics"},
                {0: "gfx1151"},
            )
        self.assertEqual(len(results), 3)
        self.assertTrue(all(item["details"]["worker_failure"] for item in results))
        self.assertTrue(all("SIGSEGV" in item["details"]["reason"] for item in results))
        self.assertTrue(all("架构专用" in item["details"]["reason"] for item in results))

    def test_gfx1151_uses_safe_inference_fallback(self):
        self.assertTrue(uses_gfx1151_fallback("rocm", "gfx1151:sramecc+:xnack-"))
        self.assertFalse(uses_gfx1151_fallback("cuda", "gfx1151"))

    def test_external_npu_runtime_probe(self):
        payload = (
            '__MLBENCH_ORT_PROBE__={"installed":true,"version":"1.23.3",'
            '"providers":["VitisAIExecutionProvider","CPUExecutionProvider"]}\n'
        )
        completed = subprocess.CompletedProcess([], 0, payload, "")
        with (
            patch.dict("os.environ", {"MLBENCH_NPU_PYTHON": "/opt/ryzenai/venv/bin/python"}),
            patch("mlbench.detect.subprocess.run", return_value=completed),
        ):
            runtime = _ort_runtime()
        self.assertTrue(runtime["external"])
        self.assertIn("VitisAIExecutionProvider", runtime["providers"])

    def test_npu_sigsegv_becomes_diagnostic_result(self):
        completed = subprocess.CompletedProcess([], -signal.SIGSEGV, "", "")
        with patch("mlbench.npu_isolation.subprocess.run", return_value=completed):
            results = run_npu_benchmarks_isolated(
                "/opt/ryzenai/venv/bin/python",
                Path("/tmp/results"),
                "quick",
                None,
                0,
                0,
                1,
                None,
                None,
                False,
                0.1,
            )
        self.assertTrue(results[0]["details"]["worker_failure"])
        self.assertIn("SIGSEGV", results[0]["details"]["reason"])


class NpuRuntimeTests(unittest.TestCase):
    def test_generic_onnxruntime_warning_is_actionable(self):
        with patch("mlbench.detect._is_ubuntu_2404", return_value=False):
            warnings = _npu_runtime_warnings(
                "Linux",
                {
                    "installed": True,
                    "executable": "/usr/bin/python",
                    "providers": ["CPUExecutionProvider"],
                },
            )
        self.assertIn("PyPI 通用 onnxruntime", warnings[0])
        self.assertIn("Ubuntu 24.04", warnings[1])

    def test_runtime_environment_adds_xrt_and_ort_libraries(self):
        with patch.dict("os.environ", {}, clear=True):
            environment = npu_subprocess_environment("/opt/ryzenai/venv/bin/python")
        self.assertEqual(environment["RYZEN_AI_INSTALLATION_PATH"], "/opt/ryzenai/venv")
        self.assertIn("/opt/ryzenai/venv/onnxruntime/lib", environment["LD_LIBRARY_PATH"])
        self.assertIn("/opt/xilinx/xrt/lib", environment["LD_LIBRARY_PATH"])

    def test_ryzenai_18_libraries_keep_system_xrt_first(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "venv"
            site_packages = root / "lib/python3.12/site-packages"
            peano = site_packages / "lnx64.o/tools/peano/lib"
            voe = site_packages / "voe/lib"
            peano.mkdir(parents=True)
            voe.mkdir(parents=True)
            executable = root / "bin/python"
            existing = f"{voe}:/legacy/lib"
            with patch.dict(
                "os.environ",
                {"LD_LIBRARY_PATH": existing, "XILINX_XRT": "/opt/xilinx/xrt"},
                clear=True,
            ):
                environment = npu_subprocess_environment(str(executable))
        libraries = environment["LD_LIBRARY_PATH"].split(":")
        self.assertEqual(libraries[0], "/opt/xilinx/xrt/lib")
        self.assertIn(str(peano), libraries)
        self.assertLess(libraries.index("/opt/xilinx/xrt/lib"), libraries.index(str(voe)))

    def test_uses_bundled_quicktest_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "venv"
            model = root / "quicktest/model.onnx"
            model.parent.mkdir(parents=True)
            model.touch()
            with patch.dict("os.environ", {"RYZEN_AI_INSTALLATION_PATH": str(root)}):
                selected, source = _quicktest_model(None)
        self.assertEqual(selected, model.resolve())
        self.assertEqual(source, "ryzen_ai_quicktest")

    def test_quicktest_feed_resolves_dynamic_batch(self):
        metadata = type(
            "InputMetadata",
            (),
            {"name": "input", "type": "tensor(float)", "shape": [None, 3, 32, 32]},
        )()
        import numpy as np

        feed, batch = _make_feed([metadata], 4, np.random.default_rng(9))
        self.assertEqual(feed["input"].shape, (4, 3, 32, 32))
        self.assertEqual(batch, 4)

    def test_profile_proves_vitisai_execution(self):
        events = [
            {"args": {"provider": "CPUExecutionProvider"}},
            {"args": {"provider": "VitisAIExecutionProvider"}},
            {"args": {"provider": "VitisAIExecutionProvider"}},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(events), encoding="utf-8")
            counts = _profile_provider_counts(path)
        self.assertEqual(counts["VitisAIExecutionProvider"], 2)


class LLMTests(unittest.TestCase):
    def test_llm_runtime_disables_optional_native_jit_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            configure_llm_runtime()
            self.assertEqual(os.environ["TORCH_DISABLE_NATIVE_JIT"], "1")

    def test_llm_runtime_preserves_explicit_native_jit_opt_in(self):
        with patch.dict(os.environ, {"TORCH_DISABLE_NATIVE_JIT": "0"}):
            configure_llm_runtime()
            self.assertEqual(os.environ["TORCH_DISABLE_NATIVE_JIT"], "0")

    def test_presets_fit_expected_quantization(self):
        _, qwen_model, qwen_quantization, qwen_estimate = resolve_llm_configuration(
            "qwen", None, "auto"
        )
        _, kimi_model, kimi_quantization, kimi_estimate = resolve_llm_configuration(
            "kimi", None, "auto"
        )
        self.assertEqual(qwen_model, "Qwen/Qwen3-4B-Instruct-2507")
        self.assertEqual(qwen_quantization, "none")
        self.assertLess(qwen_estimate, 22.0)
        self.assertEqual(kimi_model, "moonshotai/Kimi-VL-A3B-Instruct")
        self.assertEqual(kimi_quantization, "4bit")
        self.assertLess(kimi_estimate, 22.0)

        _, _, unquantized_kimi, unquantized_estimate = resolve_llm_configuration(
            "kimi", None, "none"
        )
        self.assertEqual(unquantized_kimi, "none")
        self.assertGreater(unquantized_estimate, 24.0)

    def test_custom_model_does_not_inherit_unknown_memory_estimate(self):
        _, model, quantization, estimate = resolve_llm_configuration(
            "kimi", "/models/custom", "auto"
        )
        self.assertEqual(model, "/models/custom")
        self.assertEqual(quantization, "none")
        self.assertIsNone(estimate)

    def test_vram_budget_honors_free_memory_limit_and_reserve(self):
        self.assertEqual(vram_budget_gib(24.0, 23.5), 21.5)
        self.assertEqual(vram_budget_gib(48.0, 47.0), 22.0)
        self.assertEqual(vram_budget_gib(16.0, 15.0), 13.0)

    def test_generation_metrics_separate_prefill_and_decode(self):
        metrics = generation_metrics(
            [0.1, 0.2],
            [0.9, 1.0],
            [1.0, 1.2],
            [20, 20],
            10,
            1,
        )
        self.assertAlmostEqual(metrics["ttft_p50_ms"], 150.0)
        self.assertAlmostEqual(metrics["prefill_tokens_per_second"], 20.0 / 0.3)
        self.assertAlmostEqual(metrics["decode_tokens_per_second"], 38.0 / 1.9)
        self.assertAlmostEqual(metrics["output_tokens_per_second"], 40.0 / 2.2)
        self.assertAlmostEqual(metrics["e2e_p50_seconds"], 1.1)

    def test_decode_metric_does_not_subtract_independent_ttft(self):
        metrics = generation_metrics([0.4], [0.1], [0.2], [4], 32, 1)
        self.assertAlmostEqual(metrics["decode_tokens_per_second"], 30.0)


class PowerTests(unittest.TestCase):
    def test_amd_json_power(self):
        payload = [{"gpu": 0, "power": {"socket_power": {"value": 125, "unit": "W"}}}]
        self.assertEqual(_json_power(payload), 125.0)

    def test_text_power_units(self):
        self.assertEqual(_text_power("Average Package Power: 75000 mW"), 75.0)

    def test_efficiency(self):
        result = {"value": 100.0, "unit": "images/s", "details": {}}
        add_power_details(result, {"status": "ok", "average_w": 20.0})
        self.assertEqual(result["details"]["efficiency"]["value"], 5.0)


class ReportTests(unittest.TestCase):
    @staticmethod
    def result(precision="fp32"):
        return {
            "backend": "cuda",
            "device": "0: NVIDIA GeForce RTX 4090",
            "suite": "compute",
            "test": "dense_matmul",
            "precision": precision,
            "value": 46.241,
            "unit": "TFLOP/s",
            "status": "ok",
            "details": {
                "power": {"average_w": 313.1, "peak_w": 398.8},
                "efficiency": {"value": 0.148, "unit": "TFLOP/s/W"},
            },
        }

    def test_cjk_display_width(self):
        self.assertEqual(_display_width("平均功耗"), 8)
        self.assertEqual(_display_width(_pad_display("项目", 10)), 10)

    def test_wide_table_lines_align(self):
        output = format_results([self.result(), self.result("fp16")], terminal_width=160)
        table_lines = [line for line in output.splitlines() if line.startswith(("+", "|"))]
        self.assertEqual(len({_display_width(line) for line in table_lines}), 1)
        self.assertEqual(output.count("0: NVIDIA GeForce RTX 4090"), 1)

    def test_narrow_terminal_uses_stacked_layout(self):
        output = format_results([self.result()], terminal_width=60)
        self.assertNotIn("+---", output)
        self.assertIn("- dense_matmul [fp32]", output)
        self.assertIn("功耗: 平均 313.1 W / 峰值 398.8 W", output)


if __name__ == "__main__":
    unittest.main()
