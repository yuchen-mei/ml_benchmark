import math
import unittest

from argparse import Namespace

from mlbench.cli import _device_indices, _parse_suites, _selected_backends, _validate_arguments
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
