import math
import unittest

from argparse import Namespace

from mlbench.cli import _device_indices, _parse_suites, _selected_backends, _validate_arguments
from mlbench.power import _json_power, _text_power, add_power_details
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


if __name__ == "__main__":
    unittest.main()
