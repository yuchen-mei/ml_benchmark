from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class PowerReader:
    source: str
    read: Callable[[], float | None]


class PowerSampler:
    def __init__(self, reader: PowerReader | None, interval: float) -> None:
        self.reader = reader
        self.interval = interval
        self.samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self.reader is None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self, elapsed_seconds: float) -> dict[str, Any] | None:
        if self.reader is None:
            return None
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval * 2.0))
        if not self.samples:
            return {"status": "unavailable", "source": self.reader.source, "samples": 0}
        average = sum(self.samples) / len(self.samples)
        return {
            "status": "ok",
            "source": self.reader.source,
            "samples": len(self.samples),
            "average_w": average,
            "min_w": min(self.samples),
            "peak_w": max(self.samples),
            "energy_j": average * elapsed_seconds,
            "sample_interval_s": self.interval,
        }

    def _sample_loop(self) -> None:
        assert self.reader is not None
        while not self._stop.is_set():
            value = self.reader.read()
            if value is not None and math.isfinite(value) and value >= 0:
                self.samples.append(value)
            self._stop.wait(self.interval)


def make_power_reader(backend: str, device_index: int = 0) -> PowerReader | None:
    if backend == "cuda" and shutil.which("nvidia-smi"):
        return PowerReader("nvidia-smi:power.draw", lambda: _read_nvidia(device_index))
    if backend == "rocm":
        if shutil.which("amd-smi"):
            return PowerReader("amd-smi:socket_power", lambda: _read_amd_smi(device_index))
        if shutil.which("rocm-smi"):
            return PowerReader("rocm-smi:power", lambda: _read_rocm_smi(device_index))
    if backend == "npu" and shutil.which("xrt-smi"):
        return PowerReader("xrt-smi:electrical", _read_xrt_smi)
    return None


def sample_idle_power(reader: PowerReader | None, interval: float) -> dict[str, Any] | None:
    if reader is None:
        return None
    duration = max(0.5, interval * 3.0)
    sampler = PowerSampler(reader, interval)
    sampler.start()
    time.sleep(duration)
    return sampler.stop(duration)


def add_power_details(result: dict[str, Any], power: dict[str, Any] | None) -> None:
    if power is None:
        result.setdefault("details", {})["power"] = {"status": "unsupported"}
        return
    result.setdefault("details", {})["power"] = power
    average = power.get("average_w")
    value = result.get("value")
    unit = result.get("unit")
    if power.get("status") == "ok" and average and value is not None and unit not in {"ms", "W"}:
        result["details"]["efficiency"] = {"value": value / average, "unit": f"{unit}/W"}


def idle_result(backend: str, device: str, summary: dict[str, Any] | None) -> dict[str, Any]:
    if summary is None:
        return _unavailable(backend, device, "平台没有可用的功耗遥测命令")
    if summary.get("status") != "ok":
        result = _unavailable(backend, device, f"{summary.get('source')} 未返回有效功耗")
        result["details"]["power"] = summary
        return result
    return {
        "backend": backend,
        "device": device,
        "suite": "power",
        "test": "idle_power",
        "precision": "n/a",
        "value": summary["average_w"],
        "unit": "W",
        "status": "ok",
        "details": {"power": summary},
    }


def _unavailable(backend: str, device: str, reason: str) -> dict[str, Any]:
    return {
        "backend": backend,
        "device": device,
        "suite": "power",
        "test": "idle_power",
        "precision": "n/a",
        "value": None,
        "unit": "W",
        "status": "skipped",
        "details": {"reason": reason},
    }


def _run(command: list[str]) -> str:
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=2.0)
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout if completed.returncode == 0 else ""


def _number(value: Any, unit: str | None = None) -> float | None:
    if isinstance(value, dict):
        return _number(value.get("value"), str(value.get("unit", unit or "")))
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
        if not match:
            return None
        number = float(match.group())
        unit = unit or value
    else:
        return None
    normalized_unit = (unit or "").lower()
    if "mw" in normalized_unit:
        return number / 1000.0
    if "uw" in normalized_unit or "µw" in normalized_unit:
        return number / 1_000_000.0
    return number


def _json_power(data: Any) -> float | None:
    preferred = {
        "socket_power",
        "average_socket_power",
        "current_socket_power",
        "average_graphics_package_power",
        "power_usage",
    }
    if isinstance(data, dict):
        for key, value in data.items():
            normalized = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
            is_power = normalized in preferred or (
                "power" in normalized and "cap" not in normalized and "limit" not in normalized
            )
            if is_power:
                number = _number(value)
                if number is not None:
                    return number
        for value in data.values():
            number = _json_power(value)
            if number is not None:
                return number
    elif isinstance(data, list):
        for value in data:
            number = _json_power(value)
            if number is not None:
                return number
    return None


def _read_nvidia(device_index: int) -> float | None:
    output = _run(
        [
            "nvidia-smi",
            f"--id={device_index}",
            "--query-gpu=power.draw",
            "--format=csv,noheader,nounits",
        ]
    )
    return _number(output)


def _read_amd_smi(device_index: int) -> float | None:
    output = _run(["amd-smi", "metric", "-g", str(device_index), "-p", "--json"])
    if not output:
        output = _run(["amd-smi", "metric", "--gpu", str(device_index), "--power", "--json"])
    try:
        return _json_power(json.loads(output))
    except (json.JSONDecodeError, TypeError):
        return _text_power(output)


def _read_rocm_smi(device_index: int) -> float | None:
    output = _run(["rocm-smi", "-d", str(device_index), "--showpower", "--json"])
    try:
        return _json_power(json.loads(output))
    except (json.JSONDecodeError, TypeError):
        return _text_power(output)


def _read_xrt_smi() -> float | None:
    for report in ("electrical", "telemetry"):
        output = _run(["xrt-smi", "examine", "--report", report])
        number = _text_power(output)
        if number is not None:
            return number
    return None


def _text_power(output: str) -> float | None:
    for line in output.splitlines():
        lowered = line.lower()
        if "power" not in lowered or "cap" in lowered or "limit" in lowered:
            continue
        match = re.search(r"(-?\d+(?:\.\d+)?)\s*(uw|µw|mw|w|watts?)\b", line, re.IGNORECASE)
        if match:
            return _number(match.group(1), match.group(2))
    return None
