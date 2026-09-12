"""Apple Silicon hardware metadata and safe, isolated runtime probes."""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from typing import Any

from .runtime import process_exit_reason


def apple_hardware() -> dict[str, Any]:
    if platform.system() != "Darwin":
        return {"detected": False}
    def sysctl(key: str) -> str:
        try:
            return subprocess.check_output(["/usr/sbin/sysctl", "-n", key], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    chip = sysctl("machdep.cpu.brand_string")
    memory = sysctl("hw.memsize")
    return {
        "detected": chip.startswith("Apple") or platform.machine() == "arm64",
        "name": chip or "Apple Silicon",
        "native_arm64": platform.machine() == "arm64",
        "unified_memory_bytes": int(memory) if memory.isdigit() else 0,
    }


def probe_apple_runtime(backend: str) -> dict[str, Any]:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return {"installed": False, "available": False, "reason": "需要原生 arm64 macOS Python"}
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "mlbench.apple_worker", "probe", backend],
            capture_output=True, text=True, timeout=45,
        )
        for line in reversed(completed.stdout.splitlines()):
            if line.startswith("__MLBENCH_APPLE_PROBE__=") and completed.returncode == 0:
                return json.loads(line.split("=", 1)[1])
        reason = process_exit_reason(completed.returncode, completed.stderr)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        reason = str(exc)
    return {"installed": False, "available": False, "reason": reason}


def unified_memory_budget(total_bytes: int, recommended_bytes: int, limit_gib: float = 0,
                          reserve_gib: float = 4) -> int:
    """Keep macOS headroom and never exceed Metal's recommended working set."""
    limits = [int(total_bytes * 0.75), recommended_bytes, total_bytes - int(reserve_gib * 1024**3)]
    if limit_gib > 0:
        limits.append(int(limit_gib * 1024**3))
    budget = min(limits)
    if budget <= 0:
        raise ValueError("统一内存预算不足；请减小保留量或关闭其他应用")
    return budget


def run_apple_isolated(backend: str, configuration: dict[str, Any], timeout: float = 600) -> list[dict[str, Any]]:
    from .isolation import _extract_results, _failure_result
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "mlbench.apple_worker", "run", backend],
            input=json.dumps(configuration), capture_output=True, text=True, timeout=timeout,
        )
        payload = _extract_results(completed.stdout)
        if completed.returncode == 0 and payload is not None:
            return payload
        reason = process_exit_reason(completed.returncode, completed.stderr)
    except (OSError, subprocess.SubprocessError) as exc:
        reason = str(exc)
    return [_failure_result(backend, f"Apple Silicon / {backend}", "arm64", reason)]
