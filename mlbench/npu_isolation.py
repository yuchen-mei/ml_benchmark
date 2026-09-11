from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from .npu_runtime import npu_subprocess_environment
from .runtime import process_exit_reason


_RESULT_PREFIX = "__MLBENCH_NPU_RESULT__="


def run_npu_benchmarks_isolated(
    executable: str,
    output_dir: Path,
    profile: str,
    requested_duration: float | None,
    batch_size: int,
    warmup: int,
    streams: int,
    config_file: str | None,
    model_file: str | None,
    power_enabled: bool,
    power_interval: float,
) -> list[dict[str, Any]]:
    configuration = {
        "output_dir": str(output_dir),
        "profile": profile,
        "requested_duration": requested_duration,
        "batch_size": batch_size,
        "warmup": warmup,
        "streams": streams,
        "config_file": config_file,
        "model_file": model_file,
        "power_enabled": power_enabled,
        "power_interval": power_interval,
    }
    environment = npu_subprocess_environment(executable)
    package_root = str(Path(__file__).resolve().parent.parent)
    current_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        f"{package_root}{os.pathsep}{current_pythonpath}" if current_pythonpath else package_root
    )
    try:
        completed = subprocess.run(
            [executable, "-m", "mlbench.npu_worker"],
            input=json.dumps(configuration),
            capture_output=True,
            text=True,
            check=False,
            timeout=_worker_timeout(profile, requested_duration),
            env=environment,
        )
    except subprocess.TimeoutExpired as exc:
        return [_failure_result(executable, "NPU 子进程超时", str(exc.stderr or ""))]
    if completed.returncode != 0:
        reason = f"NPU 子进程{process_exit_reason(completed.returncode, completed.stderr)}"
        return [_failure_result(executable, reason, completed.stderr, completed.returncode)]
    results = _extract_results(completed.stdout)
    if results is None:
        return [
            _failure_result(
                executable,
                "NPU 子进程没有返回有效结果",
                completed.stderr or completed.stdout,
                completed.returncode,
            )
        ]
    return results


def _extract_results(output: str) -> list[dict[str, Any]] | None:
    for line in reversed(output.splitlines()):
        if line.startswith(_RESULT_PREFIX):
            try:
                value = json.loads(line[len(_RESULT_PREFIX) :])
            except json.JSONDecodeError:
                return None
            return value if isinstance(value, list) else None
    return None


def _failure_result(
    executable: str,
    reason: str,
    diagnostic: str = "",
    returncode: int | None = None,
) -> dict[str, Any]:
    details: dict[str, Any] = {
        "reason": reason,
        "worker_failure": True,
        "runtime_python": executable,
    }
    if returncode is not None:
        details["worker_returncode"] = returncode
    if diagnostic.strip():
        details["diagnostic"] = diagnostic.strip()[-1200:]
    return {
        "backend": "npu",
        "device": "AMD NPU / VitisAIExecutionProvider",
        "suite": "diagnostic",
        "test": "npu_runtime",
        "precision": "n/a",
        "value": None,
        "unit": "",
        "status": "skipped",
        "details": details,
    }


def _worker_timeout(profile: str, requested_duration: float | None) -> float:
    duration = requested_duration if requested_duration is not None else {
        "quick": 0.5,
        "standard": 3.0,
        "extended": 10.0,
    }[profile]
    return max(900.0, 300.0 + duration * 10.0)
