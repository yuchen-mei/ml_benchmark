from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any

from .runtime import process_exit_reason


_RESULT_PREFIX = "__MLBENCH_GPU_RESULT__="


def run_gpu_benchmarks_isolated(
    backend: str,
    device_indices: list[int],
    suites: set[str],
    profile: str,
    requested_duration: float | None,
    matrix_size: int,
    batch_size: int,
    warmup: int,
    power_enabled: bool,
    power_interval: float,
    device_labels: dict[int, str] | None = None,
    device_architectures: dict[int, str] | None = None,
    apple_memory_limit_gib: float = 0,
    mps_matmul: str = "auto",
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    labels = device_labels or {}
    architectures = device_architectures or {}
    for index in device_indices:
        architecture = architectures.get(index, "")
        environment = os.environ.copy()
        if backend == "mps":
            # A GPU benchmark must never silently execute unsupported ops on the CPU.
            environment["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
            if mps_matmul != "auto":
                environment["PYTORCH_MPS_PREFER_METAL"] = "1" if mps_matmul == "metal" else "0"
        if backend == "rocm" and _is_gfx1151(architecture):
            environment.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
            if "backend:malloc" in environment.get("PYTORCH_HIP_ALLOC_CONF", ""):
                environment.pop("PYTORCH_HIP_ALLOC_CONF")
        idle_pending = power_enabled
        for task_suites, task_precisions, failure_fields in _worker_tasks(
            backend, architecture, suites
        ):
            timeout = _worker_timeout(profile, requested_duration, task_suites)
            configuration = {
                "backend": backend,
                "device_indices": [index],
                "suites": sorted(task_suites),
                "profile": profile,
                "requested_duration": requested_duration,
                "matrix_size": matrix_size,
                "batch_size": batch_size,
                "warmup": warmup,
                "power_enabled": power_enabled,
                "power_interval": power_interval,
                "device_architecture": architecture,
                "precisions": sorted(task_precisions) if task_precisions else None,
                "include_idle": idle_pending,
                "apple_memory_limit_gib": apple_memory_limit_gib,
            }
            try:
                completed = subprocess.run(
                    [sys.executable, "-m", "mlbench.gpu_worker"],
                    input=json.dumps(configuration),
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=timeout,
                    env=environment,
                )
            except subprocess.TimeoutExpired as exc:
                results.append(
                    _failure_result(
                        backend,
                        labels.get(index, f"GPU {index}"),
                        architecture,
                        f"GPU 子进程超过 {timeout:.0f} 秒仍未结束",
                        str(exc.stderr or ""),
                        fields=failure_fields,
                    )
                )
                continue
            if completed.returncode != 0:
                results.append(
                    _failure_result(
                        backend,
                        labels.get(index, f"GPU {index}"),
                        architecture,
                        f"GPU 子进程{process_exit_reason(completed.returncode, completed.stderr)}",
                        completed.stderr,
                        completed.returncode,
                        failure_fields,
                    )
                )
                continue
            payload = _extract_results(completed.stdout)
            if payload is None:
                results.append(
                    _failure_result(
                        backend,
                        labels.get(index, f"GPU {index}"),
                        architecture,
                        "GPU 子进程没有返回有效结果",
                        completed.stderr or completed.stdout,
                        completed.returncode,
                        failure_fields,
                    )
                )
                continue
            results.extend(payload)
            if any(item.get("test") == "idle_power" for item in payload):
                idle_pending = False
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
    backend: str,
    device: str,
    architecture: str,
    reason: str,
    diagnostic: str = "",
    returncode: int | None = None,
    fields: tuple[str, str, str] = ("diagnostic", "gpu_runtime", "n/a"),
) -> dict[str, Any]:
    if backend == "rocm" and _is_gfx1151(architecture):
        reason += (
            "。gfx1151 请使用架构专用 PyTorch wheel；"
            "优先通过 ./benchmark.sh 自动创建独立环境"
        )
    details: dict[str, Any] = {
        "reason": reason,
        "worker_failure": True,
        "architecture": architecture or None,
    }
    if returncode is not None:
        details["worker_returncode"] = returncode
    if diagnostic.strip():
        details["diagnostic"] = diagnostic.strip()[-1200:]
    return {
        "backend": backend,
        "device": device,
        "suite": fields[0],
        "test": fields[1],
        "precision": fields[2],
        "value": None,
        "unit": "",
        "status": "skipped",
        "details": details,
    }


def _worker_timeout(profile: str, requested_duration: float | None, suites: set[str]) -> float:
    duration = requested_duration if requested_duration is not None else {
        "quick": 0.35,
        "standard": 1.5,
        "extended": 4.0,
    }[profile]
    case_count = (4 if "compute" in suites else 0) + (1 if "memory" in suites else 0)
    case_count += 3 if "inference" in suites else 0
    return max(60.0, 45.0 + duration * max(case_count, 1) * 5.0)


def _is_gfx1151(architecture: str) -> bool:
    normalized = architecture.lower().split(":", 1)[0]
    return normalized == "gfx1151"


def _worker_tasks(
    backend: str, architecture: str, suites: set[str]
) -> list[tuple[set[str], set[str] | None, tuple[str, str, str]]]:
    if backend != "rocm" or not _is_gfx1151(architecture):
        return [(suites, None, ("diagnostic", "gpu_runtime", "n/a"))]
    tasks = []
    if "compute" in suites:
        tasks.extend(
            [
                ({"compute"}, {precision}, ("compute", "dense_matmul", precision))
                for precision in ("fp16", "fp32", "bf16")
            ]
        )
    if "memory" in suites:
        tasks.append(({"memory"}, None, ("memory", "device_copy", "fp32")))
    if "inference" in suites:
        tasks.extend(
            [
                ({"inference"}, {precision}, ("inference", "synthetic_mlp", precision))
                for precision in ("fp16", "fp32", "bf16")
            ]
        )
    return tasks
