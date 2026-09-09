from __future__ import annotations

import gc
import math
import time
from contextlib import contextmanager
from typing import Any, Callable


def _result(
    backend: str,
    device: str,
    suite: str,
    test: str,
    precision: str,
    value: float | None,
    unit: str,
    **details: Any,
) -> dict[str, Any]:
    return {
        "backend": backend,
        "device": device,
        "suite": suite,
        "test": test,
        "precision": precision,
        "value": value,
        "unit": unit,
        "status": "ok" if value is not None else "skipped",
        "details": details,
    }


def _skipped(backend: str, device: str, suite: str, test: str, precision: str, error: Exception) -> dict[str, Any]:
    item = _result(backend, device, suite, test, precision, None, "")
    item["details"] = {"reason": f"{type(error).__name__}: {error}"}
    return item


def _profile_value(profile: str, quick: Any, standard: Any, extended: Any) -> Any:
    return {"quick": quick, "standard": standard, "extended": extended}[profile]


def _auto_matrix_size(memory_bytes: int, profile: str) -> int:
    if profile == "quick":
        return 2048
    if memory_bytes >= 12 * 1024**3:
        return 8192 if profile == "extended" else 4096
    if memory_bytes >= 5 * 1024**3:
        return 4096 if profile == "extended" else 3072
    return 2048


def _duration(profile: str, requested: float | None) -> float:
    if requested is not None:
        return requested
    return _profile_value(profile, 0.35, 1.5, 4.0)


@contextmanager
def _matmul_mode(torch: Any, backend: str, tf32: bool):
    matmul = torch.backends.cuda.matmul
    cudnn = torch.backends.cudnn
    original_matmul = getattr(matmul, "allow_tf32", None)
    original_cudnn = getattr(cudnn, "allow_tf32", None)
    if backend == "cuda" and original_matmul is not None:
        matmul.allow_tf32 = tf32
    if backend == "cuda" and original_cudnn is not None:
        cudnn.allow_tf32 = tf32
    try:
        yield
    finally:
        if backend == "cuda" and original_matmul is not None:
            matmul.allow_tf32 = original_matmul
        if backend == "cuda" and original_cudnn is not None:
            cudnn.allow_tf32 = original_cudnn


def _time_gpu_operation(torch: Any, operation: Callable[[], Any], duration: float, warmup: int) -> tuple[float, int]:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()

    pilot_runs = 5
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(pilot_runs):
        operation()
    end.record()
    torch.cuda.synchronize()
    pilot_seconds = max(start.elapsed_time(end) / 1000.0, 1e-6)
    runs = max(10, min(10000, math.ceil(duration / pilot_seconds * pilot_runs)))

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(runs):
        operation()
    end.record()
    torch.cuda.synchronize()
    elapsed_seconds = start.elapsed_time(end) / 1000.0
    return elapsed_seconds, runs


class SyntheticCNN:
    def __new__(cls, torch: Any) -> Any:
        nn = torch.nn
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 1000),
        )


def run_gpu_benchmarks(
    backend: str,
    device_indices: list[int],
    suites: set[str],
    profile: str,
    requested_duration: float | None,
    matrix_size: int,
    batch_size: int,
    warmup: int,
) -> list[dict[str, Any]]:
    import torch

    results: list[dict[str, Any]] = []
    benchmark_duration = _duration(profile, requested_duration)
    for index in device_indices:
        torch.cuda.set_device(index)
        properties = torch.cuda.get_device_properties(index)
        device = f"{index}: {properties.name}"
        size = matrix_size or _auto_matrix_size(properties.total_memory, profile)
        active_batch = batch_size or _profile_value(profile, 2, 8, 16)

        if "compute" in suites:
            results.extend(_run_matmul(torch, backend, device, size, benchmark_duration, warmup))
        if "memory" in suites:
            results.append(_run_memory(torch, backend, device, properties.total_memory, profile, benchmark_duration, warmup))
        if "inference" in suites:
            results.extend(_run_inference(torch, backend, device, active_batch, benchmark_duration, warmup))

        torch.cuda.empty_cache()
        gc.collect()
    return results


def _run_matmul(
    torch: Any,
    backend: str,
    device: str,
    size: int,
    duration: float,
    warmup: int,
) -> list[dict[str, Any]]:
    cases = [("fp32", torch.float32, False)]
    if backend == "cuda":
        cases.append(("tf32", torch.float32, True))
    cases.extend([("fp16", torch.float16, False), ("bf16", torch.bfloat16, False)])
    results = []
    for label, dtype, tf32 in cases:
        try:
            with _matmul_mode(torch, backend, tf32), torch.inference_mode():
                left = torch.randn((size, size), device="cuda", dtype=dtype)
                right = torch.randn((size, size), device="cuda", dtype=dtype)
                output = None

                def operation() -> None:
                    nonlocal output
                    output = torch.mm(left, right)

                elapsed, runs = _time_gpu_operation(torch, operation, duration, warmup)
                operations = 2.0 * size**3 * runs
                tflops = operations / elapsed / 1e12
                results.append(
                    _result(
                        backend,
                        device,
                        "compute",
                        "dense_matmul",
                        label,
                        tflops,
                        "TFLOP/s",
                        matrix_size=size,
                        iterations=runs,
                        mean_ms=elapsed * 1000.0 / runs,
                    )
                )
                del left, right, output
        except (RuntimeError, TypeError) as exc:
            results.append(_skipped(backend, device, "compute", "dense_matmul", label, exc))
        finally:
            torch.cuda.empty_cache()
    return results


def _run_memory(
    torch: Any,
    backend: str,
    device: str,
    total_memory: int,
    profile: str,
    duration: float,
    warmup: int,
) -> dict[str, Any]:
    try:
        cap = _profile_value(profile, 64, 256, 512) * 1024**2
        tensor_bytes = min(cap, max(16 * 1024**2, total_memory // 32))
        elements = tensor_bytes // 4
        source = torch.empty(elements, device="cuda", dtype=torch.float32).normal_()
        destination = torch.empty_like(source)

        def operation() -> None:
            destination.copy_(source)

        elapsed, runs = _time_gpu_operation(torch, operation, duration, warmup)
        transferred = 2.0 * source.numel() * source.element_size() * runs
        return _result(
            backend,
            device,
            "memory",
            "device_copy",
            "fp32",
            transferred / elapsed / 1e9,
            "GB/s",
            buffer_mib=source.numel() * source.element_size() / 1024**2,
            iterations=runs,
            mean_ms=elapsed * 1000.0 / runs,
        )
    except RuntimeError as exc:
        return _skipped(backend, device, "memory", "device_copy", "fp32", exc)


def _run_inference(
    torch: Any,
    backend: str,
    device: str,
    batch_size: int,
    duration: float,
    warmup: int,
) -> list[dict[str, Any]]:
    cases = [("fp32", torch.float32)]
    if backend == "cuda":
        cases[0] = ("tf32", torch.float32)
    cases.extend([("fp16", torch.float16), ("bf16", torch.bfloat16)])
    results = []
    for label, dtype in cases:
        try:
            tf32 = label == "tf32"
            with _matmul_mode(torch, backend, tf32), torch.inference_mode():
                model = SyntheticCNN(torch).eval().to(device="cuda", dtype=dtype)
                inputs = torch.randn((batch_size, 3, 224, 224), device="cuda", dtype=dtype)
                output = None

                def operation() -> None:
                    nonlocal output
                    output = model(inputs)

                elapsed, runs = _time_gpu_operation(torch, operation, duration, warmup)
                images_per_second = batch_size * runs / elapsed
                results.append(
                    _result(
                        backend,
                        device,
                        "inference",
                        "synthetic_cnn",
                        label,
                        images_per_second,
                        "images/s",
                        batch_size=batch_size,
                        iterations=runs,
                        mean_batch_ms=elapsed * 1000.0 / runs,
                        mean_image_ms=elapsed * 1000.0 / (runs * batch_size),
                    )
                )
                del model, inputs, output
        except (RuntimeError, TypeError) as exc:
            results.append(_skipped(backend, device, "inference", "synthetic_cnn", label, exc))
        finally:
            torch.cuda.empty_cache()
    return results


def run_cpu_benchmarks(
    suites: set[str],
    profile: str,
    requested_duration: float | None,
    matrix_size: int,
    warmup: int,
) -> list[dict[str, Any]]:
    import numpy as np

    duration = _duration(profile, requested_duration)
    size = matrix_size or _profile_value(profile, 512, 1024, 2048)
    device = "CPU / NumPy"
    results = []
    if "compute" in suites:
        left = np.random.default_rng(7).standard_normal((size, size), dtype=np.float32)
        right = np.random.default_rng(8).standard_normal((size, size), dtype=np.float32)
        for _ in range(warmup):
            np.matmul(left, right)
        runs = 0
        started = time.perf_counter()
        while runs < 3 or time.perf_counter() - started < duration:
            np.matmul(left, right)
            runs += 1
        elapsed = time.perf_counter() - started
        results.append(
            _result(
                "cpu",
                device,
                "compute",
                "dense_matmul",
                "fp32",
                2.0 * size**3 * runs / elapsed / 1e12,
                "TFLOP/s",
                matrix_size=size,
                iterations=runs,
                mean_ms=elapsed * 1000.0 / runs,
            )
        )
    if "memory" in suites:
        megabytes = _profile_value(profile, 32, 128, 256)
        source = np.empty(megabytes * 1024**2 // 4, dtype=np.float32)
        destination = np.empty_like(source)
        runs = 0
        started = time.perf_counter()
        while runs < 5 or time.perf_counter() - started < duration:
            np.copyto(destination, source)
            runs += 1
        elapsed = time.perf_counter() - started
        transferred = 2.0 * source.nbytes * runs
        results.append(
            _result(
                "cpu",
                device,
                "memory",
                "host_copy",
                "fp32",
                transferred / elapsed / 1e9,
                "GB/s",
                buffer_mib=source.nbytes / 1024**2,
                iterations=runs,
            )
        )
    return results
