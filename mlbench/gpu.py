from __future__ import annotations

import gc
import math
import time
from contextlib import contextmanager
from typing import Any, Callable

from .power import PowerReader, PowerSampler, add_power_details, idle_result, make_power_reader, sample_idle_power


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


def _time_gpu_operation(
    torch: Any,
    operation: Callable[[], Any],
    duration: float,
    warmup: int,
    power_reader: PowerReader | None,
    power_interval: float,
) -> tuple[float, int, dict[str, Any] | None]:
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

    sampler = PowerSampler(power_reader, power_interval)
    sampler.start()
    elapsed_seconds = 0.0
    total_runs = 0
    chunk_runs = runs
    for _ in range(4):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(chunk_runs):
            operation()
        end.record()
        torch.cuda.synchronize()
        chunk_seconds = start.elapsed_time(end) / 1000.0
        elapsed_seconds += chunk_seconds
        total_runs += chunk_runs
        remaining = duration - elapsed_seconds
        if remaining <= 0:
            break
        chunk_runs = max(
            10,
            min(10000, math.ceil(chunk_runs * remaining / max(chunk_seconds, 1e-6) * 1.1)),
        )
    power = sampler.stop(elapsed_seconds)
    return elapsed_seconds, total_runs, power


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


class SyntheticMLP:
    def __new__(cls, torch: Any) -> Any:
        nn = torch.nn
        return nn.Sequential(
            nn.Linear(4096, 8192),
            nn.ReLU(inplace=True),
            nn.Linear(8192, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1000),
        )


def _is_gfx1151(backend: str, architecture: str | None) -> bool:
    normalized = (architecture or "").lower().split(":", 1)[0]
    return backend == "rocm" and normalized == "gfx1151"


def run_gpu_benchmarks(
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
    device_architecture: str | None = None,
    precisions: set[str] | None = None,
    include_idle: bool = True,
) -> list[dict[str, Any]]:
    import torch

    results: list[dict[str, Any]] = []
    benchmark_duration = _duration(profile, requested_duration)
    for index in device_indices:
        torch.cuda.set_device(index)
        properties = torch.cuda.get_device_properties(index)
        device = f"{index}: {properties.name}"
        power_reader = make_power_reader(backend, index) if power_enabled else None
        size = matrix_size or _auto_matrix_size(properties.total_memory, profile)
        active_batch = batch_size or _profile_value(profile, 2, 8, 16)

        if power_enabled and include_idle:
            results.append(idle_result(backend, device, sample_idle_power(power_reader, power_interval)))

        if "compute" in suites:
            results.extend(
                _run_matmul(
                    torch,
                    backend,
                    device,
                    size,
                    benchmark_duration,
                    warmup,
                    power_reader,
                    power_interval,
                    precisions,
                )
            )
        if "memory" in suites:
            results.append(
                _run_memory(
                    torch,
                    backend,
                    device,
                    properties.total_memory,
                    profile,
                    benchmark_duration,
                    warmup,
                    power_reader,
                    power_interval,
                )
            )
        if "inference" in suites:
            results.extend(
                _run_inference(
                    torch,
                    backend,
                    device,
                    active_batch,
                    benchmark_duration,
                    warmup,
                    power_reader,
                    power_interval,
                    device_architecture,
                    precisions,
                )
            )

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
    power_reader: PowerReader | None,
    power_interval: float,
    precisions: set[str] | None,
) -> list[dict[str, Any]]:
    cases = [("fp32", torch.float32, False)]
    if backend == "cuda":
        cases.append(("tf32", torch.float32, True))
    cases.extend([("fp16", torch.float16, False), ("bf16", torch.bfloat16, False)])
    if precisions is not None:
        cases = [case for case in cases if case[0] in precisions]
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

                elapsed, runs, power = _time_gpu_operation(
                    torch, operation, duration, warmup, power_reader, power_interval
                )
                operations = 2.0 * size**3 * runs
                tflops = operations / elapsed / 1e12
                result = _result(
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
                add_power_details(result, power)
                results.append(result)
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
    power_reader: PowerReader | None,
    power_interval: float,
) -> dict[str, Any]:
    try:
        cap = _profile_value(profile, 64, 256, 512) * 1024**2
        tensor_bytes = min(cap, max(16 * 1024**2, total_memory // 32))
        elements = tensor_bytes // 4
        source = torch.empty(elements, device="cuda", dtype=torch.float32).normal_()
        destination = torch.empty_like(source)

        def operation() -> None:
            destination.copy_(source)

        elapsed, runs, power = _time_gpu_operation(
            torch, operation, duration, warmup, power_reader, power_interval
        )
        transferred = 2.0 * source.numel() * source.element_size() * runs
        result = _result(
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
        add_power_details(result, power)
        return result
    except RuntimeError as exc:
        return _skipped(backend, device, "memory", "device_copy", "fp32", exc)


def _run_inference(
    torch: Any,
    backend: str,
    device: str,
    batch_size: int,
    duration: float,
    warmup: int,
    power_reader: PowerReader | None,
    power_interval: float,
    device_architecture: str | None,
    precisions: set[str] | None,
) -> list[dict[str, Any]]:
    cases = [("fp32", torch.float32)]
    if backend == "cuda":
        cases[0] = ("tf32", torch.float32)
    cases.extend([("fp16", torch.float16), ("bf16", torch.bfloat16)])
    if precisions is not None:
        cases = [case for case in cases if case[0] in precisions]
    results = []
    compatibility_fallback = _is_gfx1151(backend, device_architecture)
    for label, dtype in cases:
        try:
            tf32 = label == "tf32"
            with _matmul_mode(torch, backend, tf32), torch.inference_mode():
                if compatibility_fallback:
                    model = SyntheticMLP(torch).eval().to(device="cuda", dtype=dtype)
                    inputs = torch.randn((batch_size, 4096), device="cuda", dtype=dtype)
                    test = "synthetic_mlp"
                    unit = "samples/s"
                else:
                    model = SyntheticCNN(torch).eval().to(device="cuda", dtype=dtype)
                    inputs = torch.randn((batch_size, 3, 224, 224), device="cuda", dtype=dtype)
                    test = "synthetic_cnn"
                    unit = "images/s"
                output = None

                def operation() -> None:
                    nonlocal output
                    output = model(inputs)

                elapsed, runs, power = _time_gpu_operation(
                    torch, operation, duration, warmup, power_reader, power_interval
                )
                samples_per_second = batch_size * runs / elapsed
                timing_details = {
                    "batch_size": batch_size,
                    "iterations": runs,
                    "mean_batch_ms": elapsed * 1000.0 / runs,
                }
                timing_key = "mean_sample_ms" if compatibility_fallback else "mean_image_ms"
                timing_details[timing_key] = elapsed * 1000.0 / (runs * batch_size)
                result = _result(
                    backend,
                    device,
                    "inference",
                    test,
                    label,
                    samples_per_second,
                    unit,
                    **timing_details,
                )
                if compatibility_fallback:
                    result["details"]["compatibility_fallback"] = (
                        f"{device_architecture}: 使用 MLP 避开已知 MIOpen Conv2d 原生崩溃"
                    )
                add_power_details(result, power)
                results.append(result)
                del model, inputs, output
        except (RuntimeError, TypeError) as exc:
            test = "synthetic_mlp" if compatibility_fallback else "synthetic_cnn"
            results.append(_skipped(backend, device, "inference", test, label, exc))
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
