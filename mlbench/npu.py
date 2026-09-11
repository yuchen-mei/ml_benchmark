from __future__ import annotations

import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from .power import PowerSampler, add_power_details, idle_result, make_power_reader, sample_idle_power
from .stats import mean, percentile


_ORT_DTYPES = {
    "tensor(bool)": "bool",
    "tensor(double)": "float64",
    "tensor(float)": "float32",
    "tensor(float16)": "float16",
    "tensor(int8)": "int8",
    "tensor(int16)": "int16",
    "tensor(int32)": "int32",
    "tensor(int64)": "int64",
    "tensor(uint8)": "uint8",
    "tensor(uint16)": "uint16",
    "tensor(uint32)": "uint32",
    "tensor(uint64)": "uint64",
}


def _quicktest_model(model_file: str | None) -> tuple[Path, str]:
    if model_file:
        path = Path(model_file).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"NPU 模型不存在: {path}")
        return path, "custom"
    roots = []
    installation_path = os.environ.get("RYZEN_AI_INSTALLATION_PATH")
    if installation_path:
        roots.append(Path(installation_path).expanduser() / "quicktest")
    roots.append(Path(sys.prefix) / "quicktest")
    for root in _unique_paths(roots):
        if not root.is_dir():
            continue
        candidates = sorted(root.glob("*.onnx")) or sorted(root.rglob("*.onnx"))
        if candidates:
            return candidates[0].resolve(), "ryzen_ai_quicktest"
    searched = ", ".join(str(root) for root in _unique_paths(roots))
    raise FileNotFoundError(
        "未找到 Ryzen AI 安装包自带的 quicktest ONNX 模型；"
        f"已搜索: {searched}。可用 --npu-model 显式指定。"
    )


def _unique_paths(paths: list[Path]) -> list[Path]:
    unique = []
    for path in paths:
        if path not in unique:
            unique.append(path)
    return unique


def _make_feed(inputs: list[Any], requested_batch: int, rng: Any) -> tuple[dict[str, Any], int]:
    import numpy as np

    feed = {}
    effective_batch = None
    for metadata in inputs:
        dtype_name = _ORT_DTYPES.get(metadata.type)
        if dtype_name is None:
            raise RuntimeError(f"NPU quicktest 模型含不支持的输入类型: {metadata.name}={metadata.type}")
        shape = []
        for index, dimension in enumerate(metadata.shape):
            if isinstance(dimension, int) and dimension > 0:
                value = dimension
            else:
                value = requested_batch if index == 0 else 1
            shape.append(value)
        if shape and effective_batch is None:
            effective_batch = shape[0]
        dtype = np.dtype(dtype_name)
        if np.issubdtype(dtype, np.floating):
            value = rng.standard_normal(shape).astype(dtype)
        elif np.issubdtype(dtype, np.bool_):
            value = rng.integers(0, 2, size=shape).astype(dtype)
        else:
            value = rng.integers(0, 2, size=shape, dtype=dtype)
        feed[metadata.name] = value
    return feed, effective_batch or requested_batch


def run_npu_benchmarks(
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
    import numpy as np
    import onnxruntime as ort

    if "VitisAIExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("当前 ONNX Runtime 没有 VitisAIExecutionProvider")

    active_batch = batch_size or 1
    duration = requested_duration if requested_duration is not None else {
        "quick": 0.5,
        "standard": 3.0,
        "extended": 10.0,
    }[profile]
    model_path, model_source = _quicktest_model(model_file)

    provider_options = {
        "cache_dir": str(output_dir / "cache"),
        "cache_key": f"mlbench_{model_path.stem}_b{active_batch}",
    }
    if config_file:
        config_path = Path(config_file).expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"NPU 配置文件不存在: {config_path}")
        provider_options["config_file"] = str(config_path)
    Path(provider_options["cache_dir"]).mkdir(parents=True, exist_ok=True)

    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3
    profile_dir = output_dir / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    session_options.enable_profiling = True
    session_options.profile_file_prefix = str(profile_dir / "npu_profile")
    compile_started = time.perf_counter()
    session = ort.InferenceSession(
        str(model_path),
        sess_options=session_options,
        providers=["VitisAIExecutionProvider", "CPUExecutionProvider"],
        provider_options=[provider_options, {}],
    )
    compile_seconds = time.perf_counter() - compile_started
    if not session.get_providers() or session.get_providers()[0] != "VitisAIExecutionProvider":
        raise RuntimeError(f"VitisAI EP 未成为首选执行后端: {session.get_providers()}")

    rng = np.random.default_rng(9)
    feed, active_batch = _make_feed(session.get_inputs(), active_batch, rng)
    for _ in range(warmup):
        session.run(None, feed)

    device = "AMD NPU / VitisAIExecutionProvider"
    power_reader = make_power_reader("npu") if power_enabled else None
    results = []
    if power_enabled:
        results.append(idle_result("npu", device, sample_idle_power(power_reader, power_interval)))

    sampler = PowerSampler(power_reader, power_interval)
    sampler.start()
    started = time.perf_counter()
    latencies: list[float] = []

    def worker() -> list[float]:
        worker_latencies = []
        while len(worker_latencies) < 3 or time.perf_counter() - started < duration:
            run_started = time.perf_counter()
            session.run(None, feed)
            worker_latencies.append((time.perf_counter() - run_started) * 1000.0)
        return worker_latencies

    with concurrent.futures.ThreadPoolExecutor(max_workers=streams) as executor:
        futures = [executor.submit(worker) for _ in range(streams)]
        for future in futures:
            latencies.extend(future.result())
    elapsed = time.perf_counter() - started
    power = sampler.stop(elapsed)
    profile_path = Path(session.end_profiling())
    provider_counts = _profile_provider_counts(profile_path)
    if provider_counts.get("VitisAIExecutionProvider", 0) < 1:
        raise RuntimeError(
            "VitisAIExecutionProvider 已注册，但没有任何模型节点实际卸载到 NPU；"
            f"profile={profile_path}, providers={provider_counts}"
        )
    iterations = len(latencies)
    throughput = iterations * active_batch / elapsed

    common = {
        "backend": "npu",
        "device": device,
        "suite": "inference",
        "test": "ryzen_ai_quicktest_cnn",
        "precision": "int8/auto" if model_source == "ryzen_ai_quicktest" else "model/auto",
        "status": "ok",
        "details": {
            "batch_size": active_batch,
            "streams": streams,
            "iterations": iterations,
            "compile_seconds": compile_seconds,
            "providers": session.get_providers(),
            "profile_provider_counts": provider_counts,
            "profile": str(profile_path),
            "model": str(model_path),
            "model_source": model_source,
        },
    }
    throughput_result = dict(common, value=throughput, unit="images/s")
    latency_result = dict(
        common,
        test="ryzen_ai_quicktest_cnn_latency",
        value=percentile(latencies, 50),
        unit="ms",
    )
    latency_result["details"] = dict(
        common["details"],
        mean_ms=mean(latencies),
        p95_ms=percentile(latencies, 95),
    )
    add_power_details(throughput_result, power)
    add_power_details(latency_result, power)
    results.extend([throughput_result, latency_result])
    return results


def _profile_provider_counts(path: Path) -> dict[str, int]:
    try:
        events = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"无法读取 ONNX Runtime profiling 结果: {path}: {exc}") from exc
    counts: dict[str, int] = {}
    for event in events:
        provider = event.get("args", {}).get("provider")
        if provider:
            counts[provider] = counts.get(provider, 0) + 1
    return counts
