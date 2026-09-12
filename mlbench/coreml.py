"""Static FP16 Core ML CNN with explicit compute-unit and placement reporting."""
from __future__ import annotations

import concurrent.futures
import hashlib
import platform
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .apple import apple_hardware
from .gpu import _result
from .power import add_power_details, idle_result
from .stats import mean, percentile


def _make_model(path: Path, batch: int) -> None:
    import coremltools as ct
    import numpy as np
    from coremltools.converters.mil import Builder as mb

    # Build MIL directly: no dependency on the PyTorch-to-Core-ML converter version.
    rng = np.random.default_rng(2026)

    @mb.program(input_specs=[mb.TensorSpec(shape=(batch, 3, 224, 224))], opset_version=ct.target.macOS13)
    def program(inputs):
        x = inputs
        channels = [3, 64, 128, 256, 256]
        for cin, cout in zip(channels, channels[1:]):
            weights = rng.standard_normal((cout, cin, 3, 3), dtype=np.float32) * 0.02
            x = mb.conv(x=x, weight=weights, bias=np.zeros(cout, np.float32),
                        strides=[2, 2], pad_type="custom", pad=[1, 1, 1, 1])
            x = mb.relu(x=x)
        x = mb.reduce_mean(x=x, axes=[2, 3], keep_dims=False)
        weights = rng.standard_normal((1000, 256), dtype=np.float32) * 0.02
        return mb.linear(x=x, weight=weights, bias=np.zeros(1000, np.float32), name="output")

    model = ct.convert(program, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS13,
                       compute_precision=ct.precision.FLOAT16, skip_model_load=True)
    model.save(str(path))


def _placement(ct: Any, compiled: Path, units: Any) -> dict[str, Any]:
    try:
        plan = ct.models.compute_plan.MLComputePlan.load_from_path(str(compiled), compute_units=units)
        counts: Counter[str] = Counter()
        def visit(block):
            for operation in block.operations:
                usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
                name = type(usage.preferred_compute_device).__name__ if usage else "unknown"
                counts[name] += 1
                for child in operation.blocks:
                    visit(child)
        for function in plan.model_structure.program.functions.values():
            visit(function.block)
        return {
            "status": "ok", "preferred_operation_counts": dict(counts),
            "neural_engine_planned": any("NeuralEngine" in name for name in counts),
            "note": "Core ML 编译计划的首选设备；不是实测占用率，也不代表纯 NPU 时间",
        }
    except Exception as exc:
        return {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}",
                "neural_engine_planned": None}


def run_coreml_benchmarks(output_dir: str, profile: str, requested_duration: float | None,
                         batch_size: int, warmup: int, streams: int,
                         compute_units: str, power_enabled: bool) -> list[dict[str, Any]]:
    import coremltools as ct
    import numpy as np

    batch = batch_size or 1
    duration = requested_duration if requested_duration is not None else {
        "quick": 0.5, "standard": 3.0, "extended": 10.0,
    }[profile]
    key = hashlib.sha256(f"cnn-v1-b{batch}-ct{ct.__version__}-{platform.mac_ver()[0]}".encode()).hexdigest()[:16]
    cache = Path(output_dir) / "cache" / "coreml" / key
    cache.mkdir(parents=True, exist_ok=True)
    package = cache / "synthetic_cnn.mlpackage"
    compiled = cache / "synthetic_cnn.mlmodelc"
    cached = compiled.is_dir()
    started = time.perf_counter()
    if not package.exists():
        _make_model(package, batch)
    conversion_seconds = time.perf_counter() - started if not cached else 0.0
    started = time.perf_counter()
    if not compiled.exists():
        ct.models.utils.compile_model(str(package), destination_path=str(compiled))
    compile_seconds = time.perf_counter() - started if not cached else 0.0
    modes = ["cpu_and_ne", "all"] if compute_units == "compare" else [compute_units]
    results = []
    feed = {"inputs": np.random.default_rng(9).standard_normal((batch, 3, 224, 224), dtype=np.float32)}
    for mode in modes:
        units = getattr(ct.ComputeUnit, mode.upper())
        device = f"{apple_hardware()['name']} / Core ML {mode.upper()}"
        started = time.perf_counter()
        # Independent model instances let concurrent requests use the Core ML scheduler.
        models = [ct.models.CompiledMLModel(str(compiled), compute_units=units) for _ in range(streams)]
        load_seconds = time.perf_counter() - started
        placement = _placement(ct, compiled, units)
        for model in models:
            output = model.predict(feed)["output"]
            if output.shape != (batch, 1000) or not np.isfinite(output).all():
                raise RuntimeError("Core ML CNN 输出形状或数值无效")
            for _ in range(warmup):
                model.predict(feed)
        if power_enabled:
            results.append(idle_result("coreml", device, None))

        def worker(model):
            latencies = []
            while len(latencies) < 3 or time.perf_counter() - started < duration:
                run_started = time.perf_counter()
                model.predict(feed)
                latencies.append((time.perf_counter() - run_started) * 1000)
            return latencies

        with concurrent.futures.ThreadPoolExecutor(max_workers=streams) as executor:
            started = time.perf_counter()
            futures = [executor.submit(worker, model) for model in models]
            latencies = [latency for future in futures for latency in future.result()]
            elapsed = time.perf_counter() - started
        details = dict(batch_size=batch, streams=streams, iterations=len(latencies),
                       compute_units=mode.upper(), compile_seconds=compile_seconds,
                       conversion_seconds=conversion_seconds, load_seconds=load_seconds,
                       cache_hit=cached, model=str(package), placement=placement,
                       cpu_fallback_possible=True, timing="synchronous_predict_wall_clock",
                       mean_ms=mean(latencies), p95_ms=percentile(latencies, 95))
        for test, value, unit in [
            ("synthetic_cnn", len(latencies) * batch / elapsed, "images/s"),
            ("synthetic_cnn_latency", percentile(latencies, 50), "ms"),
        ]:
            item = _result("coreml", device, "inference", test, "fp16/fp32-io", value, unit, **details)
            add_power_details(item, None)
            results.append(item)
    return results
