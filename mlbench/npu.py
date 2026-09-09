from __future__ import annotations

import concurrent.futures
import os
import time
from pathlib import Path
from typing import Any

from .stats import mean, percentile


def _make_model(path: Path, batch_size: int) -> None:
    try:
        import onnx
        from onnx import TensorProto, helper, numpy_helper
    except ImportError as exc:
        raise RuntimeError("生成 NPU 测试模型需要 onnx 包；请在 Ryzen AI 环境中安装 onnx") from exc

    import numpy as np

    rng = np.random.default_rng(2026)
    nodes = []
    initializers = []
    channels = [3, 64, 128, 256, 256]
    current = "input"
    for layer, (input_channels, output_channels) in enumerate(zip(channels, channels[1:]), start=1):
        weight_name = f"conv{layer}_weight"
        bias_name = f"conv{layer}_bias"
        conv_output = f"conv{layer}_output"
        relu_output = f"relu{layer}_output"
        weights = rng.standard_normal((output_channels, input_channels, 3, 3), dtype=np.float32) * 0.02
        bias = np.zeros((output_channels,), dtype=np.float32)
        initializers.extend(
            [numpy_helper.from_array(weights, weight_name), numpy_helper.from_array(bias, bias_name)]
        )
        nodes.append(
            helper.make_node(
                "Conv",
                [current, weight_name, bias_name],
                [conv_output],
                kernel_shape=[3, 3],
                pads=[1, 1, 1, 1],
                strides=[2, 2],
            )
        )
        nodes.append(helper.make_node("Relu", [conv_output], [relu_output]))
        current = relu_output

    linear_weight = rng.standard_normal((256, 1000), dtype=np.float32) * 0.02
    linear_bias = np.zeros((1000,), dtype=np.float32)
    initializers.extend(
        [
            numpy_helper.from_array(linear_weight, "linear_weight"),
            numpy_helper.from_array(linear_bias, "linear_bias"),
        ]
    )
    nodes.extend(
        [
            helper.make_node("GlobalAveragePool", [current], ["pool_output"]),
            helper.make_node("Flatten", ["pool_output"], ["flat_output"], axis=1),
            helper.make_node("Gemm", ["flat_output", "linear_weight", "linear_bias"], ["output"]),
        ]
    )
    graph = helper.make_graph(
        nodes,
        "mlbench_synthetic_cnn",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [batch_size, 3, 224, 224])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [batch_size, 1000])],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="hetero-mlbench",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = min(model.ir_version, 9)
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)


def run_npu_benchmarks(
    output_dir: Path,
    profile: str,
    requested_duration: float | None,
    batch_size: int,
    warmup: int,
    streams: int,
    config_file: str | None,
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
    model_path = output_dir / "models" / f"synthetic_cnn_b{active_batch}.onnx"
    if not model_path.exists():
        _make_model(model_path, active_batch)

    provider_options = {
        "cache_dir": str(output_dir / "cache"),
        "cache_key": f"mlbench_synthetic_cnn_b{active_batch}",
    }
    if config_file:
        config_path = Path(config_file).expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"NPU 配置文件不存在: {config_path}")
        provider_options["config_file"] = str(config_path)
    os.makedirs(provider_options["cache_dir"], exist_ok=True)

    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3
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
    inputs = rng.standard_normal((active_batch, 3, 224, 224), dtype=np.float32)
    feed = {session.get_inputs()[0].name: inputs}
    for _ in range(warmup):
        session.run(None, feed)

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
    iterations = len(latencies)
    throughput = iterations * active_batch / elapsed

    common = {
        "backend": "npu",
        "device": "AMD NPU / VitisAIExecutionProvider",
        "suite": "inference",
        "test": "synthetic_cnn",
        "precision": "fp32-input/auto",
        "status": "ok",
        "details": {
            "batch_size": active_batch,
            "streams": streams,
            "iterations": iterations,
            "compile_seconds": compile_seconds,
            "providers": session.get_providers(),
            "model": str(model_path),
        },
    }
    throughput_result = dict(common, value=throughput, unit="images/s")
    latency_result = dict(common, test="synthetic_cnn_latency", value=percentile(latencies, 50), unit="ms")
    latency_result["details"] = dict(
        common["details"],
        mean_ms=mean(latencies),
        p95_ms=percentile(latencies, 95),
    )
    return [throughput_result, latency_result]

