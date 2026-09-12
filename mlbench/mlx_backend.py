"""Native Metal workloads and causal LLM generation using MLX."""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any, Callable

from .apple import apple_hardware, unified_memory_budget
from .gpu import _auto_matrix_size, _duration, _profile_value, _result
from .llm import LLM_PRESETS, generation_metrics
from .power import add_power_details, idle_result


def _setup(mx: Any, memory_limit_gib: float = 0) -> tuple[str, int]:
    if not mx.metal.is_available():
        raise RuntimeError("MLX Metal GPU 不可用")
    mx.set_default_device(mx.gpu)
    hardware = apple_hardware()
    info = mx.device_info()
    budget = unified_memory_budget(hardware["unified_memory_bytes"], int(info["max_recommended_working_set_size"]), memory_limit_gib)
    mx.set_memory_limit(budget)
    mx.set_cache_limit(min(budget // 8, 4 * 1024**3))
    return f"0: {hardware['name']} / MLX Metal", budget


def _measure(mx: Any, operation: Callable, duration: float, warmup: int) -> tuple[float, int]:
    for _ in range(max(1, warmup)):
        mx.eval(operation())
    mx.synchronize()
    runs = 0
    started = time.perf_counter()
    while runs < 3 or time.perf_counter() - started < duration:
        # Each operation creates a fresh graph; eval prevents timing lazy graph construction.
        for _ in range(8):
            mx.async_eval(operation())
            runs += 1
        mx.synchronize()
    return time.perf_counter() - started, runs


def run_mlx_benchmarks(suites: list[str], profile: str, requested_duration: float | None,
                       matrix_size: int, batch_size: int, warmup: int,
                       power_enabled: bool, memory_limit_gib: float = 0) -> list[dict[str, Any]]:
    import mlx.core as mx
    import mlx.nn as nn

    device, budget = _setup(mx, memory_limit_gib)
    duration = _duration(profile, requested_duration)
    size = matrix_size or _auto_matrix_size(budget, profile)
    batch = batch_size or _profile_value(profile, 2, 8, 16)
    results = [idle_result("mlx", device, None)] if power_enabled else []
    mx.random.seed(2026)
    for label, dtype in [("fp32", mx.float32), ("fp16", mx.float16), ("bf16", mx.bfloat16)]:
        if "compute" in suites:
            left = mx.random.normal((size, size)).astype(dtype)
            right = mx.random.normal((size, size)).astype(dtype)
            mx.eval(left, right)
            _check_memory(mx, budget)
            elapsed, runs = _measure(mx, lambda: left @ right, duration, warmup)
            _check_memory(mx, budget)
            results.append(_result("mlx", device, "compute", "dense_matmul", label,
                                   2 * size**3 * runs / elapsed / 1e12, "TFLOP/s",
                                   matrix_size=size, iterations=runs, mean_ms=elapsed * 1000 / runs))
            del left, right
        if "inference" in suites:
            model = nn.Sequential(
                nn.Conv2d(3, 64, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(256, 256, 3, stride=2, padding=1), nn.ReLU(),
                lambda x: mx.mean(x, axis=(1, 2)), nn.Linear(256, 1000),
            )
            model.set_dtype(dtype)
            model.eval()
            inputs = mx.random.normal((batch, 224, 224, 3)).astype(dtype)
            mx.eval(model.parameters(), inputs)
            forward = mx.compile(model)
            elapsed, runs = _measure(mx, lambda: forward(inputs), duration, warmup)
            _check_memory(mx, budget)
            results.append(_result("mlx", device, "inference", "synthetic_cnn", label,
                                   batch * runs / elapsed, "images/s", batch_size=batch,
                                   iterations=runs, mean_batch_ms=elapsed * 1000 / runs,
                                   layout="NHWC", compiled=True))
            del model, inputs, forward
        mx.clear_cache()
    if "memory" in suites:
        # MLX arrays may alias on copy. Force a real layout conversion instead.
        side = _profile_value(profile, 4096, 8192, 8192)
        source = mx.random.normal((side, side))
        mx.eval(source)
        elapsed, runs = _measure(mx, lambda: mx.contiguous(source.T), duration, warmup)
        _check_memory(mx, budget)
        results.append(_result("mlx", device, "memory", "device_transpose", "fp32",
                               2 * source.nbytes * runs / elapsed / 1e9, "GB/s",
                               buffer_mib=source.nbytes / 1024**2, iterations=runs,
                               note="转置并物化连续缓冲区；与 device_copy 是不同负载"))
    for item in results:
        item["details"].update(memory_type="unified", memory_budget_bytes=budget,
                               timing="synchronized_wall_clock", cpu_fallback=False)
        add_power_details(item, None)
    return results


def _validate_model_code(config: dict, model_path: Path, trust_remote_code: bool) -> None:
    model_file = config.get("model_file")
    if model_file:
        if not trust_remote_code:
            raise ValueError("MLX 模型包含自定义 Python 代码；需要显式 --trust-remote-code")
        if not (model_path / model_file).resolve().is_relative_to(model_path.resolve()):
            raise ValueError("MLX model_file 必须位于模型目录内")


def _check_memory(mx: Any, budget: int) -> None:
    # MLX's allocator limit is advisory, so enforce the benchmark budget ourselves.
    if mx.get_active_memory() > budget or mx.get_peak_memory() > budget:
        raise ValueError("MLX 内存使用超过统一内存预算；请减少 batch/token 数或使用预量化模型")


def run_mlx_llm(preset_name: str, model_override: str | None, quantization: str,
                prompt_tokens: int, new_tokens: int, batch_size: int, runs: int, warmup: int,
                trust_remote_code: bool, cache_dir: str | None, local_files_only: bool,
                power_enabled: bool, memory_limit_gib: float = 0) -> list[dict[str, Any]]:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    if preset_name == "kimi" and not model_override:
        raise ValueError("Kimi VL 是多模态模型；MLX LLM 请使用 qwen/llama/deepseek 或兼容的文本模型")
    device, budget = _setup(mx, memory_limit_gib)
    model_id = model_override or LLM_PRESETS[preset_name].model_id
    active_quantization = "4bit" if quantization == "auto" else quantization
    model_path = Path(model_id).expanduser()
    started = time.perf_counter()
    if not model_path.is_dir():
        from huggingface_hub import snapshot_download
        model_path = Path(snapshot_download(
            model_id, cache_dir=cache_dir, local_files_only=local_files_only,
            allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.tiktoken"]
            + (["*.py"] if trust_remote_code else []),
        ))
    download_seconds = time.perf_counter() - started
    config = json.loads((model_path / "config.json").read_text())
    _validate_model_code(config, model_path, trust_remote_code)
    weight_bytes = sum(path.stat().st_size for path in model_path.glob("*.safetensors"))
    if weight_bytes > budget:
        raise ValueError("模型文件超过统一内存预算；请使用已量化的 MLX 模型")
    started = time.perf_counter()
    existing_quantization = config.get("quantization") or config.get("quantization_config")
    if existing_quantization:
        actual_bits = existing_quantization.get("bits")
        if quantization != "auto" and active_quantization != f"{actual_bits}bit":
            raise ValueError("模型已有量化；请求的精度与模型不一致，请使用 --llm-quantization auto")
        precision = f"{actual_bits}bit" if actual_bits else "mixed-quantized"
    model, tokenizer = load(str(model_path), tokenizer_config={"trust_remote_code": trust_remote_code}, lazy=True)
    if not existing_quantization and active_quantization in {"fp32", "fp16", "bf16"}:
        model.set_dtype({"fp32": mx.float32, "fp16": mx.float16, "bf16": mx.bfloat16}[active_quantization])
        precision = active_quantization
    elif not existing_quantization and active_quantization in {"4bit", "8bit"}:
        nn.quantize(model, bits=int(active_quantization[0]), group_size=64)
        precision = active_quantization
    elif not existing_quantization:
        precision = "unquantized"
    mx.eval(model.parameters())
    mx.synchronize()
    model_bytes = sum(value.nbytes for _, value in tree_flatten(model.parameters()))
    parameter_dtypes = sorted({str(value.dtype) for _, value in tree_flatten(model.parameters())})
    _check_memory(mx, budget)
    load_seconds = time.perf_counter() - started
    seed = tokenizer.encode("Explain how matrix multiplication accelerates machine learning. ")
    if not seed:
        raise ValueError("Tokenizer 返回了空输入")
    tokens = (seed * ((prompt_tokens + len(seed) - 1) // len(seed)))[:prompt_tokens]
    context_limit = config.get("max_position_embeddings")
    if context_limit and prompt_tokens + new_tokens > context_limit:
        raise ValueError(f"请求 token 数超过模型上下文长度 {context_limit}")

    def request():
        mx.synchronize()
        start = time.perf_counter()
        inputs = mx.array([tokens] * batch_size)
        cache = make_prompt_cache(model)
        outputs = []
        for offset in range(0, prompt_tokens, 2048):
            chunk_end = min(offset + 2048, prompt_tokens)
            logits = model(inputs[:, offset:chunk_end], cache=cache)
            if chunk_end < prompt_tokens:
                mx.eval([entry.state for entry in cache])
                _check_memory(mx, budget)
        # Keep one KV-cache chain for prefill and decode; never reuse it across requests.
        next_token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(next_token)
        del logits
        mx.synchronize()
        first = time.perf_counter()
        _check_memory(mx, budget)
        outputs.append(next_token)
        for _ in range(new_tokens - 1):
            next_token = mx.argmax(model(next_token[:, None], cache=cache)[:, -1, :], axis=-1)
            mx.eval(next_token)
            _check_memory(mx, budget)
            outputs.append(next_token)
        mx.synchronize()
        decoded = time.perf_counter()
        output_tokens = mx.stack(outputs, axis=1).tolist()
        for row in output_tokens:
            tokenizer.decode(row)
        end = time.perf_counter()
        return first - start, decoded - first, end - start

    for _ in range(warmup):
        request()
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()
    measurements = [request() for _ in range(runs)]
    peak = mx.get_peak_memory()
    if peak > budget:
        raise ValueError("生成峰值内存超过统一内存预算")
    ttft, decode, e2e = map(list, zip(*measurements))
    metrics = generation_metrics(ttft, decode, e2e, [new_tokens * batch_size] * runs, prompt_tokens, batch_size)
    details = dict(model=model_id, batch_size=batch_size, runs=runs,
                   prompt_tokens_per_request=prompt_tokens, new_tokens_per_request=new_tokens,
                   model_vram_gib=model_bytes / 1024**3, peak_vram_gib=peak / 1024**3,
                   vram_budget_gib=budget / 1024**3, memory_type="unified",
                   download_seconds=download_seconds, load_seconds=load_seconds,
                   generation="greedy_fixed_length_mlx_kv_cache", cpu_fallback=False,
                   parameter_dtypes=parameter_dtypes,
                   prefill_chunk_tokens=2048,
                   timing="synchronized_wall_clock", **metrics)
    results = [idle_result("mlx", device, None)] if power_enabled else []
    for test, value, unit in [
        ("llm_ttft_p50", metrics["ttft_p50_ms"], "ms"),
        ("llm_prefill", metrics["prefill_tokens_per_second"], "tokens/s"),
        ("llm_decode", metrics["decode_tokens_per_second"], "tokens/s"),
        ("llm_output_e2e", metrics["output_tokens_per_second"], "tokens/s"),
        ("llm_e2e_p50", metrics["e2e_p50_seconds"], "s"),
        ("llm_peak_vram", peak / 1024**3, "GiB"),
    ]:
        result = _result("mlx", device, "llm", test, precision, value, unit, **details)
        add_power_details(result, None)
        results.append(result)
    return results
