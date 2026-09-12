from __future__ import annotations

import gc
import importlib.util
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .power import PowerSampler, add_power_details, idle_result, make_power_reader, sample_idle_power
from .stats import mean, percentile


GIB = 1024**3


@dataclass(frozen=True)
class LLMPreset:
    name: str
    model_id: str
    default_quantization: str
    full_precision_vram_gib: float
    int8_vram_gib: float
    int4_vram_gib: float
    gated: bool = False
    requires_remote_code: bool = False


LLM_PRESETS: dict[str, LLMPreset] = {
    "llama": LLMPreset(
        name="Llama 3.2 3B Instruct",
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        default_quantization="none",
        full_precision_vram_gib=7.0,
        int8_vram_gib=5.0,
        int4_vram_gib=4.0,
        gated=True,
    ),
    "qwen": LLMPreset(
        name="Qwen3 4B Instruct 2507",
        model_id="Qwen/Qwen3-4B-Instruct-2507",
        default_quantization="none",
        full_precision_vram_gib=9.0,
        int8_vram_gib=6.0,
        int4_vram_gib=5.0,
    ),
    "deepseek": LLMPreset(
        name="DeepSeek R1 Distill Qwen 7B",
        model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        default_quantization="none",
        full_precision_vram_gib=17.0,
        int8_vram_gib=10.0,
        int4_vram_gib=7.0,
    ),
    "kimi": LLMPreset(
        name="Kimi VL A3B Instruct",
        model_id="moonshotai/Kimi-VL-A3B-Instruct",
        default_quantization="4bit",
        full_precision_vram_gib=34.0,
        int8_vram_gib=20.0,
        int4_vram_gib=13.0,
        requires_remote_code=True,
    ),
}


def resolve_llm_configuration(
    preset_name: str,
    model_override: str | None,
    quantization: str,
) -> tuple[LLMPreset, str, str, float | None]:
    preset = LLM_PRESETS[preset_name]
    model_id = model_override or preset.model_id
    active_quantization = (
        preset.default_quantization if quantization == "auto" and model_override is None else quantization
    )
    if active_quantization in {"auto", "fp32", "fp16", "bf16"}:
        active_quantization = "none"
    estimates = {
        "none": preset.full_precision_vram_gib,
        "8bit": preset.int8_vram_gib,
        "4bit": preset.int4_vram_gib,
    }
    estimate = estimates[active_quantization] if model_override is None else None
    if quantization == "fp32" and estimate is not None:
        estimate *= 2
    return preset, model_id, active_quantization, estimate


def vram_budget_gib(
    total_gib: float,
    free_gib: float,
    limit_gib: float = 24.0,
    reserve_gib: float = 2.0,
) -> float:
    return max(0.0, min(total_gib, free_gib, limit_gib) - reserve_gib)


def generation_metrics(
    ttft_seconds: list[float],
    decode_seconds: list[float],
    e2e_seconds: list[float],
    output_token_counts: list[int],
    prompt_tokens: int,
    batch_size: int,
) -> dict[str, float]:
    runs = len(e2e_seconds)
    total_e2e = sum(e2e_seconds)
    total_output_tokens = sum(output_token_counts)
    first_tokens = batch_size * runs
    decode_tokens = max(0, total_output_tokens - first_tokens)
    return {
        "ttft_p50_ms": percentile(ttft_seconds, 50) * 1000.0,
        "ttft_p95_ms": percentile(ttft_seconds, 95) * 1000.0,
        "ttft_mean_ms": mean(ttft_seconds) * 1000.0,
        "prefill_tokens_per_second": prompt_tokens
        * batch_size
        * runs
        / max(sum(ttft_seconds), 1e-9),
        "decode_tokens_per_second": decode_tokens / max(sum(decode_seconds), 1e-9),
        "output_tokens_per_second": total_output_tokens / max(total_e2e, 1e-9),
        "e2e_p50_seconds": percentile(e2e_seconds, 50),
        "e2e_p95_seconds": percentile(e2e_seconds, 95),
        "e2e_mean_seconds": mean(e2e_seconds),
        "total_output_tokens": float(total_output_tokens),
    }


def run_llm_benchmarks(
    backend: str,
    device_indices: list[int],
    preset_name: str,
    model_override: str | None,
    quantization: str,
    prompt_tokens: int,
    new_tokens: int,
    batch_size: int,
    runs: int,
    warmup: int,
    trust_remote_code: bool,
    vram_limit_gib: float,
    vram_reserve_gib: float,
    cache_dir: Path | None,
    local_files_only: bool,
    power_enabled: bool,
    power_interval: float,
) -> list[dict[str, Any]]:
    configure_llm_runtime()
    preset, model_id, active_quantization, estimate = resolve_llm_configuration(
        preset_name, model_override, quantization
    )
    if model_override is None and preset.requires_remote_code and not trust_remote_code:
        raise ValueError(
            f"{preset.name} 需要执行模型仓库代码；确认来源后添加 --trust-remote-code"
        )
    torch, auto_model, auto_tokenizer, quantization_config = _dependencies(active_quantization)
    _disable_registered_native_jit(torch)

    results: list[dict[str, Any]] = []
    for index in device_indices:
        results.extend(
            _run_device(
                torch=torch,
                auto_model=auto_model,
                auto_tokenizer=auto_tokenizer,
                quantization_config=quantization_config,
                backend=backend,
                device_index=index,
                preset=preset,
                model_id=model_id,
                quantization=active_quantization,
                requested_dtype=quantization if quantization in {"fp32", "fp16", "bf16"} else "auto",
                estimated_vram_gib=estimate,
                prompt_tokens=prompt_tokens,
                new_tokens=new_tokens,
                batch_size=batch_size,
                runs=runs,
                warmup=warmup,
                trust_remote_code=trust_remote_code,
                vram_limit_gib=vram_limit_gib,
                vram_reserve_gib=vram_reserve_gib,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                power_enabled=power_enabled,
                power_interval=power_interval,
            )
        )
    return results


def _dependencies(quantization: str) -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise ImportError(
            "LLM 档位需要 transformers 与 accelerate；请运行 ./benchmark.sh --profile llm"
        ) from exc
    if importlib.util.find_spec("accelerate") is None:
        raise ImportError("LLM 档位缺少 accelerate；请运行 pip install 'accelerate>=1.0'")
    if quantization in {"4bit", "8bit"} and importlib.util.find_spec("bitsandbytes") is None:
        raise ImportError(
            f"{quantization} 量化需要 bitsandbytes；请安装兼容当前 CUDA/ROCm 的 bitsandbytes"
        )
    return torch, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def _run_device(
    *,
    torch: Any,
    auto_model: Any,
    auto_tokenizer: Any,
    quantization_config: Any,
    backend: str,
    device_index: int,
    preset: LLMPreset,
    model_id: str,
    quantization: str,
    requested_dtype: str,
    estimated_vram_gib: float | None,
    prompt_tokens: int,
    new_tokens: int,
    batch_size: int,
    runs: int,
    warmup: int,
    trust_remote_code: bool,
    vram_limit_gib: float,
    vram_reserve_gib: float,
    cache_dir: Path | None,
    local_files_only: bool,
    power_enabled: bool,
    power_interval: float,
) -> list[dict[str, Any]]:
    torch.cuda.set_device(device_index)
    torch.cuda.empty_cache()
    gc.collect()
    properties = torch.cuda.get_device_properties(device_index)
    device_name = f"{device_index}: {properties.name}"
    with torch.cuda.device(device_index):
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    total_gib = total_bytes / GIB
    free_gib = free_bytes / GIB
    budget_gib = vram_budget_gib(total_gib, free_gib, vram_limit_gib, vram_reserve_gib)
    if estimated_vram_gib is not None and estimated_vram_gib > budget_gib:
        raise RuntimeError(
            f"{preset.name} 预计需要约 {estimated_vram_gib:.1f} GiB，"
            f"当前 24GB 安全预算仅 {budget_gib:.1f} GiB；请释放显存或改用更小/量化模型"
        )

    dtype, dtype_label = _selected_dtype(torch, requested_dtype)
    precision = dtype_label
    load_kwargs: dict[str, Any] = {
        "cache_dir": str(cache_dir) if cache_dir else None,
        "device_map": device_index,
        "local_files_only": local_files_only,
        "low_cpu_mem_usage": True,
        "trust_remote_code": trust_remote_code,
    }
    if quantization == "4bit":
        load_kwargs["quantization_config"] = quantization_config(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        precision = f"int4/{dtype_label}"
    elif quantization == "8bit":
        load_kwargs["quantization_config"] = quantization_config(load_in_8bit=True)
        precision = "int8"
    else:
        load_kwargs[_transformers_dtype_key()] = dtype
    load_kwargs = {key: value for key, value in load_kwargs.items() if value is not None}

    tokenizer = None
    model = None
    try:
        load_started = time.perf_counter()
        try:
            tokenizer = auto_tokenizer.from_pretrained(
                model_id,
                cache_dir=str(cache_dir) if cache_dir else None,
                local_files_only=local_files_only,
                trust_remote_code=trust_remote_code,
            )
            model = auto_model.from_pretrained(model_id, **load_kwargs).eval()
        except Exception as exc:
            raise RuntimeError(_load_failure_message(preset, model_id, quantization, exc)) from exc
        load_seconds = time.perf_counter() - load_started
        _verify_gpu_only_placement(model, device_index)
        if requested_dtype != "auto" and getattr(model, "is_quantized", False):
            raise ValueError("源模型已量化，无法作为原始浮点精度基准；请提供未量化权重")

        model_vram_gib = max(
            float(model.get_memory_footprint()) / GIB,
            torch.cuda.memory_allocated(device_index) / GIB,
        )
        if model_vram_gib > budget_gib:
            raise RuntimeError(
                f"模型加载后占用 {model_vram_gib:.1f} GiB，超过 24GB 安全预算 "
                f"{budget_gib:.1f} GiB；拒绝 CPU/磁盘卸载后的失真测试"
            )

        context_limit = _context_limit(model)
        if context_limit is not None and prompt_tokens + new_tokens > context_limit:
            raise ValueError(
                f"请求需要 {prompt_tokens + new_tokens} tokens，超过模型上下文上限 "
                f"{context_limit}；请降低输入或输出长度"
            )

        cpu_input_ids, cpu_attention_mask = _fixed_prompt(tokenizer, torch, prompt_tokens, batch_size)
        pad_token_id = _pad_token_id(tokenizer)
        target_device = torch.device(f"cuda:{device_index}")
        generation_args = {
            "do_sample": False,
            "num_beams": 1,
            "pad_token_id": pad_token_id,
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "min_p": None,
            "typical_p": None,
            "epsilon_cutoff": None,
            "eta_cutoff": None,
            "use_cache": True,
        }

        power_reader = make_power_reader(backend, device_index) if power_enabled else None
        device_results: list[dict[str, Any]] = []
        if power_enabled:
            torch.cuda.synchronize(device_index)
            device_results.append(
                idle_result(backend, device_name, sample_idle_power(power_reader, power_interval))
            )

        for _ in range(warmup):
            _timed_generate(
                torch,
                model,
                tokenizer,
                cpu_input_ids,
                cpu_attention_mask,
                target_device,
                min(4, new_tokens),
                generation_args,
                device_index,
            )

        torch.cuda.reset_peak_memory_stats(device_index)
        ttft_seconds = []
        decode_seconds = []
        for _ in range(runs):
            ttft, decode = _timed_generation_phases(
                torch,
                model,
                cpu_input_ids,
                cpu_attention_mask,
                target_device,
                new_tokens,
                generation_args,
                device_index,
            )
            ttft_seconds.append(ttft)
            decode_seconds.append(decode)

        e2e_seconds = []
        output_token_counts = []
        sampler = PowerSampler(power_reader, power_interval)
        sampler.start()
        for _ in range(runs):
            elapsed, output_tokens = _timed_generate(
                torch,
                model,
                tokenizer,
                cpu_input_ids,
                cpu_attention_mask,
                target_device,
                new_tokens,
                generation_args,
                device_index,
            )
            e2e_seconds.append(elapsed)
            output_token_counts.append(output_tokens)
        power = sampler.stop(sum(e2e_seconds))
        peak_vram_gib = torch.cuda.max_memory_allocated(device_index) / GIB
        if peak_vram_gib > budget_gib:
            raise RuntimeError(
                f"生成峰值显存 {peak_vram_gib:.1f} GiB 超过 24GB 安全预算 {budget_gib:.1f} GiB；"
                "请降低 --llm-prompt-tokens、--llm-new-tokens 或 --batch-size"
            )

        metrics = generation_metrics(
            ttft_seconds,
            decode_seconds,
            e2e_seconds,
            output_token_counts,
            prompt_tokens,
            batch_size,
        )
        common = {
            "preset": preset.name,
            "model": model_id,
            "quantization": quantization,
            "parameter_dtype": dtype_label,
            "batch_size": batch_size,
            "prompt_tokens_per_request": prompt_tokens,
            "new_tokens_per_request": new_tokens,
            "runs": runs,
            "model_load_s": load_seconds,
            "model_vram_gib": model_vram_gib,
            "peak_vram_gib": peak_vram_gib,
            "vram_budget_gib": budget_gib,
            "physical_vram_gib": total_gib,
            "model_context_limit": context_limit,
            "phase_timing_scope": "model.generate CUDA timeline with KV cache",
            "e2e_timing_scope": "host-to-device + model.generate + output decode",
            "torch_native_jit_disabled": os.environ.get("TORCH_DISABLE_NATIVE_JIT") == "1",
        }
        if power and power.get("status") == "ok" and metrics["total_output_tokens"]:
            common["energy_per_output_token_j"] = (
                power["energy_j"] / metrics["total_output_tokens"]
            )

        ttft_result = _result(
            backend,
            device_name,
            "llm",
            "llm_ttft_p50",
            precision,
            metrics["ttft_p50_ms"],
            "ms",
            **common,
            p95_ms=metrics["ttft_p95_ms"],
            mean_ms=metrics["ttft_mean_ms"],
        )
        prefill_result = _result(
            backend,
            device_name,
            "llm",
            "llm_prefill",
            precision,
            metrics["prefill_tokens_per_second"],
            "tokens/s",
            **common,
        )
        decode_result = _result(
            backend,
            device_name,
            "llm",
            "llm_decode",
            precision,
            metrics["decode_tokens_per_second"],
            "tokens/s",
            **common,
        )
        output_result = _result(
            backend,
            device_name,
            "llm",
            "llm_output_e2e",
            precision,
            metrics["output_tokens_per_second"],
            "tokens/s",
            **common,
        )
        e2e_result = _result(
            backend,
            device_name,
            "llm",
            "llm_e2e_p50",
            precision,
            metrics["e2e_p50_seconds"],
            "s/request",
            **common,
            p95_s=metrics["e2e_p95_seconds"],
            mean_s=metrics["e2e_mean_seconds"],
        )
        e2e_result["details"]["power"] = power or {"status": "unsupported"}
        peak_result = _result(
            backend,
            device_name,
            "llm",
            "llm_peak_vram",
            precision,
            peak_vram_gib,
            "GiB",
            **common,
        )
        add_power_details(decode_result, power)
        add_power_details(output_result, power)
        device_results.extend(
            [ttft_result, prefill_result, decode_result, output_result, e2e_result, peak_result]
        )
        return device_results
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise RuntimeError(
                "LLM 生成发生显存不足；请降低 --llm-prompt-tokens、--llm-new-tokens 或 "
                "--batch-size，或启用 4bit 量化"
            ) from exc
        raise
    finally:
        del tokenizer, model
        torch.cuda.empty_cache()
        gc.collect()


def _preferred_dtype(torch: Any) -> tuple[Any, str]:
    supports_bf16 = getattr(torch.cuda, "is_bf16_supported", None)
    if supports_bf16 is not None and supports_bf16():
        return torch.bfloat16, "bf16"
    return torch.float16, "fp16"


def _selected_dtype(torch: Any, requested: str) -> tuple[Any, str]:
    if requested == "auto":
        return _preferred_dtype(torch)
    if requested == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前 GPU 不支持 BF16")
    if requested == "fp32":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[requested], requested


def configure_llm_runtime() -> None:
    os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")


def _disable_registered_native_jit(torch: Any) -> None:
    if os.environ.get("TORCH_DISABLE_NATIVE_JIT") != "1":
        return
    try:
        torch._native.triton_utils.deregister_op_overrides()
    except (AttributeError, ImportError):
        pass


def _transformers_dtype_key() -> str:
    import transformers

    numbers = tuple(int(value) for value in re.findall(r"\d+", transformers.__version__)[:2])
    return "dtype" if numbers >= (4, 56) else "torch_dtype"


def _fixed_prompt(
    tokenizer: Any,
    torch: Any,
    prompt_tokens: int,
    batch_size: int,
) -> tuple[Any, Any]:
    seed = (
        "Explain how heterogeneous accelerators execute machine-learning inference, "
        "including memory movement, batching, latency, throughput, and energy efficiency. "
    )
    seed_ids = tokenizer.encode(seed, add_special_tokens=False)
    if not seed_ids:
        raise RuntimeError("分词器没有为基准提示词生成 token")
    prefix = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    payload_length = prompt_tokens - len(prefix)
    repeated = (seed_ids * ((max(0, payload_length) + len(seed_ids) - 1) // len(seed_ids)))[
        : max(0, payload_length)
    ]
    token_ids = (prefix + repeated)[:prompt_tokens]
    input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def _pad_token_id(tokenizer: Any) -> int:
    if tokenizer.pad_token_id is not None:
        return int(tokenizer.pad_token_id)
    eos_token_id = tokenizer.eos_token_id
    if isinstance(eos_token_id, list):
        eos_token_id = eos_token_id[0] if eos_token_id else None
    if eos_token_id is not None:
        return int(eos_token_id)
    return 0


def _timed_generate(
    torch: Any,
    model: Any,
    tokenizer: Any,
    cpu_input_ids: Any,
    cpu_attention_mask: Any,
    target_device: Any,
    new_tokens: int,
    generation_args: dict[str, Any],
    device_index: int,
) -> tuple[float, int]:
    torch.cuda.synchronize(device_index)
    started = time.perf_counter()
    input_ids = cpu_input_ids.to(target_device)
    attention_mask = cpu_attention_mask.to(target_device)
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=new_tokens,
            min_new_tokens=new_tokens,
            **generation_args,
        )
    torch.cuda.synchronize(device_index)
    generated = output[:, input_ids.shape[1] :].detach().cpu()
    tokenizer.batch_decode(generated, skip_special_tokens=True)
    elapsed = time.perf_counter() - started
    output_tokens = int(generated.numel())
    del input_ids, attention_mask, output, generated
    return elapsed, output_tokens


class _CudaEventCriteria:
    def __init__(self, torch: Any) -> None:
        self.torch = torch
        self.events: list[Any] = []

    def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> Any:
        del scores, kwargs
        event = self.torch.cuda.Event(enable_timing=True)
        event.record()
        self.events.append(event)
        return self.torch.zeros(
            input_ids.shape[0],
            dtype=self.torch.bool,
            device=input_ids.device,
        )


def _timed_generation_phases(
    torch: Any,
    model: Any,
    cpu_input_ids: Any,
    cpu_attention_mask: Any,
    target_device: Any,
    new_tokens: int,
    generation_args: dict[str, Any],
    device_index: int,
) -> tuple[float, float]:
    from transformers.generation.stopping_criteria import StoppingCriteriaList

    input_ids = cpu_input_ids.to(target_device)
    attention_mask = cpu_attention_mask.to(target_device)
    torch.cuda.synchronize(device_index)
    start_event = torch.cuda.Event(enable_timing=True)
    criteria = _CudaEventCriteria(torch)
    start_event.record()
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=new_tokens,
            min_new_tokens=new_tokens,
            stopping_criteria=StoppingCriteriaList([criteria]),
            **generation_args,
        )
    torch.cuda.synchronize(device_index)
    if len(criteria.events) != new_tokens:
        raise RuntimeError(
            f"逐 token 计时只收到 {len(criteria.events)}/{new_tokens} 个事件，无法计算解码吞吐"
        )
    ttft_seconds = start_event.elapsed_time(criteria.events[0]) / 1000.0
    decode_seconds = criteria.events[0].elapsed_time(criteria.events[-1]) / 1000.0
    del input_ids, attention_mask, output, criteria
    return ttft_seconds, decode_seconds


def _verify_gpu_only_placement(model: Any, device_index: int) -> None:
    device_map = getattr(model, "hf_device_map", None)
    forbidden = []
    if device_map:
        for module, location in device_map.items():
            normalized = str(location).lower()
            if normalized not in {str(device_index), f"cuda:{device_index}"}:
                forbidden.append(f"{module}={location}")
    for name, parameter in model.named_parameters():
        parameter_device = parameter.device
        if parameter_device.type != "cuda" or parameter_device.index != device_index:
            forbidden.append(f"{name}={parameter_device}")
        if len(forbidden) >= 4:
            break
    if forbidden:
        raise RuntimeError(
            "模型没有完整驻留在目标单卡，无法公平比较端到端 GPU 推理: "
            + ", ".join(forbidden[:4])
        )


def _context_limit(model: Any) -> int | None:
    model_config = getattr(model, "config", None)
    for config in (model_config, getattr(model_config, "text_config", None)):
        value = getattr(config, "max_position_embeddings", None)
        if isinstance(value, int) and value > 0:
            return value
    return None


def _load_failure_message(
    preset: LLMPreset,
    model_id: str,
    quantization: str,
    error: Exception,
) -> str:
    hints = []
    if preset.gated and model_id == preset.model_id:
        hints.append("先接受模型许可并设置 HF_TOKEN")
    if quantization in {"4bit", "8bit"}:
        hints.append("确认 bitsandbytes 支持当前 CUDA/ROCm GPU")
    suffix = f"；{'；'.join(hints)}" if hints else ""
    return f"加载模型 {model_id} 失败: {type(error).__name__}: {error}{suffix}"


def _result(
    backend: str,
    device: str,
    suite: str,
    test: str,
    precision: str,
    value: float,
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
        "status": "ok",
        "details": details,
    }
