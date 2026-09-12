"""Plan and isolate LLM precision × prompt-length benchmark cases."""
from __future__ import annotations

import json
import subprocess
import sys
from typing import Any, Callable

from .gpu import _result
from .isolation import _extract_results
from .llm import LLM_PRESETS
from .runtime import process_exit_reason


LLM_PRECISIONS = ("fp32", "fp16", "bf16", "8bit", "4bit")
LLM_PROMPT_LENGTHS = (128, 256, 512, 1024, 2048)


def selected_models(preset: str, model_override: str | None) -> list[str]:
    if model_override:
        return ["qwen" if preset == "all" else preset]
    return list(LLM_PRESETS) if preset == "all" else [preset]


def selected_precisions(value: str) -> list[str]:
    if value == "all":
        return list(LLM_PRECISIONS)
    if value not in {*LLM_PRECISIONS, "auto", "none"}:
        raise ValueError(f"无效 LLM 精度: {value}")
    return [value]


def prompt_lengths(value: str | int | None) -> list[int]:
    if value is None:
        return list(LLM_PROMPT_LENGTHS)
    try:
        lengths = [int(part.strip()) for part in str(value).split(",")]
    except ValueError as exc:
        raise ValueError("llm-prompt-tokens 必须是正整数或逗号分隔的正整数列表") from exc
    if not lengths or any(length < 1 for length in lengths):
        raise ValueError("llm-prompt-tokens 必须至少为 1")
    return sorted(set(lengths))


def failure_result(backend: str, device: str, configuration: dict[str, Any], reason: str) -> dict[str, Any]:
    return _result(
        backend, device, "llm", "llm_configuration", configuration["quantization"], None, "",
        reason=reason, worker_failure=True,
        prompt_tokens_per_request=configuration["prompt_tokens"],
        new_tokens_per_request=configuration["new_tokens"], batch_size=configuration["batch_size"],
        runs=configuration["runs"],
        model=configuration.get("model_override") or LLM_PRESETS[configuration["preset_name"]].model_id,
    )


def run_llm_sweep(backend: str, device: str, configuration: dict[str, Any],
                  precisions: list[str], lengths: list[int],
                  progress: Callable[[str], None] | None = None,
                  on_case: Callable[[list[dict[str, Any]]], None] | None = None) -> list[dict[str, Any]]:
    results = []
    total = len(precisions) * len(lengths)
    model = configuration.get("model_override") or LLM_PRESETS[configuration["preset_name"]].model_id
    for precision_index, precision in enumerate(precisions):
        for length_index, length in enumerate(lengths):
            position = precision_index * len(lengths) + length_index + 1
            if progress:
                progress(f"LLM [{position}/{total}] {model} / {device} / {precision} / prefill {length} tokens")
            case = dict(configuration, quantization=precision, prompt_tokens=length)
            # Fresh process per case releases model/KV/allocator state even after native failures.
            try:
                completed = subprocess.run(
                    [sys.executable, "-m", "mlbench.llm_worker", backend],
                    input=json.dumps(case), capture_output=True, text=True, timeout=7200,
                )
                payload = _extract_results(completed.stdout)
                if completed.returncode != 0 or not payload:
                    reason = process_exit_reason(completed.returncode, completed.stderr) if completed.returncode else "LLM 子进程没有有效结果"
                    payload = [failure_result(backend, device, case, reason)]
            except (OSError, subprocess.SubprocessError) as exc:
                payload = [failure_result(backend, device, case, f"{type(exc).__name__}: {exc}")]
            for item in payload:
                item["device"] = device
                item.setdefault("details", {}).update(
                    requested_precision=precision, prompt_tokens_per_request=length,
                    new_tokens_per_request=case["new_tokens"], batch_size=case["batch_size"],
                    runs=case["runs"],
                    model=model, preset=configuration["preset_name"],
                )
            results.extend(payload)
            if on_case:
                on_case(payload)
            if progress and any(item.get("details", {}).get("worker_failure") for item in payload):
                progress(f"  未完成：{payload[-1]['details'].get('reason', '运行失败')}；继续下一组")
    return results
