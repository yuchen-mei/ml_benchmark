from __future__ import annotations

import json
import shutil
import unicodedata
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def print_results(results: list[dict[str, Any]]) -> None:
    terminal_width = shutil.get_terminal_size(fallback=(120, 24)).columns
    print("\n" + format_results(results, terminal_width))


def format_results(results: list[dict[str, Any]], terminal_width: int = 120) -> str:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in results:
        if item["suite"] == "llm" or "requested_precision" in item.get("details", {}):
            continue
        groups[(item["backend"], item["device"])].append(item)

    lines = ["测试结果"]
    for (backend, device), items in groups.items():
        lines.extend(["", f"[{backend}] {device}"])
        table = _result_table(items)
        if max((_display_width(line) for line in table), default=0) <= terminal_width:
            lines.extend(table)
        else:
            lines.extend(_stacked_results(items))
        if backend == "coreml":
            for item in items:
                if "placement" in item.get("details", {}):
                    lines.append(_placement_note(item["details"]["placement"]))
                    break
    lines.extend(_llm_tables(results, terminal_width))
    return "\n".join(lines)


def _llm_tables(results: list[dict[str, Any]], terminal_width: int) -> list[str]:
    groups = defaultdict(dict)
    for item in results:
        if item["suite"] != "llm":
            continue
        detail = item.get("details", {})
        group = (item["backend"], item["device"], detail.get("model", "unknown"),
                 detail.get("batch_size", 1), detail.get("new_tokens_per_request"))
        key = (detail.get("requested_precision", item["precision"]), detail.get("prompt_tokens_per_request"))
        groups[group].setdefault(key, {})[item["test"]] = item
    lines = []
    metrics = ("llm_ttft_p50", "llm_prefill", "llm_decode", "llm_output_e2e", "llm_peak_vram")
    headers = ["精度", "输入 tokens", "TTFT ms", "Prefill tok/s", "Decode tok/s", "E2E tok/s", "峰值 GiB"]
    for (backend, device, model, batch, output), cases in groups.items():
        lines.extend(["", f"[{backend}] {device} / {model}", f"batch {batch}; 每请求输出 {output} tokens"])
        rows, reasons = [], []
        for (precision, prompt), items in cases.items():
            row = [precision, str(prompt)]
            for metric in metrics:
                value = items.get(metric, {}).get("value")
                row.append(f"{value:.3f}" if value is not None else "-")
            rows.append(row)
            for item in items.values():
                if item["status"] != "ok":
                    reasons.append(f"! {precision} / {prompt} tokens: {item['details'].get('reason', 'unknown')}")
        widths = [max(_display_width(header), *(_display_width(row[i]) for row in rows))
                  for i, header in enumerate(headers)]
        border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
        if _display_width(border) <= terminal_width:
            lines.extend([border, _table_line(headers, widths, ["left"] * len(headers)), border])
            lines.extend(_table_line(row, widths, ["left"] + ["right"] * 6) for row in rows)
            lines.append(border)
        else:
            for row in rows:
                lines.append(f"- {row[0]} / {row[1]} input tokens")
                lines.extend(f"  {header}: {value}" for header, value in zip(headers[2:], row[2:]))
        lines.extend(reasons)
    return lines


def _placement_note(placement: dict[str, Any]) -> str:
    planned = placement.get("neural_engine_planned")
    status = "包含 Neural Engine" if planned else "未分配到 Neural Engine" if planned is False else "无法确认 Neural Engine 分配"
    return f"Core ML 编译计划：{status}；结果为端到端推理，包含可能的 CPU 回退。"


def _result_table(results: list[dict[str, Any]]) -> list[str]:
    headers = ["项目", "精度", "结果", "平均功耗", "峰值功耗", "能效"]
    alignments = ["left", "left", "right", "right", "right", "right"]
    maximums = [24, 14, 22, 12, 12, 24]
    rows = [_result_row(item) for item in results]
    widths = [
        min(maximum, max(_display_width(header), *(_display_width(row[index]) for row in rows)))
        for index, (header, maximum) in enumerate(zip(headers, maximums))
    ]
    border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
    lines = [border, _table_line(headers, widths, ["left"] * len(headers)), border]
    lines.extend(_table_line(row, widths, alignments) for row in rows)
    lines.append(border)
    for item in results:
        if item["status"] != "ok":
            reason = item.get("details", {}).get("reason", "unknown")
            lines.append(f"! {item['test']} / {item['precision']}: {reason}")
    return lines


def _result_row(item: dict[str, Any]) -> list[str]:
    power = item.get("details", {}).get("power", {})
    efficiency = item.get("details", {}).get("efficiency", {})
    if item["status"] == "ok":
        result = f"{item['value']:.3f} {item['unit']}"
    else:
        result = "跳过"
    if item.get("unit") == "W":
        average_power = "-"
        peak_power = "-"
    else:
        average_power = _format_optional(power.get("average_w"), "W", 1)
        peak_power = _format_optional(power.get("peak_w"), "W", 1)
    efficiency_text = _format_optional(efficiency.get("value"), efficiency.get("unit", ""), 3)
    return [item["test"], item["precision"], result, average_power, peak_power, efficiency_text]


def _table_line(values: list[str], widths: list[int], alignments: list[str]) -> str:
    cells = [
        _pad_display(value, width, alignment)
        for value, width, alignment in zip(values, widths, alignments)
    ]
    return "| " + " | ".join(cells) + " |"


def _stacked_results(results: list[dict[str, Any]]) -> list[str]:
    lines = []
    for item in results:
        test, precision, result, average_power, peak_power, efficiency = _result_row(item)
        lines.append(f"- {test} [{precision}]")
        lines.append(f"  结果: {result}")
        if average_power != "-" or peak_power != "-":
            lines.append(f"  功耗: 平均 {average_power} / 峰值 {peak_power}")
        if efficiency != "-":
            lines.append(f"  能效: {efficiency}")
        if item["status"] != "ok":
            lines.append(f"  原因: {item.get('details', {}).get('reason', 'unknown')}")
    return lines


def _display_width(value: str) -> int:
    width = 0
    for character in str(value):
        if unicodedata.combining(character) or unicodedata.category(character) in {"Cf", "Cc"}:
            continue
        width += 2 if unicodedata.east_asian_width(character) in {"W", "F"} else 1
    return width


def _truncate_display(value: str, width: int) -> str:
    value = str(value)
    if _display_width(value) <= width:
        return value
    if width <= 3:
        return "." * width
    available = width - 3
    output = []
    used = 0
    for character in value:
        character_width = _display_width(character)
        if used + character_width > available:
            break
        output.append(character)
        used += character_width
    return "".join(output) + "..."


def _pad_display(value: str, width: int, alignment: str = "left") -> str:
    value = _truncate_display(str(value), width)
    padding = " " * max(0, width - _display_width(value))
    return padding + value if alignment == "right" else value + padding


def save_report(
    output_dir: Path,
    environment: dict[str, Any],
    arguments: dict[str, Any],
    results: list[dict[str, Any]],
    *,
    paths: tuple[Path, Path] | None = None,
    complete: bool = True,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    payload = {
        "schema_version": 2,
        "created_at": datetime.now().astimezone().isoformat(),
        "environment": environment,
        "arguments": arguments,
        "results": results,
        "complete": complete,
    }
    json_path, markdown_path = paths or (output_dir / f"mlbench_{timestamp}.json", output_dir / f"mlbench_{timestamp}.md")
    for path, content in [(json_path, json.dumps(payload, ensure_ascii=False, indent=2)),
                          (markdown_path, _markdown(payload))]:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    return json_path, markdown_path


def _markdown(payload: dict[str, Any]) -> str:
    system = payload["environment"]["system"]
    lines = [
        "# Heterogeneous MLBench Report",
        "",
        f"- Created: `{payload['created_at']}`",
        f"- System: `{system['platform']}`",
        f"- Python: `{system['python']}`",
        f"- Backends: `{', '.join(payload['environment']['available_backends'])}`",
    ]
    apple = payload["environment"]["hardware"].get("apple_silicon", {})
    if apple.get("detected"):
        lines.append(f"- Apple Silicon: `{apple['name']}`, unified memory `{apple['unified_memory_bytes'] / 1024**3:.0f} GiB`")
    llm_results = [item for item in payload["results"] if item["suite"] == "llm"]
    if llm_results:
        models = sorted({item.get("details", {}).get("model", "unknown") for item in llm_results})
        precisions = sorted({item.get("details", {}).get("requested_precision", item["precision"]) for item in llm_results})
        prompts = sorted({item["details"]["prompt_tokens_per_request"] for item in llm_results
                          if item.get("details", {}).get("prompt_tokens_per_request") is not None})
        lines.extend([
            f"- Models: {', '.join(f'`{model}`' for model in models)}",
            f"- Precisions: `{', '.join(precisions)}`",
            f"- Prefill lengths: `{prompts}` tokens per request",
            "- Model/peak memory and per-case timing are recorded in the rows and JSON details.",
        ])
    if not payload.get("complete", True):
        lines.extend(["", "> Partial report: benchmark is still running or was interrupted."])
    lines.extend(
        [
            "",
            "| Backend | Device | Model | Input tokens | Output tokens | Suite | Test | Precision | Result | Avg Power | Peak Power | Energy | Efficiency |",
            "|---|---|---|---:|---:|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in payload["results"]:
        if item["status"] == "ok":
            result = f"{item['value']:.4f} {item['unit']}"
        else:
            result = f"Skipped: {item.get('details', {}).get('reason', 'unknown')}"
        power = item.get("details", {}).get("power", {})
        efficiency = item.get("details", {}).get("efficiency", {})
        fields = [
            item["backend"],
            item["device"],
            item.get("details", {}).get("model", "-"),
            item.get("details", {}).get("prompt_tokens_per_request", "-"),
            item.get("details", {}).get("new_tokens_per_request", "-"),
            item["suite"],
            item["test"],
            item["precision"],
            result,
            _format_optional(power.get("average_w"), "W"),
            _format_optional(power.get("peak_w"), "W"),
            _format_optional(power.get("energy_j"), "J"),
            _format_optional(efficiency.get("value"), efficiency.get("unit", "")),
        ]
        lines.append("| " + " | ".join("<br>".join(str(field).splitlines()).replace("|", "\\|")
                                      for field in fields) + " |")
    lines.extend(
        [
            "",
            "> TFLOP/s is calculated from dense GEMM operation count. It is not the vendor's theoretical TOPS rating.",
            "> AMD NPU results use VitisAI EP with CPU fallback enabled for unsupported graph nodes.",
            "",
        ]
    )
    for item in payload["results"]:
        if item["backend"] == "coreml" and item["test"] == "synthetic_cnn":
            lines.append(f"> {item['device']}: {_placement_note(item['details'].get('placement', {}))}")
    if any(item["backend"] in {"mps", "mlx", "coreml"} for item in payload["results"]):
        lines.extend([
            "> Apple GPU timings include command submission and synchronization. Memory is shared with macOS.",
            "> Core ML placement is a compiler plan, not measured Neural Engine utilization. Apple power telemetry is unavailable in this runner.",
        ])
    return "\n".join(lines)


def _format_optional(value: Any, unit: str, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f} {unit}".strip()
