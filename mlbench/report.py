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
        groups[(item["backend"], item["device"])].append(item)

    lines = ["测试结果"]
    for (backend, device), items in groups.items():
        lines.extend(["", f"[{backend}] {device}"])
        table = _result_table(items)
        if max((_display_width(line) for line in table), default=0) <= terminal_width:
            lines.extend(table)
        else:
            lines.extend(_stacked_results(items))
    return "\n".join(lines)


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
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    payload = {
        "schema_version": 2,
        "created_at": datetime.now().astimezone().isoformat(),
        "environment": environment,
        "arguments": arguments,
        "results": results,
    }
    json_path = output_dir / f"mlbench_{timestamp}.json"
    markdown_path = output_dir / f"mlbench_{timestamp}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(_markdown(payload), encoding="utf-8")
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
    llm_result = next((item for item in payload["results"] if item["suite"] == "llm"), None)
    if llm_result is not None:
        details = llm_result.get("details", {})
        lines.extend(
            [
                f"- LLM: `{details.get('model', 'unknown')}` (`{llm_result['precision']}`)",
                f"- Workload: `{details.get('prompt_tokens_per_request')} input + "
                f"{details.get('new_tokens_per_request')} output tokens`, "
                f"batch `{details.get('batch_size')}`, runs `{details.get('runs')}`",
                f"- VRAM: model `{details.get('model_vram_gib', 0):.2f} GiB`, "
                f"peak `{details.get('peak_vram_gib', 0):.2f} GiB`, "
                f"budget `{details.get('vram_budget_gib', 0):.2f} GiB`",
            ]
        )
    lines.extend(
        [
            "",
            "| Backend | Device | Suite | Test | Precision | Result | Avg Power | Peak Power | Energy | Efficiency |",
            "|---|---|---|---|---|---:|---:|---:|---:|---:|",
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
            item["suite"],
            item["test"],
            item["precision"],
            result,
            _format_optional(power.get("average_w"), "W"),
            _format_optional(power.get("peak_w"), "W"),
            _format_optional(power.get("energy_j"), "J"),
            _format_optional(efficiency.get("value"), efficiency.get("unit", "")),
        ]
        lines.append("| " + " | ".join(str(field).replace("|", "\\|") for field in fields) + " |")
    lines.extend(
        [
            "",
            "> TFLOP/s is calculated from dense GEMM operation count. It is not the vendor's theoretical TOPS rating.",
            "> NPU results use VitisAI EP with CPU fallback enabled for unsupported graph nodes.",
            "",
        ]
    )
    return "\n".join(lines)


def _format_optional(value: Any, unit: str, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f} {unit}".strip()
