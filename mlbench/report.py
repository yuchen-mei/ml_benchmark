from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def print_results(results: list[dict[str, Any]]) -> None:
    print("\n测试结果")
    print("=" * 96)
    print(f"{'后端':<7} {'设备':<28} {'项目':<23} {'精度':<15} {'结果':>14}")
    print("-" * 96)
    for item in results:
        device = item["device"][:27]
        if item["status"] == "ok":
            rendered = f"{item['value']:.3f} {item['unit']}"
        else:
            rendered = "跳过"
        print(
            f"{item['backend']:<7} {device:<28} {item['test']:<23} "
            f"{item['precision']:<15} {rendered:>14}"
        )
        if item["status"] != "ok":
            print(f"        原因: {item.get('details', {}).get('reason', 'unknown')}")


def save_report(
    output_dir: Path,
    environment: dict[str, Any],
    arguments: dict[str, Any],
    results: list[dict[str, Any]],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    payload = {
        "schema_version": 1,
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
        "",
        "| Backend | Device | Suite | Test | Precision | Result |",
        "|---|---|---|---|---|---:|",
    ]
    for item in payload["results"]:
        if item["status"] == "ok":
            result = f"{item['value']:.4f} {item['unit']}"
        else:
            result = f"Skipped: {item.get('details', {}).get('reason', 'unknown')}"
        fields = [
            item["backend"],
            item["device"],
            item["suite"],
            item["test"],
            item["precision"],
            result,
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
