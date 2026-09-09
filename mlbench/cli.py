from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

from . import __version__
from .detect import detect_environment, render_doctor
from .gpu import run_cpu_benchmarks, run_gpu_benchmarks
from .npu import run_npu_benchmarks
from .report import print_results, save_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlbench",
        description="NVIDIA GPU 与 AMD GPU/NPU 一键 ML 算力测试",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("doctor", help="仅诊断硬件和运行时")
    run = subparsers.add_parser("run", help="运行基准测试")
    run.add_argument(
        "--backend",
        choices=["all", "cuda", "rocm", "npu", "cpu"],
        default="all",
        help="all 会测试所有可用加速器；没有加速器时回退 CPU",
    )
    run.add_argument("--profile", choices=["quick", "standard", "extended"], default="standard")
    run.add_argument("--suite", default="all", help="all 或逗号分隔的 compute,inference,memory")
    run.add_argument("--duration", type=float, default=None, help="每项测试的最短秒数")
    run.add_argument("--matrix-size", type=int, default=0, help="GEMM 方阵边长，0 为自动")
    run.add_argument("--batch-size", type=int, default=0, help="CNN batch，0 为自动；NPU 默认 1")
    run.add_argument("--warmup", type=int, default=3, help="预热轮数")
    run.add_argument("--device", default="all", help="GPU 编号，如 0 或 0,1")
    run.add_argument("--npu-streams", type=int, default=1, help="NPU 并发请求数")
    run.add_argument("--npu-config", default=None, help="可选 VitisAI EP config_file")
    run.add_argument("--no-power", action="store_true", help="关闭加速器功耗采样")
    run.add_argument("--power-interval", type=float, default=0.1, help="功耗采样间隔秒数，默认 0.1")
    run.add_argument("--output-dir", default="results", help="报告与 NPU 缓存目录")
    run.add_argument("--json-only", action="store_true", help="控制台只输出 JSON")
    run.add_argument("--verbose", action="store_true", help="失败时输出调用栈")
    return parser


def _parse_suites(raw: str) -> set[str]:
    valid = {"compute", "inference", "memory"}
    if raw == "all":
        return valid
    suites = {part.strip().lower() for part in raw.split(",") if part.strip()}
    unknown = suites - valid
    if not suites or unknown:
        raise ValueError(f"无效 suite: {', '.join(sorted(unknown)) or raw}")
    return suites


def _device_indices(raw: str, count: int) -> list[int]:
    if raw == "all":
        return list(range(count))
    indices = [int(part.strip()) for part in raw.split(",")]
    invalid = [index for index in indices if index < 0 or index >= count]
    if invalid:
        raise ValueError(f"GPU 编号超出范围: {invalid}; 可用范围 0..{count - 1}")
    return indices


def _selected_backends(requested: str, available: list[str]) -> list[str]:
    if requested != "all":
        if requested not in available:
            raise RuntimeError(f"请求的后端 {requested} 不可用；当前可用: {', '.join(available)}")
        return [requested]
    accelerators = [backend for backend in available if backend != "cpu"]
    return accelerators or ["cpu"]


def _arguments_dict(args: argparse.Namespace) -> dict[str, Any]:
    return {key: value for key, value in vars(args).items() if key not in {"verbose"}}


def _validate_arguments(args: argparse.Namespace, suites: set[str]) -> None:
    if args.duration is not None and args.duration <= 0:
        raise ValueError("duration 必须大于 0")
    if args.matrix_size < 0:
        raise ValueError("matrix-size 不能小于 0")
    if args.batch_size < 0:
        raise ValueError("batch-size 不能小于 0")
    if args.warmup < 0:
        raise ValueError("warmup 不能小于 0")
    if args.npu_streams < 1:
        raise ValueError("npu-streams 必须至少为 1")
    if args.power_interval <= 0:
        raise ValueError("power-interval 必须大于 0")
    if args.backend == "npu" and "inference" not in suites:
        raise ValueError("AMD NPU 当前仅支持 inference suite")
    if args.backend == "cpu" and suites == {"inference"}:
        raise ValueError("CPU 回退当前不提供 inference suite")


def _run(args: argparse.Namespace) -> int:
    environment = detect_environment()
    suites = _parse_suites(args.suite)
    _validate_arguments(args, suites)
    backends = _selected_backends(args.backend, environment["available_backends"])
    output_dir = Path(args.output_dir).expanduser().resolve()
    results: list[dict[str, Any]] = []

    if not args.json_only:
        print(render_doctor(environment))
        print(f"\n即将测试   : {', '.join(backends)}")
        print(f"测试档位   : {args.profile}")
        print(f"功耗采样   : {'关闭' if args.no_power else f'开启 ({args.power_interval:.2f}s)'}")

    for backend in backends:
        if backend in {"cuda", "rocm"}:
            count = len(environment["runtime"]["torch"]["devices"])
            indices = _device_indices(args.device, count)
            results.extend(
                run_gpu_benchmarks(
                    backend,
                    indices,
                    suites,
                    args.profile,
                    args.duration,
                    args.matrix_size,
                    args.batch_size,
                    args.warmup,
                    not args.no_power,
                    args.power_interval,
                )
            )
        elif backend == "npu":
            if "inference" in suites:
                results.extend(
                    run_npu_benchmarks(
                        output_dir,
                        args.profile,
                        args.duration,
                        args.batch_size,
                        args.warmup,
                        args.npu_streams,
                        args.npu_config,
                        not args.no_power,
                        args.power_interval,
                    )
                )
        elif backend == "cpu":
            results.extend(
                run_cpu_benchmarks(
                    suites,
                    args.profile,
                    args.duration,
                    args.matrix_size,
                    args.warmup,
                )
            )

    json_path, markdown_path = save_report(output_dir, environment, _arguments_dict(args), results)
    if args.json_only:
        print(json_path.read_text(encoding="utf-8"))
    else:
        print_results(results)
        print(f"\nJSON 报告 : {json_path}")
        print(f"Markdown  : {markdown_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    actual = list(sys.argv[1:] if argv is None else argv)
    if not actual or actual[0].startswith("-"):
        actual.insert(0, "run")
    parser = _parser()
    args = parser.parse_args(actual)
    try:
        if args.command == "doctor":
            print(render_doctor(detect_environment()))
            return 0
        return _run(args)
    except (RuntimeError, ValueError, FileNotFoundError, ImportError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        if getattr(args, "verbose", False):
            traceback.print_exc()
        return 2
