from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

from . import __version__
from .apple import run_apple_isolated
from .detect import detect_environment, render_doctor
from .gpu import run_cpu_benchmarks
from .isolation import run_gpu_benchmarks_isolated
from .llm import (
    LLM_PRESETS,
    configure_llm_runtime,
)
from .llm_sweep import prompt_lengths, selected_precisions, selected_models, run_llm_sweep
from .npu_isolation import run_npu_benchmarks_isolated
from .report import print_results, save_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlbench",
        description="NVIDIA、AMD 与 Apple Silicon GPU/NPU 一键 ML 算力测试",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("doctor", help="仅诊断硬件和运行时")
    run = subparsers.add_parser("run", help="运行基准测试")
    run.add_argument(
        "--backend",
        choices=["all", "cuda", "rocm", "npu", "mps", "mlx", "coreml", "cpu"],
        default="all",
        help="all 会测试所有可用加速器；没有加速器时回退 CPU",
    )
    run.add_argument(
        "--profile",
        choices=["quick", "standard", "extended", "llm"],
        default="standard",
        help="llm 为真实大模型端到端生成档位",
    )
    run.add_argument("--suite", default="all", help="all 或逗号分隔的 compute,inference,memory；也可为 llm")
    run.add_argument("--duration", type=float, default=None, help="每项测试的最短秒数")
    run.add_argument("--matrix-size", type=int, default=0, help="GEMM 方阵边长，0 为自动")
    run.add_argument("--batch-size", type=int, default=0, help="CNN/LLM batch，0 为自动；LLM/NPU 默认 1")
    run.add_argument("--warmup", type=int, default=3, help="预热轮数")
    run.add_argument("--device", default="all", help="GPU 编号，如 0 或 0,1")
    run.add_argument("--npu-streams", type=int, default=1, help="NPU 并发请求数")
    run.add_argument("--npu-config", default=None, help="可选 VitisAI EP config_file")
    run.add_argument("--npu-model", default=None, help="可选 ONNX 模型；默认使用 Ryzen AI quicktest 模型")
    run.add_argument("--coreml-compute-units", choices=["compare", "cpu_and_ne", "all", "cpu_and_gpu", "cpu_only"],
                     default="compare", help="默认比较 CPU+Neural Engine 与 Core ML 自动分配")
    run.add_argument("--mps-matmul", choices=["auto", "metal", "mpsgraph"], default="auto",
                     help="MPS 矩阵乘内核选择；auto 使用 PyTorch/环境默认值")
    run.add_argument("--apple-memory-limit-gib", type=float, default=0,
                     help="Apple GPU 统一内存上限；0 自动使用系统内存的 75%% 与 Metal 建议上限的较小值")
    run.add_argument(
        "--llm-preset",
        choices=["all", *sorted(LLM_PRESETS)],
        default="all",
        help="默认遍历全部模型预设；可指定一个模型",
    )
    run.add_argument("--llm-model", default=None, help="覆盖预设的 Hugging Face 模型 ID 或本地目录")
    run.add_argument(
        "--llm-quantization", "--llm-dtype",
        choices=["all", "fp32", "fp16", "bf16", "4bit", "8bit", "auto", "none"],
        default="all",
        help="默认测试 fp32/fp16/bf16/8bit/4bit；none：MLX 保留源精度，CUDA/ROCm 自动选 BF16/FP16",
    )
    run.add_argument("--llm-prompt-tokens", default=None,
                     help="输入 token 数或逗号分隔列表；默认 128,256,512,1024,2048")
    run.add_argument("--llm-new-tokens", type=int, default=128, help="每个请求强制生成的 token 数")
    run.add_argument("--llm-runs", type=int, default=3, help="TTFT 与端到端生成的测量轮数")
    run.add_argument(
        "--llm-vram-limit-gib",
        type=float,
        default=24.0,
        help="显存硬上限，最大允许 24 GiB",
    )
    run.add_argument(
        "--llm-vram-reserve-gib",
        type=float,
        default=2.0,
        help="为运行时和桌面保留的显存，默认 2 GiB",
    )
    run.add_argument("--llm-cache-dir", default=None, help="可选模型缓存目录")
    run.add_argument("--llm-local-files-only", action="store_true", help="仅使用本地已有模型文件")
    run.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="允许执行模型仓库代码；Kimi 官方模型需要显式开启",
    )
    run.add_argument("--no-power", action="store_true", help="关闭加速器功耗采样")
    run.add_argument("--power-interval", type=float, default=0.1, help="功耗采样间隔秒数，默认 0.1")
    run.add_argument("--output-dir", default="results", help="报告与 NPU 缓存目录")
    run.add_argument("--json-only", action="store_true", help="控制台只输出 JSON")
    run.add_argument("--verbose", action="store_true", help="失败时输出调用栈")
    return parser


def _parse_suites(raw: str) -> set[str]:
    valid = {"compute", "inference", "memory", "llm"}
    if raw == "all":
        return {"compute", "inference", "memory"}
    suites = {part.strip().lower() for part in raw.split(",") if part.strip()}
    unknown = suites - valid
    if not suites or unknown:
        raise ValueError(f"无效 suite: {', '.join(sorted(unknown)) or raw}")
    return suites


def _resolve_suites(profile: str, raw: str) -> set[str]:
    if profile == "llm":
        if raw not in {"all", "llm"}:
            raise ValueError("llm 档位不能与 compute/inference/memory 混跑")
        return {"llm"}
    suites = _parse_suites(raw)
    if "llm" in suites and len(suites) != 1:
        raise ValueError("llm suite 必须单独运行，避免其他项目污染显存与功耗")
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


def _gpu_metadata(environment: dict[str, Any]) -> tuple[dict[int, str], dict[int, str]]:
    labels: dict[int, str] = {}
    architectures: dict[int, str] = {}
    devices = environment["runtime"]["torch"].get("devices", [])
    amd_devices = environment["hardware"].get("amd_gpus", [])
    for position, device in enumerate(devices):
        index = int(device["index"])
        labels[index] = f"{index}: {device['name']}"
        hardware_architecture = ""
        if position < len(amd_devices):
            hardware_architecture = str(amd_devices[position].get("architecture") or "")
        architecture = hardware_architecture or str(device.get("architecture") or "")
        if architecture:
            architectures[index] = architecture
    return labels, architectures


def _validate_arguments(args: argparse.Namespace, suites: set[str]) -> None:
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        raise ValueError("duration 必须大于 0")
    if args.matrix_size < 0:
        raise ValueError("matrix-size 不能小于 0")
    if args.batch_size < 0:
        raise ValueError("batch-size 不能小于 0")
    if args.warmup < 0:
        raise ValueError("warmup 不能小于 0")
    if args.npu_streams < 1:
        raise ValueError("npu-streams 必须至少为 1")
    if not math.isfinite(args.power_interval) or args.power_interval <= 0:
        raise ValueError("power-interval 必须大于 0")
    memory_limit = getattr(args, "apple_memory_limit_gib", 0)
    if not math.isfinite(memory_limit) or memory_limit < 0:
        raise ValueError("apple-memory-limit-gib 必须是非负有限数")
    if "llm" in suites:
        if args.backend in {"cpu", "npu", "coreml", "mps"}:
            raise ValueError("llm 档位支持 CUDA/ROCm 或 Apple MLX；Mac 请使用 --backend mlx")
        if args.duration is not None:
            raise ValueError("llm 档位使用 --llm-runs 控制轮数，不接受 --duration")
        prompt_lengths(args.llm_prompt_tokens)
        selected_precisions(args.llm_quantization)
        if args.llm_new_tokens < 2:
            raise ValueError("llm-new-tokens 必须至少为 2，才能计算解码吞吐")
        if args.llm_runs < 1:
            raise ValueError("llm-runs 必须至少为 1")
        if args.backend != "mlx" and not 0 < args.llm_vram_limit_gib <= 24:
            raise ValueError("llm-vram-limit-gib 必须大于 0 且不超过 24")
        if args.backend != "mlx" and not 0 <= args.llm_vram_reserve_gib < args.llm_vram_limit_gib:
            raise ValueError("llm-vram-reserve-gib 必须非负且小于显存上限")
    if args.backend == "npu" and "inference" not in suites:
        raise ValueError("AMD NPU 当前仅支持 inference suite")
    if args.backend == "coreml" and "inference" not in suites:
        raise ValueError("Apple Core ML 当前仅支持 inference suite")
    if args.backend == "cpu" and suites == {"inference"}:
        raise ValueError("CPU 回退当前不提供 inference suite")


def _run(args: argparse.Namespace) -> int:
    suites = _resolve_suites(args.profile, args.suite)
    _validate_arguments(args, suites)
    if "llm" in suites:
        configure_llm_runtime()
    environment = detect_environment()
    backends = _selected_backends(args.backend, environment["available_backends"])
    if "llm" in suites:
        backends = [backend for backend in backends if backend in {"cuda", "rocm", "mlx"}]
        if not backends:
            raise RuntimeError("llm 档位没有发现可用的 CUDA/ROCm/MLX GPU")
    output_dir = Path(args.output_dir).expanduser().resolve()
    results: list[dict[str, Any]] = []
    report_paths = None

    def checkpoint(items: list[dict[str, Any]]) -> None:
        nonlocal report_paths
        results.extend(items)
        report_paths = save_report(output_dir, environment, _arguments_dict(args), results,
                                   paths=report_paths, complete=False)

    if not args.json_only:
        print(render_doctor(environment))
        print(f"\n即将测试   : {', '.join(backends)}")
        print(f"测试档位   : {args.profile}")
        print(f"功耗采样   : {'关闭' if args.no_power else f'开启 ({args.power_interval:.2f}s)'}")
        if "llm" in suites:
            models = [args.llm_model or LLM_PRESETS[name].model_id
                      for name in selected_models(args.llm_preset, args.llm_model)]
            print(f"LLM 模型   : {', '.join(models)}")
            print(
                f"LLM 负载   : {prompt_lengths(args.llm_prompt_tokens)} in + {args.llm_new_tokens} out, "
                f"batch {args.batch_size or 1}, {args.llm_runs} 轮, "
                f"{', '.join(selected_precisions(args.llm_quantization))}"
            )

    for backend in backends:
        if "llm" in suites:
            configuration = dict(
                preset_name=args.llm_preset, model_override=args.llm_model,
                new_tokens=args.llm_new_tokens, batch_size=args.batch_size or 1,
                runs=args.llm_runs, warmup=args.warmup, trust_remote_code=args.trust_remote_code,
                cache_dir=str(Path(args.llm_cache_dir).expanduser().resolve()) if args.llm_cache_dir else None,
                local_files_only=args.llm_local_files_only, power_enabled=not args.no_power,
            )
            count = 1 if backend == "mlx" else len(environment["runtime"]["torch"]["devices"])
            for index in _device_indices(args.device, count):
                if backend == "mlx":
                    configuration["memory_limit_gib"] = args.apple_memory_limit_gib
                    label = environment["hardware"]["apple_silicon"]["name"]
                else:
                    configuration.update(device_indices=[index], vram_limit_gib=args.llm_vram_limit_gib,
                                         vram_reserve_gib=args.llm_vram_reserve_gib, power_interval=args.power_interval)
                    label = environment["runtime"]["torch"]["devices"][index]["name"]
                for preset_name in selected_models(args.llm_preset, args.llm_model):
                    run_llm_sweep(
                        backend, f"{index}: {label}", dict(configuration, preset_name=preset_name),
                        selected_precisions(args.llm_quantization), prompt_lengths(args.llm_prompt_tokens),
                        progress=None if args.json_only else lambda message: print(message, flush=True),
                        on_case=checkpoint,
                    )
            continue
        if backend in {"cuda", "rocm", "mps"}:
            count = 1 if backend == "mps" else len(environment["runtime"]["torch"]["devices"])
            indices = _device_indices(args.device, count)
            labels, architectures = _gpu_metadata(environment)
            results.extend(
                run_gpu_benchmarks_isolated(
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
                    labels,
                    architectures,
                    args.apple_memory_limit_gib,
                    args.mps_matmul,
                )
            )
        elif backend == "mlx":
            _device_indices(args.device, 1)
            configuration = dict(
                suites=sorted(suites), profile=args.profile, requested_duration=args.duration,
                matrix_size=args.matrix_size, batch_size=args.batch_size, warmup=args.warmup,
                power_enabled=not args.no_power, memory_limit_gib=args.apple_memory_limit_gib,
            )
            timeout = max(600, (args.duration or 4) * 50)
            results.extend(run_apple_isolated("mlx", configuration, timeout))
        elif backend == "coreml":
            if "inference" in suites:
                results.extend(run_apple_isolated("coreml", dict(
                    output_dir=str(output_dir), profile=args.profile, requested_duration=args.duration,
                    batch_size=args.batch_size, warmup=args.warmup, streams=args.npu_streams,
                    compute_units=args.coreml_compute_units, power_enabled=not args.no_power,
                ), max(600, (args.duration or 10) * 5)))
        elif backend == "npu":
            if "inference" in suites:
                executable = environment["runtime"]["onnxruntime"].get("executable", sys.executable)
                results.extend(
                    run_npu_benchmarks_isolated(
                        executable,
                        output_dir,
                        args.profile,
                        args.duration,
                        args.batch_size,
                        args.warmup,
                        args.npu_streams,
                        args.npu_config,
                        args.npu_model,
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

    json_path, markdown_path = save_report(output_dir, environment, _arguments_dict(args), results,
                                           paths=report_paths)
    if args.json_only:
        print(json_path.read_text(encoding="utf-8"))
    else:
        print_results(results)
        print(f"\nJSON 报告 : {json_path}")
        print(f"Markdown  : {markdown_path}")
    return 3 if any(item.get("details", {}).get("worker_failure") for item in results) else 0


def main(argv: list[str] | None = None) -> int:
    actual = list(sys.argv[1:] if argv is None else argv)
    if not actual or (actual[0].startswith("-") and actual[0] != "--version"):
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
