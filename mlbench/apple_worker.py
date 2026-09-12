from __future__ import annotations

import json
import sys
import traceback

from .gpu_worker import _disable_core_dumps


def probe(backend: str) -> dict:
    info = {"installed": False, "available": False}
    try:
        if backend == "mlx":
            import mlx.core as mx
            from importlib.metadata import version
            info.update(installed=True, version=version("mlx"))
            if not mx.metal.is_available():
                raise RuntimeError("MLX Metal GPU 不可用")
            mx.set_default_device(mx.gpu)
            x = mx.ones((32, 32), dtype=mx.float16)
            value = x @ x
            mx.eval(value)
            mx.synchronize()
            if float(value[0, 0].item()) != 32:
                raise RuntimeError("MLX 矩阵乘探针结果错误")
            info["device"] = mx.device_info()
        elif backend == "coreml":
            import coremltools as ct
            info.update(installed=True, version=ct.__version__)
            devices = ct.models.MLModel.get_available_compute_devices()
            info["devices"] = [type(device).__name__ for device in devices]
            if not any("NeuralEngine" in name for name in info["devices"]):
                raise RuntimeError("Core ML 未枚举到 Neural Engine")
        else:
            raise ValueError(backend)
        info["available"] = True
    except Exception as exc:
        info["reason"] = f"{type(exc).__name__}: {exc}"
    return info


def main() -> int:
    _disable_core_dumps()
    mode, backend = sys.argv[1:3]
    if mode == "probe":
        print("__MLBENCH_APPLE_PROBE__=" + json.dumps(probe(backend)))
        return 0
    try:
        config = json.loads(sys.stdin.read())
        if backend == "coreml":
            from .coreml import run_coreml_benchmarks
            results = run_coreml_benchmarks(**config)
        elif backend == "mlx":
            if config.pop("llm", False):
                from .mlx_backend import run_mlx_llm
                results = run_mlx_llm(**config)
            else:
                from .mlx_backend import run_mlx_benchmarks
                results = run_mlx_benchmarks(**config)
        else:
            raise ValueError(backend)
    except Exception:
        traceback.print_exc()
        return 2
    print("__MLBENCH_GPU_RESULT__=" + json.dumps(results, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
