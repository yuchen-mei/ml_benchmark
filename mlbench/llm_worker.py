from __future__ import annotations

import json
import sys
from pathlib import Path

from .gpu_worker import _disable_core_dumps
from .llm_sweep import failure_result


def main() -> int:
    _disable_core_dumps()
    backend = sys.argv[1]
    configuration = json.loads(sys.stdin.read())
    try:
        if backend == "mlx":
            from .mlx_backend import run_mlx_llm
            results = run_mlx_llm(**configuration)
        else:
            from .llm import run_llm_benchmarks
            config = dict(configuration)
            config["cache_dir"] = Path(config["cache_dir"]) if config.get("cache_dir") else None
            results = run_llm_benchmarks(backend=backend, **config)
    except Exception as exc:
        results = [failure_result(backend, f"{backend} GPU", configuration, f"{type(exc).__name__}: {exc}")]
    print("__MLBENCH_GPU_RESULT__=" + json.dumps(results, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
