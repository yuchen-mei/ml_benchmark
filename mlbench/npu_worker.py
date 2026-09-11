from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

from .npu import run_npu_benchmarks


_RESULT_PREFIX = "__MLBENCH_NPU_RESULT__="


def _disable_core_dumps() -> None:
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ImportError, OSError, ValueError):
        pass


def main() -> int:
    _disable_core_dumps()
    try:
        configuration = json.loads(sys.stdin.read())
        configuration["output_dir"] = Path(configuration["output_dir"])
        results = run_npu_benchmarks(**configuration)
    except Exception:
        traceback.print_exc()
        return 2
    print(_RESULT_PREFIX + json.dumps(results, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
