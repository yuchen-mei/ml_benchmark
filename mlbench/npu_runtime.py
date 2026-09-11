from __future__ import annotations

import os
from pathlib import Path


def npu_subprocess_environment(executable: str) -> dict[str, str]:
    environment = os.environ.copy()
    runtime_root = _runtime_root(executable)
    installation_path = Path(
        environment.get("RYZEN_AI_INSTALLATION_PATH", str(runtime_root))
    ).expanduser()
    xrt_path = Path(environment.get("XILINX_XRT", "/opt/xilinx/xrt"))
    environment["RYZEN_AI_INSTALLATION_PATH"] = str(installation_path)
    environment["XILINX_XRT"] = str(xrt_path)
    if os.name == "posix":
        environment["PATH"] = _prepend(
            environment.get("PATH", ""),
            [str(xrt_path / "bin")],
        )
        environment["LD_LIBRARY_PATH"] = _prepend(
            environment.get("LD_LIBRARY_PATH", ""),
            _library_directories(runtime_root, installation_path, xrt_path),
        )
    return environment


def _runtime_root(executable: str) -> Path:
    return Path(executable).expanduser().absolute().parent.parent


def _library_directories(
    runtime_root: Path,
    installation_path: Path,
    xrt_path: Path,
) -> list[str]:
    roots = _unique_paths([runtime_root, installation_path])
    directories = [
        xrt_path / "lib",
        Path("/lib/x86_64-linux-gnu"),
    ]
    directories.extend(root / "onnxruntime" / "lib" for root in roots)
    for root in roots:
        site_packages = sorted((root / "lib").glob("python*/site-packages"))
        for site_package in site_packages:
            directories.extend(
                [
                    site_package / "lnx64.o" / "tools" / "peano" / "lib",
                    site_package / "lib" / "lnx64.o",
                    site_package / "onnxruntime" / "capi",
                    site_package / "voe" / "lib",
                ]
            )
        directories.extend(
            [
                root / "flexml_extras" / "lib",
                root / "deployment" / "lib",
            ]
        )
    return [str(path) for path in _unique_paths(directories)]


def _unique_paths(paths: list[Path]) -> list[Path]:
    unique = []
    for path in paths:
        if path not in unique:
            unique.append(path)
    return unique


def _prepend(current: str, entries: list[str]) -> str:
    values = []
    for value in [*entries, *current.split(os.pathsep)]:
        if value and value not in values:
            values.append(value)
    return os.pathsep.join(values)
