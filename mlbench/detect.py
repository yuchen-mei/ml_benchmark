from __future__ import annotations

import glob
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .npu_runtime import npu_subprocess_environment
from .runtime import process_exit_reason


_TORCH_PROBE_PREFIX = "__MLBENCH_TORCH_PROBE__="
_ORT_PROBE_PREFIX = "__MLBENCH_ORT_PROBE__="
_TORCH_PROBE_CODE = r'''
import json

prefix = "__MLBENCH_TORCH_PROBE__="
payload = {"installed": False, "available": False, "devices": []}
try:
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
except (ImportError, OSError, ValueError):
    pass
try:
    import torch
    payload.update({
        "installed": True,
        "version": str(torch.__version__),
        "cuda_build": getattr(torch.version, "cuda", None),
        "hip_build": getattr(torch.version, "hip", None),
    })
    if not torch.cuda.is_available():
        payload["probe"] = {
            "status": "unavailable",
            "reason": "torch.cuda.is_available() returned False",
        }
    else:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            device = {
                "index": index,
                "name": properties.name,
                "memory_bytes": properties.total_memory,
                "compute_capability": list(torch.cuda.get_device_capability(index)),
            }
            architecture = getattr(properties, "gcnArchName", None)
            if architecture:
                device["architecture"] = str(architecture)
            payload["devices"].append(device)
        payload["probe"] = {"status": "probing"}
        print(prefix + json.dumps(payload, ensure_ascii=False, separators=(",", ":")), flush=True)
        torch.cuda.set_device(0)
        left = torch.ones((32, 32), dtype=torch.float16, device="cuda")
        result = torch.mm(left, left)
        float(result.sum().item())
        torch.cuda.synchronize()
        payload["available"] = True
        payload["probe"] = {"status": "ok"}
except Exception as exc:
    payload["probe"] = {
        "status": "failed",
        "reason": f"{type(exc).__name__}: {exc}",
    }
print(prefix + json.dumps(payload, ensure_ascii=False, separators=(",", ":")), flush=True)
'''
_ORT_PROBE_CODE = r'''
import json

prefix = "__MLBENCH_ORT_PROBE__="
payload = {"installed": False, "providers": []}
try:
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
except (ImportError, OSError, ValueError):
    pass
try:
    import onnxruntime as ort
    payload.update({
        "installed": True,
        "version": str(ort.__version__),
        "providers": list(ort.get_available_providers()),
    })
except Exception as exc:
    payload["error"] = f"{type(exc).__name__}: {exc}"
print(prefix + json.dumps(payload, ensure_ascii=False, separators=(",", ":")), flush=True)
'''


def _run(command: list[str], timeout: float = 5.0) -> str:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return (completed.stdout + "\n" + completed.stderr).strip()


def _nvidia_hardware() -> list[dict[str, Any]]:
    if not shutil.which("nvidia-smi"):
        return []
    output = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    devices = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",", 3)]
        if len(fields) != 4 or not fields[0].isdigit():
            continue
        devices.append(
            {
                "index": int(fields[0]),
                "name": fields[1],
                "driver": fields[2],
                "memory_mib": _to_int(fields[3]),
            }
        )
    return devices


def _to_int(value: str) -> int | None:
    try:
        return int(float(value))
    except ValueError:
        return None


def _amd_gpu_hardware() -> list[dict[str, Any]]:
    devices: list[dict[str, Any]] = []
    for vendor_path in glob.glob("/sys/class/drm/card[0-9]*/device/vendor"):
        try:
            vendor = open(vendor_path, encoding="utf-8").read().strip().lower()
        except OSError:
            continue
        if vendor != "0x1002":
            continue
        card = os.path.basename(os.path.dirname(os.path.dirname(vendor_path)))
        device_path = os.path.join(os.path.dirname(vendor_path), "device")
        try:
            device_id = open(device_path, encoding="utf-8").read().strip()
        except OSError:
            device_id = "unknown"
        device = {"card": card, "device_id": device_id}
        if device_id.lower() == "0x1586":
            device["architecture"] = "gfx1151"
        devices.append(device)

    if devices:
        architectures = _rocm_architectures()
        for index, device in enumerate(devices):
            if "architecture" not in device and index < len(architectures):
                device["architecture"] = architectures[index]
        return devices

    for architecture in _rocm_architectures():
        devices.append({"name": architecture, "architecture": architecture})
    return devices


def _rocm_architectures() -> list[str]:
    if not shutil.which("rocminfo"):
        return []
    output = _run(["rocminfo"], timeout=8.0)
    architectures = []
    for architecture in re.findall(r"\bgfx[0-9a-z]+(?::[0-9a-z:+-]+)?", output.lower()):
        if architecture not in architectures:
            architectures.append(architecture)
    return architectures


def _amd_npu_hardware() -> dict[str, Any]:
    details: dict[str, Any] = {"detected": False}
    accelerator_nodes = sorted(glob.glob("/dev/accel/accel*"))
    if accelerator_nodes:
        details["device_nodes"] = accelerator_nodes
    for vendor_path in glob.glob("/sys/class/accel/accel*/device/vendor"):
        try:
            vendor = Path(vendor_path).read_text(encoding="utf-8").strip().lower()
        except OSError:
            continue
        if vendor in {"0x1022", "0x1002"}:
            details["detected"] = True
            details["vendor"] = vendor

    if shutil.which("xrt-smi"):
        output = _run(["xrt-smi", "examine"], timeout=10.0)
        if "NPU" in output.upper() or "AIE" in output.upper():
            details.update({"detected": True, "xrt_summary": _shorten(output)})

    if platform.system() == "Linux" and shutil.which("lspci"):
        output = _run(["lspci", "-nn"])
        matching = [
            line.strip()
            for line in output.splitlines()
            if "1022:17f0" in line.lower() or ("amd" in line.lower() and "npu" in line.lower())
        ]
        if matching:
            details.update({"detected": True, "pci": matching})
    return details


def _shorten(text: str, limit: int = 800) -> str:
    compact = "\n".join(line.rstrip() for line in text.splitlines() if line.strip())
    return compact[:limit]


def _torch_runtime() -> dict[str, Any]:
    installed = importlib.util.find_spec("torch") is not None
    fallback: dict[str, Any] = {"installed": installed, "available": False, "devices": []}
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _TORCH_PROBE_CODE],
            check=False,
            capture_output=True,
            text=True,
            timeout=20.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        reason = f"PyTorch GPU 探针失败: {type(exc).__name__}: {exc}"
        fallback.update({"error": reason, "probe": {"status": "failed", "reason": reason}})
        return fallback
    payload = _probe_payload(completed.stdout)
    if payload is not None and completed.returncode == 0:
        return payload
    reason = f"PyTorch GPU 探针{process_exit_reason(completed.returncode, completed.stderr)}"
    if payload is not None:
        payload.update(
            {
                "available": False,
                "error": reason,
                "probe": {"status": "failed", "reason": reason},
            }
        )
        return payload
    fallback.update({"error": reason, "probe": {"status": "failed", "reason": reason}})
    return fallback


def _probe_payload(output: str) -> dict[str, Any] | None:
    for line in reversed(output.splitlines()):
        if line.startswith(_TORCH_PROBE_PREFIX):
            try:
                payload = json.loads(line[len(_TORCH_PROBE_PREFIX) :])
            except json.JSONDecodeError:
                return None
            return payload if isinstance(payload, dict) else None
    return None


def _ort_runtime() -> dict[str, Any]:
    executable = os.environ.get("MLBENCH_NPU_PYTHON", sys.executable)
    fallback: dict[str, Any] = {
        "installed": False,
        "providers": [],
        "executable": executable,
        "external": os.path.abspath(executable) != os.path.abspath(sys.executable),
    }
    try:
        completed = subprocess.run(
            [executable, "-c", _ORT_PROBE_CODE],
            check=False,
            capture_output=True,
            text=True,
            timeout=20.0,
            env=npu_subprocess_environment(executable),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        fallback["error"] = f"ONNX Runtime 探针失败: {type(exc).__name__}: {exc}"
        return fallback
    payload = _prefixed_payload(completed.stdout, _ORT_PROBE_PREFIX)
    if payload is None or completed.returncode != 0:
        fallback["error"] = f"ONNX Runtime 探针{process_exit_reason(completed.returncode, completed.stderr)}"
        return fallback
    payload.update({"executable": executable, "external": fallback["external"]})
    return payload


def _prefixed_payload(output: str, prefix: str) -> dict[str, Any] | None:
    for line in reversed(output.splitlines()):
        if line.startswith(prefix):
            try:
                payload = json.loads(line[len(prefix) :])
            except json.JSONDecodeError:
                return None
            return payload if isinstance(payload, dict) else None
    return None


def detect_environment() -> dict[str, Any]:
    torch_info = _torch_runtime()
    ort_info = _ort_runtime()
    nvidia = _nvidia_hardware()
    amd_gpu = _amd_gpu_hardware()
    amd_npu = _amd_npu_hardware()

    available: list[str] = []
    if torch_info.get("available"):
        available.append("rocm" if torch_info.get("hip_build") else "cuda")
    if "VitisAIExecutionProvider" in ort_info.get("providers", []):
        available.append("npu")
    available.append("cpu")

    warnings = []
    if nvidia and "cuda" not in available:
        warnings.append("检测到 NVIDIA GPU，但当前 Python 的 PyTorch CUDA 不可用。")
    if amd_gpu and "rocm" not in available:
        probe = torch_info.get("probe", {})
        if torch_info.get("installed") and probe.get("status") == "failed":
            warnings.append(
                "检测到 AMD GPU，但 PyTorch 真实张量探针失败："
                f"{probe.get('reason', 'unknown')}。已阻止原生崩溃；请更换匹配 GPU 架构的 wheel。"
            )
        else:
            warnings.append("检测到 AMD GPU，但当前 Python 的 PyTorch ROCm/HIP 不可用。")
    if amd_npu.get("detected") and "npu" not in available:
        warnings.extend(_npu_runtime_warnings(system=platform.system(), runtime=ort_info))

    return {
        "system": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "executable": sys.executable,
            "machine": platform.machine(),
        },
        "hardware": {
            "nvidia_gpus": nvidia,
            "amd_gpus": amd_gpu,
            "amd_npu": amd_npu,
        },
        "runtime": {"torch": torch_info, "onnxruntime": ort_info},
        "available_backends": available,
        "warnings": warnings,
    }


def render_doctor(environment: dict[str, Any]) -> str:
    system = environment["system"]
    runtime = environment["runtime"]
    hardware = environment["hardware"]
    lines = [
        "MLBench 环境诊断",
        "=" * 58,
        f"系统       : {system['platform']}",
        f"Python     : {system['python']} ({system['executable']})",
        f"可用后端   : {', '.join(environment['available_backends'])}",
        f"NVIDIA GPU : {_compact_json(hardware['nvidia_gpus'])}",
        f"AMD GPU    : {_compact_json(hardware['amd_gpus'])}",
        f"AMD NPU    : {_compact_json(hardware['amd_npu'])}",
        f"PyTorch    : {_runtime_line(runtime['torch'])}",
        f"ONNX RT    : {_runtime_line(runtime['onnxruntime'])}",
    ]
    if environment["warnings"]:
        lines.append("-" * 58)
        lines.extend(f"警告       : {warning}" for warning in environment["warnings"])
    return "\n".join(lines)


def _runtime_line(runtime: dict[str, Any]) -> str:
    if not runtime.get("installed"):
        line = "不可用"
        if runtime.get("external"):
            line += f"; python={runtime.get('executable')}"
        if runtime.get("error"):
            line += f"; error={_shorten(str(runtime['error']), 300)}"
        return line
    version = runtime.get("version", "unknown")
    if "providers" in runtime:
        line = f"{version}; providers={','.join(runtime['providers'])}"
        if runtime.get("external"):
            line += f"; python={runtime.get('executable')}"
        return line
    if runtime.get("hip_build"):
        backend = "HIP/ROCm"
    elif runtime.get("cuda_build"):
        backend = "CUDA"
    else:
        backend = "GPU runtime unknown"
    line = f"{version}; {backend}; accelerator={runtime.get('available', False)}"
    probe = runtime.get("probe", {})
    if probe.get("status") == "failed":
        line += f"; probe={probe.get('reason', 'failed')}"
    return line


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _npu_runtime_warnings(system: str, runtime: dict[str, Any]) -> list[str]:
    executable = runtime.get("executable", sys.executable)
    if runtime.get("installed"):
        message = (
            f"检测到 AMD NPU，但 {executable} 中的 ONNX Runtime 不含 "
            "VitisAIExecutionProvider；PyPI 通用 onnxruntime 不能驱动该 NPU。"
        )
    else:
        message = (
            "检测到 AMD NPU 内核设备，但未找到 Ryzen AI 用户态运行时。"
            "安装 AMD Ryzen AI Software 后设置 "
            "MLBENCH_NPU_PYTHON=/path/to/ryzen-ai-venv/bin/python。"
        )
    warnings = [message]
    if runtime.get("error"):
        warnings.append(f"Ryzen AI 运行时加载错误：{_shorten(str(runtime['error']), 500)}")
    if system == "Linux" and not _is_ubuntu_2404():
        warnings.append("AMD Ryzen AI Linux 发行包当前面向 Ubuntu 24.04；其他发行版不在官方支持范围。")
    return warnings


def _is_ubuntu_2404() -> bool:
    try:
        values = {}
        for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                values[key] = value.strip().strip('"')
    except OSError:
        return False
    return values.get("ID") == "ubuntu" and values.get("VERSION_ID", "").startswith("24.04")
