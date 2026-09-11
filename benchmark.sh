#!/usr/bin/env bash
set -Eeuo pipefail
ulimit -c 0 2>/dev/null || true

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

log() {
  printf '[mlbench] %s\n' "$*"
}

has_module() {
  "$1" -c "import $2" >/dev/null 2>&1
}

has_torch_accelerator() {
  "$1" -c 'import subprocess, sys
code = r"""
try:
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
except (ImportError, OSError, ValueError):
    pass
import torch
if not torch.cuda.is_available():
    raise SystemExit(1)
value = torch.ones((32, 32), dtype=torch.float16, device="cuda")
value = torch.mm(value, value)
float(value.sum().item())
torch.cuda.synchronize()
"""
try:
    completed = subprocess.run(
        [sys.executable, "-c", code],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=20,
    )
except (OSError, subprocess.SubprocessError):
    raise SystemExit(1)
raise SystemExit(0 if completed.returncode == 0 else 1)' >/dev/null 2>&1
}

has_vitis_runtime() {
  PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" "$1" -c 'import subprocess, sys
from mlbench.npu_runtime import npu_subprocess_environment
code = r"""
try:
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
except (ImportError, OSError, ValueError):
    pass
import onnxruntime as ort
raise SystemExit(0 if "VitisAIExecutionProvider" in ort.get_available_providers() else 1)
"""
try:
    completed = subprocess.run(
        [sys.executable, "-c", code],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=20,
        env=npu_subprocess_environment(sys.executable),
    )
except (OSError, subprocess.SubprocessError):
    raise SystemExit(1)
raise SystemExit(0 if completed.returncode == 0 else 1)' >/dev/null 2>&1
}

has_accelerator_runtime() {
  has_torch_accelerator "$1" || has_vitis_runtime "$1"
}

find_npu_python() {
  local candidate directory library search_root step
  local candidates=()
  local search_roots=()
  if [[ -n "${MLBENCH_NPU_PYTHON:-}" ]]; then
    if [[ -x "${MLBENCH_NPU_PYTHON}" ]] && has_vitis_runtime "${MLBENCH_NPU_PYTHON}"; then
      printf '%s\n' "${MLBENCH_NPU_PYTHON}"
    fi
    return
  fi
  if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    candidates+=("${VIRTUAL_ENV}/bin/python")
  fi
  if [[ -n "${RYZEN_AI_INSTALLATION_PATH:-}" ]]; then
    candidates+=(
      "${RYZEN_AI_INSTALLATION_PATH}/bin/python"
      "${RYZEN_AI_INSTALLATION_PATH}/venv/bin/python"
    )
  fi
  if command -v python >/dev/null 2>&1; then
    candidates+=("$(command -v python)")
  fi
  candidates+=(
    "${HOME}"/ryzenai*/venv*/bin/python
    "${HOME}"/ryzen_ai*/venv*/bin/python
    "${HOME}"/*[Rr]yzen*/venv*/bin/python
    "${HOME}"/*[Rr]yzen*/*/venv*/bin/python
    "${HOME}"/Developer/*[Rr]yzen*/venv*/bin/python
    "${HOME}"/Developer/*[Rr]yzen*/*/venv*/bin/python
    /opt/AMD/ryzenai/venv*/bin/python
    /opt/ryzen-ai/venv*/bin/python
    /opt/ryzenai/venv*/bin/python
    /opt/*[Rr]yzen*/venv*/bin/python
  )
  for candidate in "${candidates[@]}"; do
    if [[ -x "${candidate}" ]] && has_vitis_runtime "${candidate}"; then
      printf '%s\n' "${candidate}"
      return
    fi
  done
  search_roots+=(
    "${HOME}"/*[Rr]yzen*
    "${HOME}"/Developer/*[Rr]yzen*
    /opt/*[Rr]yzen*
    /opt/AMD
  )
  for search_root in "${search_roots[@]}"; do
    [[ -d "${search_root}" ]] || continue
    while IFS= read -r library; do
      directory="$(dirname -- "${library}")"
      for ((step = 0; step < 8 && directory != "/"; step++)); do
        candidate="${directory}/bin/python"
        if [[ -x "${candidate}" ]] && has_vitis_runtime "${candidate}"; then
          printf '%s\n' "${candidate}"
          return
        fi
        directory="$(dirname -- "${directory}")"
      done
    done < <(find "${search_root}" -maxdepth 9 -type f -name libonnxruntime_vitisai_ep.so 2>/dev/null)
  done
}

configure_npu_runtime() {
  local values
  values="$(PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" "$1" -c '
import json, sys
from mlbench.npu_runtime import npu_subprocess_environment
environment = npu_subprocess_environment(sys.executable)
print(json.dumps({key: environment[key] for key in (
    "RYZEN_AI_INSTALLATION_PATH", "XILINX_XRT", "PATH", "LD_LIBRARY_PATH"
)}))
')"
  export RYZEN_AI_INSTALLATION_PATH
  RYZEN_AI_INSTALLATION_PATH="$(printf '%s' "${values}" | "$1" -c 'import json, sys; print(json.load(sys.stdin)["RYZEN_AI_INSTALLATION_PATH"])')"
  export XILINX_XRT
  XILINX_XRT="$(printf '%s' "${values}" | "$1" -c 'import json, sys; print(json.load(sys.stdin)["XILINX_XRT"])')"
  export PATH
  PATH="$(printf '%s' "${values}" | "$1" -c 'import json, sys; print(json.load(sys.stdin)["PATH"])')"
  export LD_LIBRARY_PATH
  LD_LIBRARY_PATH="$(printf '%s' "${values}" | "$1" -c 'import json, sys; print(json.load(sys.stdin)["LD_LIBRARY_PATH"])')"
}

print_npu_runtime_help() {
  cat >&2 <<'EOF'
错误: 检测到 AMD NPU 内核设备，但没有找到包含 VitisAIExecutionProvider 的 Ryzen AI Python 环境。
普通 PyPI onnxruntime 不包含 AMD NPU EP。请先安装 AMD Ryzen AI Software 1.8（Ubuntu 24.04 / Python 3.12），然后运行：
  MLBENCH_NPU_PYTHON=/path/to/ryzen-ai-venv/bin/python ./benchmark.sh --backend npu
Arch Linux 不在 AMD 当前官方 NPU 用户态支持范围；仅有 /dev/accel/accel0 还不足以运行模型。
EOF
}

requested_backend() {
  local previous=""
  local argument
  for argument in "$@"; do
    if [[ "${previous}" == "--backend" ]]; then
      printf '%s\n' "${argument}"
      return
    fi
    if [[ "${argument}" == --backend=* ]]; then
      printf '%s\n' "${argument#--backend=}"
      return
    fi
    previous="${argument}"
  done
  printf 'all\n'
}

requested_option() {
  local option="$1"
  local default="$2"
  shift 2
  local previous=""
  local argument
  for argument in "$@"; do
    if [[ "${previous}" == "${option}" ]]; then
      printf '%s\n' "${argument}"
      return
    fi
    if [[ "${argument}" == "${option}="* ]]; then
      printf '%s\n' "${argument#*=}"
      return
    fi
    previous="${argument}"
  done
  printf '%s\n' "${default}"
}

llm_requested() {
  local profile suite
  profile="$(requested_option --profile standard "$@")"
  suite="$(requested_option --suite all "$@")"
  [[ "${profile}" == "llm" || ",${suite}," == *",llm,"* ]]
}

pick_python() {
  if [[ -n "${MLBENCH_PYTHON:-}" ]]; then
    printf '%s\n' "${MLBENCH_PYTHON}"
    return
  fi
  if [[ "${REQUESTED_BACKEND:-all}" == "npu" && -n "${NPU_PYTHON:-}" ]]; then
    printf '%s\n' "${NPU_PYTHON}"
    return
  fi
  if command -v python >/dev/null 2>&1 && has_torch_accelerator "$(command -v python)"; then
    command -v python
    return
  fi
  if [[ -x "${VENV_DIR}/bin/python" ]] && has_torch_accelerator "${VENV_DIR}/bin/python"; then
    printf '%s\n' "${VENV_DIR}/bin/python"
    return
  fi
  if command -v python >/dev/null 2>&1 && has_vitis_runtime "$(command -v python)"; then
    command -v python
    return
  fi
  if [[ -x "${VENV_DIR}/bin/python" ]]; then
    printf '%s\n' "${VENV_DIR}/bin/python"
    return
  fi
  local candidate
  for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      command -v "${candidate}"
      return
    fi
  done
  return 1
}

detect_rocm_version() {
  local version=""
  local version_file
  for version_file in /opt/rocm/.info/version /opt/rocm/.info/version-dev; do
    if [[ -r "${version_file}" ]]; then
      version="$(grep -Eo '[0-9]+\.[0-9]+' "${version_file}" | head -1 || true)"
      [[ -n "${version}" ]] && break
    fi
  done
  if [[ -z "${version}" ]] && command -v hipconfig >/dev/null 2>&1; then
    version="$(hipconfig --version 2>/dev/null | grep -Eo '[0-9]+\.[0-9]+' | head -1 || true)"
  fi
  printf '%s\n' "${version}"
}

detect_amd_gpu_arch() {
  local architecture=""
  local device_file device_id
  for device_file in /sys/class/drm/card[0-9]*/device/device; do
    [[ -r "${device_file}" ]] || continue
    device_id="$(<"${device_file}")"
    case "${device_id,,}" in
      0x1586) architecture="gfx1151" ;;
    esac
    [[ -n "${architecture}" ]] && break
  done
  if [[ -z "${architecture}" ]] && command -v rocminfo >/dev/null 2>&1; then
    architecture="$(rocminfo 2>/dev/null | grep -Eo 'gfx[0-9a-z]+(:[0-9a-z:+-]+)?' | head -1 || true)"
  fi
  printf '%s\n' "${architecture%%:*}"
}

detect_cuda_index() {
  if [[ -n "${MLBENCH_TORCH_INDEX_URL:-}" ]]; then
    printf '%s\n' "${MLBENCH_TORCH_INDEX_URL}"
    return
  fi
  local version major minor tag
  version="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)"
  major="${version%%.*}"
  minor="${version#*.}"
  if [[ -z "${version}" || "${major}" == "${minor}" ]]; then
    return 1
  fi
  if (( major >= 13 )); then
    tag="cu130"
  elif (( major == 12 && minor >= 8 )); then
    tag="cu128"
  elif (( major == 12 && minor >= 6 )); then
    tag="cu126"
  elif (( major == 12 && minor >= 4 )); then
    tag="cu124"
  elif (( major == 12 && minor >= 1 )); then
    tag="cu121"
  else
    tag="cu118"
  fi
  printf 'https://download.pytorch.org/whl/%s\n' "${tag}"
}

AMD_GPU_ARCH="$(detect_amd_gpu_arch)"
if [[ -n "${MLBENCH_VENV:-}" ]]; then
  VENV_DIR="${MLBENCH_VENV}"
elif [[ "${AMD_GPU_ARCH}" == "gfx1151" ]]; then
  VENV_DIR="${ROOT_DIR}/.venv-${AMD_GPU_ARCH}"
else
  VENV_DIR="${ROOT_DIR}/.venv"
fi

if [[ "${AMD_GPU_ARCH}" == "gfx1151" ]]; then
  export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="${TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL:-1}"
  if [[ -n "${HSA_OVERRIDE_GFX_VERSION:-}" ]]; then
    log "gfx1151 架构专用 wheel 不需要 HSA_OVERRIDE_GFX_VERSION，已为本次测试移除"
    unset HSA_OVERRIDE_GFX_VERSION
  fi
  if [[ "${PYTORCH_HIP_ALLOC_CONF:-}" == *"backend:malloc"* ]]; then
    log "${AMD_GPU_ARCH} 不兼容 backend:malloc，已为本次测试移除该配置"
    unset PYTORCH_HIP_ALLOC_CONF
  fi
fi

REQUESTED_BACKEND="$(requested_backend "$@")"
NPU_PYTHON="$(find_npu_python || true)"
if [[ -n "${NPU_PYTHON}" ]]; then
  export MLBENCH_NPU_PYTHON="${NPU_PYTHON}"
  configure_npu_runtime "${NPU_PYTHON}"
elif [[ "${REQUESTED_BACKEND}" == "npu" ]]; then
  print_npu_runtime_help
  exit 2
fi

if [[ "${1:-}" == "doctor" ]]; then
  PYTHON_BIN="$(pick_python)" || {
    printf '错误: 未找到可用 Python。\n' >&2
    exit 2
  }
  export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
  exec "${PYTHON_BIN}" -m mlbench doctor
fi

PYTHON_BIN="$(pick_python)" || {
  printf '错误: 需要 Python 3.10 或更高版本。\n' >&2
  exit 2
}
LLM_PRESET="$(requested_option --llm-preset qwen "$@")"
LLM_QUANTIZATION="$(requested_option --llm-quantization auto "$@")"
LLM_MODEL="$(requested_option --llm-model "" "$@")"

if ! has_accelerator_runtime "${PYTHON_BIN}" && [[ "${PYTHON_BIN}" != "${VENV_DIR}/bin/python" ]]; then
  PYTHON_VERSION="$(${PYTHON_BIN} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  if ! "${PYTHON_BIN}" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
    printf '错误: 选择到的 Python %s 太旧，请设置 MLBENCH_PYTHON。\n' "${PYTHON_VERSION}" >&2
    exit 2
  fi
  log "创建本地环境 ${VENV_DIR} (Python ${PYTHON_VERSION})"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  PYTHON_BIN="${VENV_DIR}/bin/python"
fi

if [[ "${MLBENCH_NO_INSTALL:-0}" != "1" ]]; then
  if ! has_module "${PYTHON_BIN}" numpy; then
    log "安装基础依赖 numpy"
    "${PYTHON_BIN}" -m pip install "numpy>=1.23"
  fi

  if [[ "${REQUESTED_BACKEND}" != "cpu" && "${REQUESTED_BACKEND}" != "npu" ]] && ! has_torch_accelerator "${PYTHON_BIN}"; then
    TORCH_INSTALL_ARGS=(install --upgrade)
    if has_module "${PYTHON_BIN}" torch; then
      log "当前 PyTorch 可枚举 GPU，但真实张量探针失败；将重装匹配的发行包"
      TORCH_INSTALL_ARGS+=(--force-reinstall)
    fi
    if [[ "${REQUESTED_BACKEND}" != "rocm" ]] && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
      CUDA_INDEX="$(detect_cuda_index || true)"
      if [[ -z "${CUDA_INDEX}" ]]; then
        printf '错误: 无法从 nvidia-smi 判断兼容 CUDA 版本，请设置 MLBENCH_TORCH_INDEX_URL。\n' >&2
        exit 2
      fi
      log "检测到 NVIDIA GPU，从 ${CUDA_INDEX} 安装兼容的 PyTorch CUDA 发行包"
      "${PYTHON_BIN}" -m pip "${TORCH_INSTALL_ARGS[@]}" torch --index-url "${CUDA_INDEX}"
    elif [[ "${REQUESTED_BACKEND}" != "cuda" ]] && { command -v rocminfo >/dev/null 2>&1 || [[ -e /dev/kfd ]]; }; then
      ROCM_VERSION="$(detect_rocm_version)"
      TORCH_INDEX="${MLBENCH_TORCH_INDEX_URL:-}"
      if [[ -z "${TORCH_INDEX}" && "${AMD_GPU_ARCH}" == "gfx1151" ]]; then
        TORCH_INDEX="https://rocm.nightlies.amd.com/v2/gfx1151/"
      elif [[ -z "${TORCH_INDEX}" && -n "${ROCM_VERSION}" ]]; then
        TORCH_INDEX="https://download.pytorch.org/whl/rocm${ROCM_VERSION}"
      fi
      if [[ -z "${TORCH_INDEX}" ]]; then
        cat >&2 <<'EOF'
错误: 检测到 AMD GPU，但无法判断 ROCm 版本。
请按 PyTorch/AMD 官方兼容矩阵设置：
  MLBENCH_TORCH_INDEX_URL=https://download.pytorch.org/whl/rocmX.Y ./benchmark.sh
EOF
        exit 2
      fi
      if [[ "${AMD_GPU_ARCH}" == "gfx1151" ]]; then
        log "检测到 Strix Halo gfx1151，从架构专用索引安装 PyTorch ROCm"
      else
        log "检测到 AMD GPU，从 ${TORCH_INDEX} 安装 PyTorch ROCm 发行包"
      fi
      if ! "${PYTHON_BIN}" -m pip "${TORCH_INSTALL_ARGS[@]}" torch --index-url "${TORCH_INDEX}"; then
        cat >&2 <<EOF
错误: 没有找到与本机 Python/ROCm 匹配的 PyTorch wheel。
请查阅官方兼容矩阵，并通过 MLBENCH_TORCH_INDEX_URL 指定正确索引。
当前尝试: ${TORCH_INDEX}
EOF
        exit 2
      fi
    fi
  fi

  if llm_requested "$@"; then
    if ! has_module "${PYTHON_BIN}" transformers || ! has_module "${PYTHON_BIN}" accelerate; then
      log "安装端到端 LLM 测试依赖"
      "${PYTHON_BIN}" -m pip install "transformers>=4.51,<5" "accelerate>=1.0" "safetensors>=0.4"
    fi
    if [[ "${LLM_QUANTIZATION}" == "4bit" || "${LLM_QUANTIZATION}" == "8bit" || \
          ( "${LLM_QUANTIZATION}" == "auto" && "${LLM_PRESET}" == "kimi" && -z "${LLM_MODEL}" ) ]]; then
      if ! has_module "${PYTHON_BIN}" bitsandbytes; then
        log "安装 Kimi/量化 LLM 测试依赖 bitsandbytes"
        if ! "${PYTHON_BIN}" -m pip install "bitsandbytes>=0.49"; then
          printf '错误: bitsandbytes 安装失败，请确认当前 CUDA/ROCm GPU 在其支持范围内。\n' >&2
          exit 2
        fi
      fi
    fi
  fi
fi

if [[ "${REQUESTED_BACKEND}" != "cpu" && "${REQUESTED_BACKEND}" != "npu" ]] && \
   { command -v nvidia-smi >/dev/null 2>&1 || command -v rocminfo >/dev/null 2>&1 || [[ -e /dev/kfd ]]; } && \
   ! has_torch_accelerator "${PYTHON_BIN}"; then
  cat >&2 <<EOF
错误: PyTorch 能检测到加速器，但真实张量分配/矩阵乘探针失败，已安全停止。
GPU 架构: ${AMD_GPU_ARCH:-unknown}
Python: $(${PYTHON_BIN} --version 2>&1)
请使用匹配该 GPU 架构的 AMD/PyTorch wheel，或设置 MLBENCH_TORCH_INDEX_URL 后重试。
EOF
  exit 2
fi

export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
log "使用 $(${PYTHON_BIN} --version 2>&1) / ${PYTHON_BIN}"
exec "${PYTHON_BIN}" -m mlbench "$@"
