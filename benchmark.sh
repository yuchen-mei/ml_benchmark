#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${MLBENCH_VENV:-${ROOT_DIR}/.venv}"

log() {
  printf '[mlbench] %s\n' "$*"
}

has_module() {
  "$1" -c "import $2" >/dev/null 2>&1
}

has_accelerator_runtime() {
  "$1" -c 'import sys
try:
    import torch
    if torch.cuda.is_available():
        sys.exit(0)
except Exception:
    pass
try:
    import onnxruntime as ort
    if "VitisAIExecutionProvider" in ort.get_available_providers():
        sys.exit(0)
except Exception:
    pass
sys.exit(1)' >/dev/null 2>&1
}

has_vitis_runtime() {
  "$1" -c 'import onnxruntime as ort, sys; sys.exit(0 if "VitisAIExecutionProvider" in ort.get_available_providers() else 1)' >/dev/null 2>&1
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
  if command -v python >/dev/null 2>&1 && has_accelerator_runtime "$(command -v python)"; then
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
REQUESTED_BACKEND="$(requested_backend "$@")"
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

  if has_vitis_runtime "${PYTHON_BIN}" && ! has_module "${PYTHON_BIN}" onnx; then
    log "安装 NPU 测试模型生成依赖 onnx"
    "${PYTHON_BIN}" -m pip install "onnx>=1.13"
  fi

  if [[ "${REQUESTED_BACKEND}" != "cpu" && "${REQUESTED_BACKEND}" != "npu" ]] && ! has_module "${PYTHON_BIN}" torch; then
    if [[ "${REQUESTED_BACKEND}" != "rocm" ]] && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
      CUDA_INDEX="$(detect_cuda_index || true)"
      if [[ -z "${CUDA_INDEX}" ]]; then
        printf '错误: 无法从 nvidia-smi 判断兼容 CUDA 版本，请设置 MLBENCH_TORCH_INDEX_URL。\n' >&2
        exit 2
      fi
      log "检测到 NVIDIA GPU，从 ${CUDA_INDEX} 安装兼容的 PyTorch CUDA 发行包"
      "${PYTHON_BIN}" -m pip install torch --index-url "${CUDA_INDEX}"
    elif [[ "${REQUESTED_BACKEND}" != "cuda" ]] && { command -v rocminfo >/dev/null 2>&1 || [[ -e /dev/kfd ]]; }; then
      ROCM_VERSION="$(detect_rocm_version)"
      TORCH_INDEX="${MLBENCH_TORCH_INDEX_URL:-}"
      if [[ -z "${TORCH_INDEX}" && -n "${ROCM_VERSION}" ]]; then
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
      log "检测到 AMD GPU，从 ${TORCH_INDEX} 安装 PyTorch ROCm 发行包"
      if ! "${PYTHON_BIN}" -m pip install torch --index-url "${TORCH_INDEX}"; then
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

export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
log "使用 $(${PYTHON_BIN} --version 2>&1) / ${PYTHON_BIN}"
exec "${PYTHON_BIN}" -m mlbench "$@"
