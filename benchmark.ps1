param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$BenchmarkArgs
)

$ErrorActionPreference = "Stop"
$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = if ($env:MLBENCH_VENV) { $env:MLBENCH_VENV } else { Join-Path $RootDir ".venv" }
$RequestedBackend = "all"
$RequestedProfile = "standard"
$RequestedSuite = "all"
$LLMPreset = "all"
$LLMQuantization = "all"
$LLMModel = ""
for ($Index = 0; $Index -lt $BenchmarkArgs.Count; $Index++) {
    if ($BenchmarkArgs[$Index] -match "^--backend=(.+)$") { $RequestedBackend = $Matches[1] }
    if ($BenchmarkArgs[$Index] -match "^--profile=(.+)$") { $RequestedProfile = $Matches[1] }
    if ($BenchmarkArgs[$Index] -match "^--suite=(.+)$") { $RequestedSuite = $Matches[1] }
    if ($BenchmarkArgs[$Index] -match "^--llm-preset=(.+)$") { $LLMPreset = $Matches[1] }
    if ($BenchmarkArgs[$Index] -match "^--llm-quantization=(.+)$") { $LLMQuantization = $Matches[1] }
    if ($BenchmarkArgs[$Index] -match "^--llm-dtype=(.+)$") { $LLMQuantization = $Matches[1] }
    if ($BenchmarkArgs[$Index] -match "^--llm-model=(.+)$") { $LLMModel = $Matches[1] }
    if ($BenchmarkArgs[$Index] -eq "--backend" -and $Index + 1 -lt $BenchmarkArgs.Count) {
        $RequestedBackend = $BenchmarkArgs[$Index + 1]
    }
    if ($BenchmarkArgs[$Index] -eq "--profile" -and $Index + 1 -lt $BenchmarkArgs.Count) {
        $RequestedProfile = $BenchmarkArgs[$Index + 1]
    }
    if ($BenchmarkArgs[$Index] -eq "--suite" -and $Index + 1 -lt $BenchmarkArgs.Count) {
        $RequestedSuite = $BenchmarkArgs[$Index + 1]
    }
    if ($BenchmarkArgs[$Index] -eq "--llm-preset" -and $Index + 1 -lt $BenchmarkArgs.Count) {
        $LLMPreset = $BenchmarkArgs[$Index + 1]
    }
    if ($BenchmarkArgs[$Index] -in @("--llm-quantization", "--llm-dtype") -and $Index + 1 -lt $BenchmarkArgs.Count) {
        $LLMQuantization = $BenchmarkArgs[$Index + 1]
    }
    if ($BenchmarkArgs[$Index] -eq "--llm-model" -and $Index + 1 -lt $BenchmarkArgs.Count) {
        $LLMModel = $BenchmarkArgs[$Index + 1]
    }
}
$LLMRequested = ($RequestedProfile -eq "llm") -or ($RequestedSuite.Split(",") -contains "llm")
$LLMNeedsQuantization = $LLMQuantization -in @("all", "4bit", "8bit") -or (
    $LLMQuantization -eq "auto" -and $LLMPreset -in @("kimi", "all") -and -not $LLMModel
)

function Test-PythonCode([string]$Python, [string]$Code) {
    & $Python -c $Code *> $null
    return $LASTEXITCODE -eq 0
}

function Find-Python {
    if ($env:MLBENCH_PYTHON) { return $env:MLBENCH_PYTHON }
    foreach ($Candidate in @("python", "python3", "py")) {
        $Command = Get-Command $Candidate -ErrorAction SilentlyContinue
        if ($Command) { return $Command.Source }
    }
    throw "需要 Python 3.10 或更高版本。"
}

function Invoke-Python([string]$Python, [string[]]$Arguments) {
    & $Python @Arguments
}

if ($BenchmarkArgs.Count -gt 0 -and $BenchmarkArgs[0] -eq "doctor") {
    $Python = Find-Python
    $env:PYTHONPATH = "$RootDir;$env:PYTHONPATH"
    Invoke-Python $Python @("-m", "mlbench", "doctor")
    exit $LASTEXITCODE
}

$Python = Find-Python
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$HasAccelerator = Test-PythonCode $Python @'
import sys
try:
    import torch
    if torch.cuda.is_available(): sys.exit(0)
except Exception: pass
try:
    import onnxruntime as ort
    if "VitisAIExecutionProvider" in ort.get_available_providers(): sys.exit(0)
except Exception: pass
sys.exit(1)
'@

if (-not $HasAccelerator) {
    if (-not (Test-Path $VenvPython)) {
        Write-Host "[mlbench] 创建本地环境 $VenvDir"
        Invoke-Python $Python @("-m", "venv", $VenvDir)
    }
    $Python = $VenvPython
}

if ($env:MLBENCH_NO_INSTALL -ne "1") {
    if (-not (Test-PythonCode $Python "import numpy")) {
        & $Python -m pip install "numpy>=1.23"
    }
    $HasVitis = Test-PythonCode $Python 'import onnxruntime as ort, sys; sys.exit(0 if "VitisAIExecutionProvider" in ort.get_available_providers() else 1)'
    if ($HasVitis -and -not (Test-PythonCode $Python "import onnx")) {
        Write-Host "[mlbench] 安装 NPU 测试模型生成依赖 onnx"
        & $Python -m pip install "onnx>=1.13"
    }
    if ($RequestedBackend -notin @("cpu", "npu") -and -not (Test-PythonCode $Python "import torch")) {
        if ($RequestedBackend -ne "rocm" -and (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
            $TorchIndex = $env:MLBENCH_TORCH_INDEX_URL
            if (-not $TorchIndex) {
                $Smi = (& nvidia-smi | Out-String)
                if ($Smi -match "CUDA Version:\s*(\d+)\.(\d+)") {
                    $Major = [int]$Matches[1]
                    $Minor = [int]$Matches[2]
                    if ($Major -ge 13) { $Tag = "cu130" }
                    elseif ($Major -eq 12 -and $Minor -ge 8) { $Tag = "cu128" }
                    elseif ($Major -eq 12 -and $Minor -ge 6) { $Tag = "cu126" }
                    elseif ($Major -eq 12 -and $Minor -ge 4) { $Tag = "cu124" }
                    elseif ($Major -eq 12 -and $Minor -ge 1) { $Tag = "cu121" }
                    else { $Tag = "cu118" }
                    $TorchIndex = "https://download.pytorch.org/whl/$Tag"
                }
            }
            if (-not $TorchIndex) { throw "无法判断兼容 CUDA 版本，请设置 MLBENCH_TORCH_INDEX_URL。" }
            Write-Host "[mlbench] 从 $TorchIndex 安装兼容的 PyTorch CUDA 发行包"
            & $Python -m pip install torch --index-url $TorchIndex
        } elseif ($RequestedBackend -ne "cuda" -and $env:MLBENCH_TORCH_INDEX_URL) {
            Write-Host "[mlbench] 从指定索引安装 PyTorch ROCm 发行包"
            & $Python -m pip install torch --index-url $env:MLBENCH_TORCH_INDEX_URL
        }
    }
    if ($LLMRequested -and (
        -not (Test-PythonCode $Python "import transformers") -or
        -not (Test-PythonCode $Python "import accelerate")
    )) {
        Write-Host "[mlbench] 安装端到端 LLM 测试依赖"
        & $Python -m pip install "transformers>=4.51,<5" "accelerate>=1.0" "safetensors>=0.4"
    }
    if ($LLMRequested -and $LLMNeedsQuantization -and -not (Test-PythonCode $Python "import bitsandbytes")) {
        Write-Host "[mlbench] 安装 Kimi/量化 LLM 测试依赖 bitsandbytes"
        & $Python -m pip install "bitsandbytes>=0.49"
    }
}

$env:PYTHONPATH = "$RootDir;$env:PYTHONPATH"
& $Python -m mlbench @BenchmarkArgs
exit $LASTEXITCODE
