# Heterogeneous MLBench

一个面向 **NVIDIA GPU、AMD GPU（ROCm）和 AMD Ryzen AI NPU** 的一键 ML 算力测试工具。它会自动识别本机可用后端，在同一份报告里输出矩阵算力、显存带宽、CNN 推理吞吐、功耗与性能每瓦。

## 一键运行

Linux / WSL：

```bash
chmod +x benchmark.sh
./benchmark.sh
```

Windows PowerShell：

```powershell
.\benchmark.ps1
```

默认使用 `standard` 档位并以 100 ms 间隔采样功耗，结果写入 `results/mlbench_*.json` 和 `results/mlbench_*.md`。第一次运行可能创建 `.venv` 并安装 NumPy/PyTorch；驱动和 ROCm/Ryzen AI 系统运行时不会被脚本擅自修改。

快速冒烟测试：

```bash
./benchmark.sh --profile quick
```

只检查环境，不安装依赖：

```bash
./benchmark.sh doctor
```

## 支持范围

| 硬件 | 执行后端 | 默认项目 |
|---|---|---|
| NVIDIA GPU | PyTorch CUDA + `nvidia-smi` | FP32/TF32/FP16/BF16 GEMM、显存拷贝、CNN 推理、功耗 |
| AMD GPU | PyTorch HIP/ROCm + `amd-smi`/`rocm-smi` | FP32/FP16/BF16 GEMM、显存拷贝、CNN 推理、功耗 |
| AMD Ryzen AI NPU | ONNX Runtime VitisAI EP + `xrt-smi` | 静态 CNN 吞吐、P50/P95 延迟、首次编译时间、可用时的功耗 |
| CPU 回退 | NumPy | FP32 GEMM、内存拷贝 |

PyTorch 的 ROCm 版本沿用 `torch.cuda` Python API，所以 GPU 基准核心不需要维护两份实现。NPU 使用 ONNX opset 17 的静态合成 CNN，首次运行会由 VitisAI EP 编译并缓存。

## 运行时准备

### NVIDIA GPU

先安装可用的 NVIDIA 驱动。脚本检测到 `nvidia-smi` 后，会按照驱动报告的最高 CUDA 版本选择兼容的 PyTorch 官方 wheel 索引，避免新 Torch 需要的 CUDA 运行时超过驱动能力：

```bash
./benchmark.sh --backend cuda
```

如果已有可用的 PyTorch CUDA 环境，脚本会直接复用，不创建新环境。

也可手工覆盖 wheel 索引，例如：

```bash
MLBENCH_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 \
  ./benchmark.sh --backend cuda
```

### AMD GPU / ROCm

先按 AMD 支持矩阵安装 ROCm。脚本会读取 `/opt/rocm/.info/version`，从对应的 PyTorch ROCm wheel 索引安装 `torch`：

```bash
./benchmark.sh --backend rocm
```

若 PyTorch wheel 的 ROCm 版本与系统 ROCm 不同，显式指定官方索引：

```bash
MLBENCH_TORCH_INDEX_URL=https://download.pytorch.org/whl/rocmX.Y \
  ./benchmark.sh --backend rocm
```

AMD 对部分 Radeon/Ryzen 平台提供独立 wheel；此时建议先按 AMD 文档装好 `torch`，再运行本工具。

### AMD Ryzen AI NPU

NPU 需要 AMD 提供的 NPU 驱动、XRT 和 Ryzen AI 软件包。Linux 下先激活 Ryzen AI 安装器创建的 Python 环境并加载 XRT，然后运行：

```bash
source /path/to/ryzen-ai-venv/bin/activate
source /opt/xilinx/xrt/setup.sh
./benchmark.sh --backend npu
```

工具会确认 `VitisAIExecutionProvider` 真正注册并成为首选 EP。测试图中不支持的算子可能由 CPU EP 回退执行，因此这是**端到端应用吞吐**，不是厂商标称的理论 NPU TOPS。

如当前 Ryzen AI 版本要求显式 BF16 编译配置，可使用仓库自带示例：

```bash
./benchmark.sh --backend npu --npu-config config/vai_ep_config.json
```

## 常用命令

```bash
# 测试当前 Python 中所有可用加速器（不额外跑 CPU）
./benchmark.sh --backend all

# 指定测试项与 GPU
./benchmark.sh --backend cuda --device 0 --suite compute,memory

# 两块 GPU 依次测试
./benchmark.sh --backend rocm --device 0,1 --profile extended

# 固定 GEMM 尺寸、CNN batch 和单项持续时间
./benchmark.sh --matrix-size 8192 --batch-size 16 --duration 5

# NPU 多路并发吞吐
./benchmark.sh --backend npu --npu-streams 4 --duration 10

# 调整功耗采样间隔，或完全关闭功耗采样
./benchmark.sh --power-interval 0.2
./benchmark.sh --no-power

# 仅输出机器可读 JSON
./benchmark.sh --profile quick --json-only

# 禁止启动器自动 pip install
MLBENCH_NO_INSTALL=1 ./benchmark.sh

# 明确只测 CPU 时不会安装 PyTorch GPU 依赖
./benchmark.sh --backend cpu --profile quick
```

也可以直接执行 Python 模块：

```bash
PYTHONPATH=. python -m mlbench doctor
PYTHONPATH=. python -m mlbench run --backend cpu --profile quick
```

## 指标解释

- `dense_matmul`：按 `2 × M × N × K` 计算实测 TFLOP/s，反映大矩阵乘法吞吐。
- `device_copy`：一次拷贝按一次读取加一次写入计算 GB/s，不等同于厂商标称显存带宽。
- `synthetic_cnn`：固定 224×224 输入的四层卷积网络端到端吞吐，单位 images/s。
- `synthetic_cnn_latency`：NPU 同步推理 P50；P95 和均值写在 JSON 的 `details` 中。
- `idle_power`：负载开始前的平均功耗；负载平均/峰值功耗和估算能耗写入每项结果的 `details.power`。
- `details.efficiency`：当前性能除以负载平均功耗，例如 TFLOP/s/W、images/s/W。
- 不同精度、batch、驱动、功耗模式和散热状态的结果不能直接混为同一排名。

## 说明与限制

- 一份 PyTorch wheel 只绑定一种 GPU 运行时；同机同时装有 NVIDIA GPU 和 AMD GPU 时，需要分别使用 CUDA 与 ROCm Python 环境运行，生成的 JSON 可后续合并比较。
- AMD NPU 是推理加速器，本工具不会对其运行 PyTorch 训练或通用 GEMM 测试。
- NPU 首次模型编译时间单独记录，不计入稳定态吞吐；缓存放在 `results/cache/`。
- 功耗来自驱动遥测而非外置功率计；NPU 平台若不暴露 electrical/telemetry 传感器，会明确标记为跳过。
- `quick` 用于验证，`standard` 用于日常比较，`extended` 用于较稳定的长时间测量。

## 官方运行时文档

- [PyTorch 安装选择器](https://pytorch.org/get-started/locally/)
- [PyTorch HIP/ROCm 语义](https://docs.pytorch.org/docs/stable/notes/hip.html)
- [AMD ROCm PyTorch 安装](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/native_linux/install-pytorch.html)
- [Ryzen AI Linux 安装](https://ryzenai.docs.amd.com/en/latest/linux.html)
- [Ryzen AI VitisAI EP 模型运行](https://ryzenai.docs.amd.com/en/latest/modelrun.html)
- [NVIDIA SMI 查询字段](https://docs.nvidia.com/deploy/nvidia-smi/index.html)
- [AMD SMI CLI](https://rocm.docs.amd.com/projects/amdsmi/en/latest/how-to/amdsmi-cli-tool.html)
- [XRT SMI electrical/telemetry](https://xilinx.github.io/XRT/master/html/xrt-smi.html)
