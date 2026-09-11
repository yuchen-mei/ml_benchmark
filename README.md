# Heterogeneous MLBench

一个面向 **NVIDIA GPU、AMD GPU（ROCm）和 AMD Ryzen AI NPU** 的一键 ML 算力测试工具。它会自动识别本机可用后端，在同一份报告里输出矩阵算力、显存带宽、CNN 推理吞吐、真实 LLM 生成性能、功耗与性能每瓦。

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

默认使用 `standard` 档位并以 100 ms 间隔采样功耗，结果写入 `results/mlbench_*.json` 和 `results/mlbench_*.md`。第一次运行可能创建 `.venv` 并安装 NumPy/PyTorch；Strix Halo 使用独立的 `.venv-gfx1151`，只有选择 `llm` 档位时才会按需安装 Transformers/Accelerate。驱动和 ROCm/Ryzen AI 系统运行时不会被脚本擅自修改。

终端结果按设备分组，并按照 Unicode 实际显示宽度对齐中文表头；终端宽度不足时自动切换为逐项纵向布局，避免换行破坏列结构。

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
| AMD GPU | PyTorch HIP/ROCm + `amd-smi`/`rocm-smi` | FP32/FP16/BF16 GEMM、显存拷贝、CNN/兼容 MLP 推理、功耗 |
| NVIDIA / AMD GPU | Transformers `generate()` | Llama、Qwen、DeepSeek、Kimi 真实权重端到端生成、TTFT、prefill/decode 吞吐、峰值显存、功耗 |
| AMD Ryzen AI NPU | ONNX Runtime VitisAI EP + `xrt-smi` | Ryzen AI quicktest CNN 吞吐、P50/P95 延迟、首次编译时间、可用时的功耗 |
| CPU 回退 | NumPy | FP32 GEMM、内存拷贝 |

PyTorch 的 ROCm 版本沿用 `torch.cuda` Python API，所以 GPU 基准核心不需要维护两份实现。NPU 默认复用当前 Ryzen AI 安装包自带且已针对 NPU 量化的 quicktest CNN，首次运行会由 VitisAI EP 编译并缓存；也可用 `--npu-model` 指定其他 ONNX 模型。

## 24GB LLM 端到端档位

默认命令会下载并测试 Qwen3 4B 的真实权重，负载为每请求 `512` 个输入 token、强制生成 `128` 个 token、batch 1，测量 3 轮：

```bash
./benchmark.sh --profile llm
```

可选预设均以单张不超过 24GB 显存为目标：

| 预设 | 官方模型 | 默认加载方式 | 保守预检占用 | 额外要求 |
|---|---|---:|---:|---|
| `llama` | [Llama 3.2 3B Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) | BF16/FP16 | 约 7 GiB | 先接受许可并设置 `HF_TOKEN` |
| `qwen` | [Qwen3 4B Instruct 2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | BF16/FP16 | 约 9 GiB | 无 |
| `deepseek` | [DeepSeek R1 Distill Qwen 7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) | BF16/FP16 | 约 17 GiB | 无 |
| `kimi` | [Kimi VL A3B Instruct](https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct) | NF4 4bit | 约 13 GiB | `bitsandbytes`，且须显式信任仓库代码 |

```bash
# NVIDIA 或 AMD GPU；后端也可省略并自动检测
./benchmark.sh --backend cuda --profile llm --llm-preset qwen
./benchmark.sh --backend rocm --profile llm --llm-preset deepseek

# Llama 官方仓库受许可保护
HF_TOKEN=hf_xxx ./benchmark.sh --profile llm --llm-preset llama

# Kimi 官方小型开放权重仍有 16B 总参数，因此默认采用 4bit
./benchmark.sh --profile llm --llm-preset kimi --trust-remote-code
```

安全预算取“物理显存、当前空闲显存、24 GiB”三者的最小值，再默认保留 2 GiB。工具先按预设估算拦截，再检查模型实际占用和生成峰值；模型会被强制完整放在一张 GPU 上，检测到 CPU/磁盘卸载便终止，避免用变慢的 offload 结果冒充 GPU 成绩。48GB 卡也仍按 24GB 上限测试，16GB 卡则自动收紧到当前可用容量。

可调整请求尺寸、轮数和显存保留量：

```bash
./benchmark.sh --profile llm \
  --llm-preset qwen \
  --llm-prompt-tokens 1024 \
  --llm-new-tokens 256 \
  --llm-runs 5 \
  --batch-size 1
```

也可覆盖为任意兼容 Transformers 的本地目录或 Hugging Face 模型；未知模型不采用预设显存估算，但加载后与生成峰值的硬限制仍生效：

```bash
./benchmark.sh --profile llm \
  --llm-model /models/my-causal-lm \
  --llm-quantization none \
  --llm-local-files-only
```

`llm` 不会加入普通 `--suite all`，以免日常测试意外下载数 GB 权重；也不与 GEMM/CNN 混跑，以免缓存和显存状态污染功耗结果。当前通用 LLM 档位只支持 CUDA/ROCm GPU，Ryzen AI NPU 仍运行静态 CNN 路径。Kimi 的 4bit 路径依赖 [bitsandbytes 支持的硬件后端](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/docs/source/installation.mdx)，部分 ROCm 版本或 GPU 可能暂不兼容。

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

#### Ryzen AI Max / Strix Halo (`gfx1151`)

通用 PyTorch ROCm wheel 可能做到 `torch.cuda.is_available() == True`，却在第一次显存分配时原生 `SIGSEGV`。启动器现在会执行真实 FP16 分配与矩阵乘探针，而不是只检查枚举结果；检测到 `gfx1151` 时会隔离旧 `.venv`，自动使用 `.venv-gfx1151` 和架构专用 TheRock wheel 索引，并移除不再需要的 `HSA_OVERRIDE_GFX_VERSION` 兼容伪装：

```bash
./benchmark.sh --backend rocm
```

所有合成 GPU 测试也在子进程中运行。即使驱动、MIOpen 或 PyTorch 再次原生崩溃，主程序仍会生成带诊断信息的报告并以状态码 3 退出，而不会出现整个 `benchmark.sh` core dump。由于 `gfx1151` 的 MIOpen Conv2d 仍存在版本相关崩溃，推理项会明确改名为 `synthetic_mlp`；其他 AMD GPU 继续运行 `synthetic_cnn`。

AMD 当前生产支持矩阵验证的是 Ubuntu 24.04、Python 3.12、PyTorch 2.9.1 与 ROCm 7.2.1。Arch Linux 属于尽力支持；如需固定其他架构专用索引，可覆盖：

```bash
MLBENCH_TORCH_INDEX_URL=https://rocm.nightlies.amd.com/v2/gfx1151/ \
  ./benchmark.sh --backend rocm
```

### AMD Ryzen AI NPU

NPU 需要 AMD 提供的 NPU 驱动、XRT 和完整 Ryzen AI 软件包。`/dev/accel/accel0` 只代表内核驱动已枚举设备；PyPI 的通用 `onnxruntime` 不包含 `VitisAIExecutionProvider`，不能驱动 NPU。AMD 的 Linux 安装包需要单独下载并接受许可，因此本脚本不会用普通 wheel 伪装成 NPU 运行时。

Ryzen AI 1.8 的官方 Linux 流程要求 Ubuntu 24.04 和 Python 3.12。安装并通过包内 `quicktest.py` 后，可直接指定安装器创建的 venv，无需手工激活：

```bash
MLBENCH_NPU_PYTHON=/path/to/ryzen-ai-venv/bin/python \
  ./benchmark.sh --backend npu
```

若设置了 `RYZEN_AI_INSTALLATION_PATH`，或 venv 位于常见的 `$HOME/ryzenai*/venv*`、`/opt/AMD/ryzenai/venv*` 路径，启动器会自动发现它，并补齐 `XILINX_XRT`、`PATH` 和 `LD_LIBRARY_PATH`。针对 Ryzen AI 1.8，脚本还会自动加入 Peano 依赖目录，并把系统 XRT 放在厂商 venv 自带旧 XRT 之前，避免 EP 被误判为不可用或在创建 runner 时崩溃。`--backend all` 可以让 GPU 使用 ROCm venv、NPU 使用 Ryzen AI venv，结果仍汇总到同一报告。

工具会确认 `VitisAIExecutionProvider` 注册并成为首选 EP，还会读取 ONNX Runtime profiling，要求至少一个模型节点实际由 VitisAI EP 执行。测试图中不支持的算子仍可能由 CPU EP 回退，因此这是**端到端应用吞吐**，不是厂商标称的理论 NPU TOPS。

Arch Linux 当前不在 AMD 官方 Ryzen AI 用户态支持范围。即使能看到 `/dev/accel/accel0`，仍需自行移植完整 Ryzen AI 1.8 用户态，或在 AMD 支持的 Ubuntu 24.04 环境中运行；仅安装 `amdxdna`/XRT 无法补出 Vitis AI EP。

如需测试自己的模型或显式编译配置，可使用：

```bash
./benchmark.sh --backend npu \
  --npu-model /path/to/model.onnx \
  --npu-config /path/to/vai_ep_config.json
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

# 真实 Qwen 端到端生成；LLM 依赖只在此时安装
./benchmark.sh --profile llm --llm-preset qwen

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
- `synthetic_mlp`：`gfx1151` 的无卷积兼容推理负载，单位 samples/s；不可与 `synthetic_cnn` 横向比较。
- `ryzen_ai_quicktest_cnn_latency`：NPU quicktest CNN 同步推理 P50；P95 和均值写在 JSON 的 `details` 中。
- `llm_ttft_p50`：输入已驻留 GPU 后，模型 prefill 与贪心选出首 token 的 P50 时间；P95/均值写在 `details`。
- `llm_prefill`：以输入 token 数除以 TTFT 得到的 prefill 吞吐估计。
- `llm_decode`：在同一条 KV-cache 生成链中直接测量首 token 之后的持续解码吞吐，不使用两次独立计时相减。
- `llm_output_e2e`：完整请求的输出 token 吞吐，计入输入搬运、`generate()` 和输出解码。
- `llm_e2e_p50`：完整请求 P50 延迟；P95/均值写在 `details`。
- `llm_peak_vram`：预热后完整生成期间 PyTorch 记录的峰值显存；模型静态占用同时写入 `details.model_vram_gib`。
- `idle_power`：负载开始前的平均功耗；负载平均/峰值功耗和估算能耗写入每项结果的 `details.power`。
- `details.efficiency`：当前性能除以负载平均功耗，例如 TFLOP/s/W、images/s/W。
- `details.energy_per_output_token_j`：LLM 完整生成阶段的估算焦耳/token。
- 不同精度、batch、驱动、功耗模式和散热状态的结果不能直接混为同一排名。

## 说明与限制

- 一份 PyTorch wheel 只绑定一种 GPU 运行时；同机同时装有 NVIDIA GPU 和 AMD GPU 时，需要分别使用 CUDA 与 ROCm Python 环境运行，生成的 JSON 可后续合并比较。
- AMD NPU 是推理加速器，本工具不会对其运行 PyTorch 训练或通用 GEMM 测试。
- NPU 首次模型编译时间单独记录，不计入稳定态吞吐；缓存放在 `results/cache/`。
- 功耗来自驱动遥测而非外置功率计；NPU 平台若不暴露 electrical/telemetry 传感器，会明确标记为跳过。
- `quick` 用于验证，`standard` 用于日常比较，`extended` 用于较稳定的长时间测量。
- `llm` 使用真实模型权重，首次运行的下载与模型加载时间单独记录，不计入生成吞吐；不同模型、量化方式和 token 长度不能直接混为同一排名。
- `llm` 默认设置 `TORCH_DISABLE_NATIVE_JIT=1`，避开需要本机 C 编译器与 Python 开发头文件的实验性 PyTorch native-JIT 路由；已配置完整编译环境时可显式设为 `0`，但两种模式的成绩不应直接混比。

## 官方运行时文档

- [PyTorch 安装选择器](https://pytorch.org/get-started/locally/)
- [PyTorch HIP/ROCm 语义](https://docs.pytorch.org/docs/stable/notes/hip.html)
- [AMD ROCm PyTorch 安装](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/native_linux/install-pytorch.html)
- [AMD Ryzen APU Linux 支持矩阵](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityryz/native_linux/native_linux_compatibility.html)
- [TheRock `gfx1151` 架构专用 wheel](https://github.com/ROCm/TheRock/blob/main/docs/packaging/legacy_per_family_releases.md)
- [Ryzen AI Linux 安装](https://ryzenai.docs.amd.com/en/latest/linux.html)
- [Ryzen AI 1.8 发布说明](https://ryzenai.docs.amd.com/en/main/relnotes.html)
- [Ryzen AI VitisAI EP 模型运行](https://ryzenai.docs.amd.com/en/latest/modelrun.html)
- [NVIDIA SMI 查询字段](https://docs.nvidia.com/deploy/nvidia-smi/index.html)
- [AMD SMI CLI](https://rocm.docs.amd.com/projects/amdsmi/en/latest/how-to/amdsmi-cli-tool.html)
- [XRT SMI electrical/telemetry](https://xilinx.github.io/XRT/master/html/xrt-smi.html)
- [Transformers 文本生成参数](https://huggingface.co/docs/transformers/main_classes/text_generation)
- [Transformers 大模型加载与 device map](https://huggingface.co/docs/transformers/main/models)
