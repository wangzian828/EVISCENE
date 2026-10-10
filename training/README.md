# Training / 训练

Install utilities with `pip install -e ./training`; see `eviscene-train --help`.
GPU dependencies are separate. 训练依赖单独安装，不由代码包隐式下载权重。

| Setting / 配置 | SFT | PSD |
| --- | --- | --- |
| Initialization / 初始化 | Qwen3.5-9B | SFT step 3084 |
| Update / 更新方式 | Full parameters | LoRA r=32, alpha=32, dropout=0 |
| Learning rate / 学习率 | 1e-5 | 4e-5, constant |
| Epochs / 轮次 | 3 (3084 steps) | 5 |
| Context / 上下文 | 131072 | 131072 |
| Batch per GPU × accumulation | 1 × 1 | 1 × 32 |
| Distributed / 分布式 | FSDP2, SP=4 | FSDP2, SP=4 |

Paper-reported stack: Python 3.12, PyTorch 2.10.0 / CUDA 12.8,
Transformers 5.12.1, ms-swift 4.4.2, flash-attn 2.8.3, Liger 0.8.0;
four H20 GPUs. These are manuscript settings, not a verified launch receipt
for every checkpoint. 本表来自论文，不冒充已验证的一键复现配置。

Entry points: SFT `convert-policy` → `audit`;
PSD `build-psd-datums` → `verify-psd-datums`.
Adapters: `sft_loss_scale.py`, `psd_ms_swift.py`.
SFT weights reasoning/tool calls/final answers 1/2/2. PSD supervises only
completion prediction positions; privileged hints never enter student inputs.
训练数据须自行准备，不能用 EviLens 测试集或私有评测答案替代。
