# Data interface / 数据接口

**Published dataset / 发布数据：** images + paired metadata; train/test are separate.
Labels and reference annotations must not enter the agent prompt.
图片配套元数据；训练集与测试集分开。标签和参考答案不进入 Agent 输入。

**Agent input / 推理输入：** `runtime_input/cases.jsonl`, one object per line:

```json
{"case_id":"sample-001","image_path":"images/sample-001.jpg","image_sha256":"<64-character SHA-256>"}
```

Only these three keys are allowed. Image paths are relative to `runtime_input/`.
The package also needs `manifest.json`; use
`python -m scripts.prepare_runtime_input --help` to build a local package from
metadata containing `case_id` and `image_path`. Other metadata is discarded.
仅保留这三个字段；转换工具不复制标签或私有标注，也不覆盖已有输出。

**Training / 训练：** SFT consumes reviewed trajectories, not image labels alone.
PSD consumes frozen student prefixes, completion tokens, teacher top-k targets,
and bound media; hints are teacher-only.
SFT 使用审核后的轨迹；PSD 使用冻结前缀、续写 token、教师 top-k 分布及绑定图片。
Judge annotations stay separate. See [training settings](../training/README.md).
