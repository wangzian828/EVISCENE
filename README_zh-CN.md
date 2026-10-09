<div align="center">

# EviScene

### Beyond “Is It AI?” for Agentic Image Factuality Assessment

[English](./README.md) | [简体中文](./README_zh-CN.md)

[GitHub](https://github.com/wangzian828/EVISCENE) · [Hugging Face](https://huggingface.co/ZiAnwang/EviScene)

</div>

EviScene 核查图像所描绘的情境是否有事实证据支持，而不只是判断图像是否由 AI 生成。
仅输入一张图像，Agent 就会识别需要核查的问题、检索外部证据，并给出 supported
或 refuted 判断及基于证据的报告，无需用户额外提供文字断言。

## 项目概览

- **以图像为中心展开调查。** 检查图像内容，并随着新证据的出现逐步细化事实问题。
- **主动寻找证据。** 多模态 ReAct 策略结合视觉检查、网页搜索、图像检索和证据检查。
- **经过训练的紧凑策略模型。** 监督微调（SFT）学习经过审核的核查轨迹；特权信息辅助的
  在线自蒸馏（PSD）利用训练阶段的提示改进选定决策。部署后的学生模型不接收这些提示。
- **评测不止看分类结果。** EviLens 同时考察最终判断是否正确，以及报告中的证据是否足以
  支撑结论。

## EviLens 数据集

EviLens 是一个图像事实性基准，覆盖受控构造和网页收集的图像。
supported/refuted 标签表示事实是否得到支持，不表示图像是否由 AI 生成。

数据集发布内容为图片与配套元数据：

| 划分 | 图片数量 | 元数据 |
| --- | ---: | --- |
| 训练集 | 8,490 | 图像路径、断言、事实性标签及来源/类别字段 |
| 测试集 | 1,527 | 图像路径、断言、事实性标签及来源/类别字段 |

测试集仅用于评测，不用于训练或模型选择。这些图片与元数据划分不同于 SFT 教师轨迹
和 PSD 蒸馏目标。

## 实验结果

以下为论文中 Qwen3.5-9B 在 EviLens 测试集上的部分结果，数值单位均为百分比。

| 策略 | 平衡准确率（BAcc） | Macro F1 | SESR |
| --- | ---: | ---: | ---: |
| 直接预测 | 65.70 | 57.26 | 1.70 |
| 基础 Agent | 72.34 | 74.57 | 42.89 |
| SFT 后 | 78.81 | **75.69** | 50.49 |
| **EviScene（SFT + PSD）** | **79.05** | 74.67 | **52.26** |

SESR（Strict Evidence Sufficiency Rate，严格证据充分率）要求最终判断正确，且对
决定结论的图像事实提供充分证据。

## 代码、模型与数据

- [GitHub](https://github.com/wangzian828/EVISCENE)：Agent 运行时、工具集成、训练工具
  与评测代码。
- [Hugging Face](https://huggingface.co/ZiAnwang/EviScene)：统一存放代码、SFT 与 PSD
  权重、EviLens 图片与元数据，以及论文。

发布文件将在 arXiv 预印本上线后上传。

## 许可证

代码采用 Apache-2.0。模型权重、图片和论文分别适用各自的授权条款，代码许可证
不替代这些内容的许可证。
