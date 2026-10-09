<div align="center">

# EviScene

### Beyond “Is It AI?” for Agentic Image Factuality Assessment

[English](./README.md) | [简体中文](./README_zh-CN.md)

[GitHub](https://github.com/wangzian828/EVISCENE) · [Hugging Face](https://huggingface.co/ZiAnwang/EviScene)

</div>

EviScene investigates whether the situation depicted in an image is supported
by factual evidence—not simply whether the image was AI-generated. Given a
single image, the agent identifies questions to investigate, retrieves external
evidence, and produces a supported or refuted verdict with an evidence-grounded
report. No user-written claim is required.

## Overview

- **Image-centered investigation.** The agent inspects the image and refines
  factual questions as new evidence becomes available.
- **Evidence-seeking tool use.** A multimodal ReAct policy combines visual
  inspection, web search, image retrieval, and evidence inspection.
- **A compact trained policy.** Supervised fine-tuning (SFT) learns from reviewed
  investigations. Privileged on-policy self-distillation (PSD) improves selected
  decisions using training-time hints; the deployed student does not receive
  these hints.
- **Evaluation beyond classification.** EviLens measures both verdict accuracy
  and whether a report provides sufficient evidence for its conclusion.

## EviLens

EviLens is an image factuality benchmark covering images from controlled
construction and web collection. Its labels describe factual support, not
AI-generation provenance.

The dataset release consists of images and paired metadata:

| Split | Images | Metadata |
| --- | ---: | --- |
| Train | 8,490 | Image path, claim, factuality label, and provenance/category fields |
| Test | 1,527 | Image path, claim, factuality label, and provenance/category fields |

The test split is for evaluation, not training or model selection. These
image/metadata splits are distinct from SFT teacher trajectories and PSD
distillation targets.

## Results

Selected Qwen3.5-9B results from the paper on the EviLens test set. All values
are percentages.

| Policy | Balanced accuracy | Macro F1 | SESR |
| --- | ---: | ---: | ---: |
| Direct prediction | 65.70 | 57.26 | 1.70 |
| Base agent | 72.34 | 74.57 | 42.89 |
| After SFT | 78.81 | **75.69** | 50.49 |
| **EviScene (SFT + PSD)** | **79.05** | 74.67 | **52.26** |

SESR (Strict Evidence Sufficiency Rate) requires both a correct verdict and
sufficient evidence for the decisive image-grounded fact.

## Code, Models, and Data

- [GitHub](https://github.com/wangzian828/EVISCENE): agent runtime, tool
  integrations, training utilities, and evaluation code.
- [Hugging Face](https://huggingface.co/ZiAnwang/EviScene): the unified project
  repository for code, SFT and PSD checkpoints, EviLens images and metadata,
  and the paper.

The release files will be uploaded after the arXiv preprint is available.

## License

The code uses Apache-2.0. Model checkpoints, images, and the manuscript have
separate terms; the code license does not replace their respective licenses.
