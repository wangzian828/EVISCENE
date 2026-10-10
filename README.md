<div align="center">

<h1>EviScene</h1>

<strong>Beyond “Is It AI?” for Agentic Image Factuality Assessment</strong>

<p>Inspect the image. Find the evidence. Verify the facts.</p>

<p>
  <a href="https://wangzian828.github.io/EVISCENE/"><img src="assets/badge-website.svg" alt="Project website" height="36"></a>
  <a href="https://arxiv.org/abs/2610.14609"><img src="assets/badge-arxiv.svg" alt="Read the paper on arXiv" height="36"></a>
  <a href="https://huggingface.co/papers/2610.14609"><img src="assets/badge-hf-paper.svg" alt="Paper page on Hugging Face" height="36"></a>
  <a href="https://huggingface.co/ZiAnwang/EviScene"><img src="assets/badge-models-data.svg" alt="Models and data on Hugging Face" height="36"></a>
</p>

[English](./README.md) · [简体中文](./README_zh-CN.md)

[Overview](#overview) · [Quick Start](#quick-start) · [Models & Data](#models-and-data) · [Results](#results) · [Training](#training) · [Evaluation](#evaluation) · [Citation](#citation)

</div>

EviScene investigates whether the facts depicted in an image are supported by
evidence—not simply whether the image was AI-generated. Given an image, the
agent inspects visual details, searches for external evidence, and produces a
verdict with an evidence-grounded report. No user-written claim is required
at inference time.

This repository provides the agent runtime, tool integrations, SFT and PSD
training utilities, and evaluation code.

<a id="overview"></a>

## 🔎 Overview

![EviScene: an image-only agent that iteratively inspects visual details, retrieves evidence, and writes a grounded report.](assets/architecture.webp)

- **Evidence-seeking investigation.** A single ReAct policy chooses visual
  inspection, web search, image retrieval, or evidence inspection according
  to what remains unresolved.
- **A compact, trained policy.** Supervised fine-tuning (SFT) learns from
  reviewed teacher trajectories. Privileged on-policy self-distillation (PSD)
  refines selected decisions using training-time hints; the deployed student
  does not receive those hints.
- **Evaluation beyond the verdict.** EviLens evaluates image factuality, while
  the evaluation code measures both classification performance and whether
  the final report has sufficient factual grounds.

<details>
<summary><strong>Repository layout and command-line entry points</strong></summary>

| Directory | Purpose |
| --- | --- |
| `eviscene/` | Agent runtime, tools, and evaluation |
| `training/eviscene_training/` | SFT and PSD data preparation and training adapters |
| `scripts/` | Dataset preparation and evaluation entry points |

The installed commands are `eviscene` for inference and `eviscene-train` for
training preparation. Python modules use the same `eviscene` and
`eviscene_training` namespaces.

</details>

<a id="quick-start"></a>

## 🚀 Quick Start

### 1. Install

Use **Python 3.11 or 3.12** in a fresh environment. The commands below assume
Bash; on Windows, activate the environment with `.venv\Scripts\Activate.ps1`
in PowerShell instead of `source`.

```bash
git clone https://github.com/wangzian828/EVISCENE.git
cd EVISCENE
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[all]' -e ./training
```

For a lightweight runtime installation, use `python -m pip install -e .`; `image-retrieval` adds
OSS image-upload support. OCR uses the Baidu API only. Installing the
Python packages does **not** download an EviScene checkpoint or start a model
server. GPU training dependencies are separate; see [Training](#training).

### 2. Configure the model and tools

Start a compatible multimodal model server with your EviScene checkpoint. It
must expose an OpenAI-compatible chat-completions endpoint with image support.
Set the URL and served model name to match that server; the values below are
examples, not a hosted EviScene API.

<details>
<summary><strong>Required configuration: .env settings</strong></summary>

Create a local `.env` file at the repository root:

```dotenv
# Agent policy and visual tools: your local multimodal server.
QWEN35_LOCAL_BASE_URL=http://127.0.0.1:8000/v1
QWEN35_LOCAL_MODEL=eviscene
QWEN_LOCAL_API_KEY=none

# Text search, image search, and reverse-image search.
SERPER_API_KEY=YOUR_SERPER_API_KEY
VISUAL_SEARCH_PROVIDER=serper_lens

# Fetch readable webpage content through Jina Reader.
BROWSE_FETCH_PROVIDER=jina
JINA_API_KEY=YOUR_JINA_API_KEY

# Upload images to obtain URLs usable by reverse-image search.
IMAGE_UPLOAD_PROVIDER=oss
OSS_ACCESS_KEY_ID=YOUR_OSS_ACCESS_KEY_ID
OSS_ACCESS_KEY_SECRET=YOUR_OSS_ACCESS_KEY_SECRET
OSS_ENDPOINT=YOUR_OSS_ENDPOINT
OSS_BUCKET_NAME=YOUR_OSS_BUCKET_NAME

# Baidu OCR API and fresh tool observations.
OCR_BACKEND=baidu
BAIDU_OCR_API_KEY=YOUR_BAIDU_OCR_API_KEY
BAIDU_OCR_SECRET_KEY=YOUR_BAIDU_OCR_SECRET_KEY
TOOL_CACHE_ENABLED=0
PERCEPTION_CACHE_ENABLED=0
```

Replace the credential placeholders and use your server's API key if required.
The `.env` file is Git-ignored. OCR sends image content to Baidu; it does not
use a local OCR model or fall back to another provider.

</details>

**Model inference can be local; external search is not offline.** Search,
webpage reading, and image hosting require network access and may incur
provider charges. Reverse-image search sends the input image to the configured
hosting/search services; use only images you are authorized to share.

<details>
<summary><strong>Advanced configuration: local model routing and overrides</strong></summary>

Project settings use the `EVISCENE_*` prefix, for example `EVISCENE_DATA_ROOT`
for artifact storage and `EVISCENE_ENV_FILE` for a custom environment file.
Existing `IFV_*` settings are accepted; an explicitly set `EVISCENE_*` value
takes precedence. Provider settings such as `QWEN35_LOCAL_MODEL` keep their names.

The CLI and Python API default to `student-qwen3.5-local`; cloud model providers
must be explicitly selected. The profile routes the policy and visual tools to the
local model. **Webpage evidence extraction defaults to the same local policy
model and endpoint**, with no extra extraction settings required. The
`BROWSE_EXTRACT_PROVIDER`, `BROWSE_EXTRACT_MODEL`, `BROWSE_EXTRACT_BASE_URL`,
`BROWSE_EXTRACT_WIRE_API`, and `BROWSE_EXTRACT_API_KEY` settings allow an explicit
override. Existing environment or `.env` overrides still take precedence;
remove a legacy `BROWSE_EXTRACT_PROVIDER=gemini` setting to use the local default.
Extraction does not fall back to Gemini if local inference fails. Gemini-based
evaluation is a separate, optional step—not part of this inference example.

</details>

### 3. Investigate an image

Run from the repository root, replacing the image path with your own file:

```bash
eviscene path/to/image.jpg --profile student-qwen3.5-local --output-dir runs/example
```

The CLI prints the verdict, confidence, assessment, and tool-call statistics.
It also saves the canonical trace as `runs/example/image.jpg.json` for this
filename. To inspect the reasoning, tool observations, and final report:

```bash
python -m eviscene.render_trace_html runs/example/image.jpg.json
```

Open the generated `runs/example/image.jpg.html` in a browser. The runtime's
`real` / `fake` verdict labels correspond to supported / refuted factuality;
they are not AI-generation provenance labels. Model outputs are not guaranteed
to be correct; inspect the cited evidence before relying on a conclusion.

<a id="models-and-data"></a>

## 📦 Models and Data

> **Resources:** Code, checkpoints, dataset files, and the paper are publicly
> available. The [Hugging Face EviScene repository](https://huggingface.co/ZiAnwang/EviScene)
> contains code, SFT and PSD checkpoints, the EviLens
> train/test images and metadata, and the paper PDF.

| Artifact | Contents | Availability |
| --- | --- | --- |
| [EviScene-SFT](https://huggingface.co/ZiAnwang/EviScene/tree/main/models/sft) | Qwen3.5-9B policy after supervised fine-tuning | Available |
| [EviScene-PSD](https://huggingface.co/ZiAnwang/EviScene/tree/main/models/psd) | Qwen3.5-9B policy after SFT and PSD | Available |
| [EviLens train](https://huggingface.co/ZiAnwang/EviScene/tree/main/datasets/EviLens/train) | 8,647 images with paired metadata | Available |
| [EviLens test](https://huggingface.co/ZiAnwang/EviScene/tree/main/datasets/EviLens/test) | 1,527 images with paired metadata | Available |

Code is at the Hugging Face repository root. The two checkpoints belong under
`models/sft/` and `models/psd/`; the manuscript is `paper/EviScene.pdf`.
The dataset is in `datasets/EviLens/`, with separate `train/` and `test/` folders.
Each model directory contains merged safetensors weights and the corresponding
configuration, tokenizer, and processor files. The code license does not grant
rights to model weights or third-party images; consult the model and dataset cards.
EviLens image/metadata splits are distinct from
the teacher trajectories and distillation targets used by the training
pipeline. Keep the test split out of training and model selection.

<a id="results"></a>

## 📊 Results

Selected results from the current paper snapshot on EviLens. All values are
percentages; higher is better. These are reported paper results, not scores
recomputed from this code snapshot.

| Qwen3.5-9B setting | BAcc | Macro F1 | Supported recall | Refuted recall | SESR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct prediction | 65.70 | 57.26 | 79.05 | 52.35 | 1.70 |
| Base agent | 72.34 | 74.57 | 51.19 | 93.48 | 42.89 |
| After SFT | 78.81 | 75.69 | 76.39 | 81.22 | 50.49 |
| **EviScene: SFT + PSD** | **79.05** | 74.67 | 80.11 | 78.00 | **52.26** |

SFT + PSD improves BAcc by **6.71 percentage points** and SESR by **9.37
percentage points** over the same-backbone base agent. Macro F1 and class-wise
recall are included to show the trade-offs rather than implying every metric
improves.

- **BAcc:** the mean of supported and refuted recall.
- **Macro F1:** the mean of the two class-wise F1 scores.
- **SESR:** Strict Evidence Sufficiency Rate—the percentage of cases with a
  correct verdict and sufficient grounds for the decisive image-grounded fact.

<a id="training"></a>

## 🛠️ Training

The training pipeline has two stages:

1. **SFT:** audit teacher investigations, export accepted trajectories, and
   convert them to the policy training format.
2. **PSD:** collect on-policy training rollouts, locate failed decisions,
   construct procedural hints, verify repairs, and train on selected
   teacher-distribution targets alongside preservation targets.

| Step | Implementation |
| --- | --- |
| Build reviewed SFT examples | [`build_sft_training_package.py`](scripts/trajectory/build_sft_training_package.py) |
| Convert policy examples | [`eviscene-train convert-policy`](training/eviscene_training/policy.py) |
| Collect on-policy rollouts | [`collect_psd_rollouts.py`](scripts/collect_psd_rollouts.py) |
| Localize and verify training failures | [`run_psd_judge.py`](scripts/run_psd_judge.py) |
| Construct hints and search for repairs | [`run_psd_repair_driver.py`](scripts/run_psd_repair_driver.py) |
| Build sparse teacher targets | [`build-psd-datums`](training/eviscene_training/psd_datums.py) |
| Apply the PSD training loss | [`psd_ms_swift.py`](training/eviscene_training/psd_ms_swift.py) |

Inspect the available preparation commands with:

```bash
eviscene-train --help
python -m scripts.trajectory.build_sft_training_package --help
eviscene-train build-psd-datums --help
python -m scripts.run_psd_repair_driver --help
```

The lightweight package does not install a GPU training stack or download weights.
See the short [training settings](training/README.md) and [data interface](docs/data.md).
The settings are paper-reported, not an independently verified reproduction run.

<a id="evaluation"></a>

## 🔬 Evaluation

Agent inference and evidence judging are separate. The agent receives the
image, not the reference labels or evaluator-only material. The judge reads
the saved investigation afterward.

### Run the agent

With a prepared runtime case package and the model/tool configuration above:

```bash
python -m eviscene.eval.run_cases \
  --benchmark path/to/runtime_input/cases.jsonl \
  --profile student-qwen3.5-local \
  --output-dir runs/eval-full \
  --concurrency 4 \
  --base-sampling-seed 1729
```

This saves `run_manifest.json`, `run_results.jsonl`, `summary.json`, and per-case
traces. Runtime inputs must follow the [data interface](docs/data.md);
an arbitrary image/label JSONL is not a drop-in replacement. The concurrency
above is a starting point for your server, not a claim about paper hardware.
The [recoverable rollout wrapper](scripts/run_agent_test_rollout.py) additionally
supports bounded engineering retries for compatible test releases.

### Judge saved investigations

The [post-hoc judge](scripts/audit_agent_private_gold.py) requires separately
supplied reference annotations and `GEMINI_API_KEY` or `GOOGLE_API_KEY`:

```bash
python -m scripts.audit_agent_private_gold \
  --run-dir runs/eval-full \
  --manifest path/to/test-manifest.jsonl \
  --private-gold-sidecar path/to/test-private-gold.jsonl \
  --output-dir runs/eval-full-judge \
  --judge-model gemini-3.7-flash \
  --thinking-level low \
  --max-output-tokens 32768 \
  --concurrency 4
```

This is a paid external evaluation step. The annotation paths are placeholders;
evaluator reference material is not included in this code repository. Public
image-level labels alone are insufficient to reproduce the evidence-quality
audit. Preserve the dataset version, case set, model configuration, and metric
denominator when comparing results; failed or unjudged cases are not successes.

For larger runs, the [streaming judge](scripts/server/stream_agent_judges.py)
can judge completed cases while inference continues. The
[summary script](scripts/summarize_agent_test_evaluation.py) aggregates post-hoc
audit records; the [metric definitions](eviscene/eval/private_gold_metrics.py) are
included in the repository.

### Tool ablations

Add one flag to `eviscene.eval.run_cases` and use a separate output directory:

| Variant | Flag | Removed tools |
| --- | --- | --- |
| No web search | `--disable-web-search` | `text_search` |
| No image retrieval | `--disable-image-retrieval` | `text_image_search`, `reverse_image_search` |
| No evidence inspection | `--disable-evidence-inspection` | `visit`, `compare_with_reference` |

Direct visual operations and final reporting remain available in all three
ablations. Keep the model, case set, seeds, and action budgets fixed. The exact
tool-family mapping is defined in [`tool_ablation.py`](eviscene/orchestrator/tool_ablation.py).

<a id="citation"></a>

## 📝 Citation

**EviScene: Beyond “Is It AI?” for Agentic Image Factuality Assessment.**
Read the [paper on arXiv](https://arxiv.org/abs/2610.14609)
or [download BibTeX](https://wangzian828.github.io/EVISCENE/assets/eviscene.bib).
If you use EviScene, please cite:

```bibtex
@misc{wang2026evisceneisaiagentic,
  title={EviScene: Beyond "Is It AI?" for Agentic Image Factuality Assessment},
  author={ZiAn Wang and Ruiqi Liu and Zhengbo Zhang and Yi Han and
          Jialiang Shen and Ruoxin Chen and Manni Cui and Ximo Zhu and
          Jin Jiang and Mingzhe Liu and Linhao Wang and Wenhao Wang and Shu Wu},
  year={2026},
  eprint={2610.14609},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2610.14609},
}
```

Machine-readable citation metadata is in [`CITATION.cff`](CITATION.cff).

## 🤝 Contributing

See [Contributing](CONTRIBUTING.md) for code and package checks, and
[Security](SECURITY.md) for handling credentials and external tool access.

## ⚖️ License

Code is licensed under [Apache-2.0](LICENSE); see [NOTICE](NOTICE).
Weights, datasets, and the manuscript have separate terms. The code license
does not replace upstream model, image, or service licenses.
