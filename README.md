
# SpikeScore-GHD
### SpikeScore: Generalizable Hallucination Detection — Reproduction Guide

> **Abstract.**  
> Hallucination detection is critical for deploying large language models (LLMs) in real-world applications. Existing hallucination detection methods achieve strong performance when the training and test data come from the same domain, but they suffer from poor cross-domain generalization. We study an important yet overlooked problem, **generalizable hallucination detection (GHD)**: train on a single domain, then generalize to diverse related domains. We simulate multi-turn dialogues following the model’s initial answer and observe that hallucination-initiated dialogues exhibit **larger uncertainty fluctuations** than factual ones. We propose **SpikeScore**, which quantifies **abrupt local fluctuations** in multi-turn score trajectories. Through theory and experiments, SpikeScore shows **strong cross-domain separability** between hallucinated and non-hallucinated responses and outperforms representative baselines and generalization-oriented methods.

---

# SpikeScore-GHD

# SpikeScore: Generalizable Hallucination Detection — Reproduction Guide

> This README explains how to **replicate the experiments** from our paper **Beyond In-Domain Detection: SpikeScore for Cross-Domain Hallucination Detection**. The pipeline simulates **multi-turn dialogues** after an initial model response, records hidden states, computes uncertainty scores (e.g., **SEP**, **SAPLMA**), and aggregates them into SpikeScore for hallucination detection.

---

## Table of Contents

* [Overview](#overview)
* [Hardware & Model Requirements](#hardware--model-requirements)
* [Setup & Installation](#setup--installation)
* [Credential Configuration](#credential-configuration)
* [Step 1 — Run Multi-Turn Dialogue Generation](#step-1--run-multi-turn-dialogue-generation)
* [Step 2 — Compute SEP/RS Metrics](#step-2--compute-seprs-metrics)
* [Step 3 — Evaluate AUC](#step-3--evaluate-auc)
* [Optional — Train a Probe](#optional--train-a-probe)
* [Configuration Reference](#configuration-reference)
* [Outputs & File Structure](#outputs--file-structure)
* [Tips & Troubleshooting](#tips--troubleshooting)

---

## Overview

To reproduce the experiments:

1. **Forward-pass** prompts/answers through an LLM to collect:

   * Generated **Multi-Turn Dialogue responses**
   * **All-layer hidden states**
2. Compute **Multi-Turn Dialogue collapse** metrics (SEP or RS) from hidden states.
3. Aggregate results and report **AUC** and **distribution plots**.

> **Recommendation**
> Use **FP16 non-quantized** checkpoints. For hidden-state extraction, quantized models often require extra (de)quantization, which can reduce precision and be slower than FP16.

---

## Hardware & Model Requirements

* **GPU**: FP16 support required.
* **Typical single-GPU VRAM** (approx.):

| Model Size | VRAM     |
| ---------- | -------- |
| 7B / 8B    | \~24 GB  |
| 13B        | \~48 GB  |
| 70B        | \~160 GB |

> 70B usually requires **tensor parallelism** and/or **CPU offload**. Hidden-state capture is memory-intensive.

---

## Setup & Installation

```bash
# In your project root
pip install -r requirements.txt
```

---

## Credential Configuration

Some components require:

* [**Weights & Biases (W\&B)**](https://wandb.ai/) for logging.
* A **NIL/NLI backend** to label LLM responses (default: **OpenAI API**).

### 1) Configure NIL / OpenAI Client

Edit `./utils.py`:

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-YOUR_KEY",
    base_url="https://YOUR_ENDPOINT/v1"
)
```

### 2) Environment Variables for Semantic-Entropy Scripts

Edit `./semantic-entropy-probes/semantic_uncertainty/generate_answers.py` (ensure the path uses `/`, not a dot):

```python
import os

os.environ.setdefault("WANDB_ENT", "your-wandb-entity")
os.environ.setdefault("OPENAI_API_KEY", "sk-YOUR_KEY")
os.environ.setdefault("WANDB_API_KEY", "your-wandb-api-key")
```

Or set them in your shell (recommended):

```bash
export WANDB_ENT=your-wandb-entity
export OPENAI_API_KEY=sk-YOUR_KEY
export WANDB_API_KEY=your-wandb-api-key
```

> **Security**: Never commit API keys. Prefer environment variables over hard-coding.

---

## Step 1 — Run Multi-Turn Dialogue Generation

Run:

```bash
python ./reasoning_lite.py
```

The script reads `config.json` to configure the experiment and generates **Multi-Turn Dialogue chains** plus **hidden states**.

**Minimal example `config.json`:**

```json
{
  "model_path": "./models/xxx",
  "dataset_path": "./datasets/xxx.json",
  "output_dir": "./outputs/xxx",

  "sampling": {
    "strategy": "random",
    "n": 1000,
    "seed": 42
  },

  "enable_thinking": false,

  "strategy": "progressive",
  "max_steps": 20,
  "temperature": 0.2,
  "top_p": 0.7,

  "enable_ppl_detection": true,
  "generation_timeout": 180,
  "model_max_tokens": 128000,
  "max_new_tokens": 1024
}
```

---

## Step 2 — Compute SEP/RS Metrics

Compute collapse-style metrics from hidden states.

```bash
python ./metrics_probe.py \
  --json_dir ./outputs \
  --pkl_dir ./outputs \
  --probe_path /path/to/probe.pkl \
  --token_key last_token_embedding \
  --layer_range 13,21
```

**Arguments**

| Argument        | Type | Default                | Description                                  |
| --------------- | ---- | ---------------------- | -------------------------------------------- |
| `--json_dir`    | str  | —                      | Directory of `.json` results                 |
| `--pkl_dir`     | str  | —                      | Directory of `.pkl` hidden states            |
| `--probe_path`  | str  | —                      | Path to trained probe `.pkl`                 |
| `--token_key`   | str  | `last_token_embedding` | Hidden-state key to use                      |
| `--layer_range` | str  | `None`                 | Optional inclusive layer range, e.g. `13,21` |

The script writes predictions back into each JSON under:

```json
"metrics": {
  "probe_pred": ...
}
```

> **No probe yet?** See [Optional — Train a Probe](#optional--train-a-probe).

---

## Step 3 — Evaluate AUC

Aggregate metrics and labels, then compute **AUC** and plots:

```bash
python ./eval_probe.py \
  --data_dir ./outputs \
  --report_dir ./reports
```

* Recursively scans for `.json`/`.pkl` pairs.
* Produces **AUC** and **histograms** in `--report_dir`.

---

## Optional — Train a Probe

We provide an accelerated (async/multithreaded) pipeline adapted from [*semantic-uncertainty*](https://github.com/jlko/semantic_uncertainty) and [*semantic-entropy-probes*](https://github.com/OATML/semantic-entropy-probes). We have re-engineered the original codebase with an asynchronous, multi-threaded design, **yielding roughly a 1000× speedup.** **This is not an exaggeration**—the original repositories used synchronous, single-threaded pipelines with blocking, which meant that even on strong hardware it could take months to obtain only a small fraction of the results.

### 1) Environment

```bash
conda env update -f ./semantic-entropy-probes/sep_enviroment.yaml
conda activate se_probes
```

### 2) Generate Training Data (labels + hidden states)

```bash
python ./semantic-entropy-probes/semantic_uncertainty/generate_answers.py \
  --model_name=./models/Llama-3.2-3B-Instruct \
  --dataset=nq \
  --num_samples=2000 \
  --random_seed=42 \
  --no-compute_p_ik \
  --no-compute_p_ik_answerable \
  --p_true_num_fewshot=10 \
  --num_generations=10 \
  --num_few_shot=0 \
  --model_max_new_tokens=100 \
  --brief_prompt=chat \
  --metric=llm_gpt-4o-mini \
  --entailment_model=llm_gpt-4o-mini
```

What it does:

* Generates multiple responses per item
* Computes **semantic-entropy labels**
* Records **hidden states** for probe training

### 3) Train

From one or multiple directories:

```bash
python ./semantic-entropy-probes/probes_trainer.py \
  --data_dirs ./semantic_uncertainty/run1,./semantic_uncertainty/run2 \
  --save_path ./probe_output/probe_name.pkl
```

In code, you can specify multiple sources and naming:

```python
# From multiple directories
results = trainer.train_from_directories(
    directories=[
        './semantic_uncertainty/xxxx',
        './semantic_uncertainty/xxxx'
    ],
    dataset_names=['nq', 'nq']  # match each directory
)

# Save path pattern
exp_folder = datetime.datetime.now().strftime("llama3.2-3b_all_probe_%Y%m%d_%H%M%S")
saved_path = trainer.save_models(
    save_dir=f"./probe_output/{exp_folder}",
    prefix="llama3.2-3b_all_tbg_probe"
)
```

Use the resulting `.pkl` as `--probe_path` in **Step 2**.

---

## Configuration Reference

Key fields in `config.json`:

* **`model_path`** *(str)*: Local or HF path to the LLM checkpoint.
* **`dataset_path`** *(str)*: Input dataset in JSON/JSONL; see repo examples.
* **`output_dir`** *(str)*: Destination for JSON/PKL pairs.
* **`sampling.strategy`** *(str)*: e.g., `random`.
* **`sampling.n`** *(int)*: Number of items to sample.
* **`sampling.seed`** *(int)*: RNG seed for reproducibility.
* **`enable_thinking`** *(bool)*: Enable auxiliary “thinking” mode if supported.
* **`strategy`** *(str)*: Generation strategy, e.g., `progressive`.
* **`max_steps`** *(int)*: Max CoT steps (per item).
* **`temperature`**, **`top_p`**: Decoding parameters.
* **`enable_ppl_detection`** *(bool)*: Enable perplexity-based checks.
* **`generation_timeout`** *(int, sec)*: Per-item timeout.
* **`model_max_tokens`** *(int)*: Total context budget.
* **`max_new_tokens`** *(int)*: Max generation length.

---

## Outputs & File Structure

Each example produces a **JSON/PKL pair** under `./outputs/...`:

```
outputs/
  1da17f67a300.json   # metadata, CoT steps, NIL labels, metrics
  1da17f67a300.pkl    # hidden states captured during generation
```

**JSON (illustrative fields)**:

* `id`: unique sample id
* `question`: input prompt
* `cot_steps`: stepwise generations
* `nil_label`: entailment/consistency label from NIL
* `metrics`: populated later (e.g., `"probe_pred"`)

**PKL**:

* Serialized hidden states (layer × token tensors) for the generated answer.

---

## Tips & Troubleshooting

> **Precision & Speed**
> Prefer **FP16 non-quantized** models for faithful hidden states and faster extraction.

> **OOM / Memory**
> Reduce `max_new_tokens`, sequence length, or batch size; consider model sharding / CPU offload for large models.

> **Probe Alignment**
> Ensure the **same `token_key` and `layer_range`** are used during probe training and inference.

> **W\&B**
> First run may prompt `wandb login`. Confirm `WANDB_ENT`/`WANDB_API_KEY` as needed.

> **API Keys**
> Use environment variables; never commit secrets.

---


---

## Citation

If you find this repository or our work useful, please consider citing our paper:

```bibtex
@inproceedings{deng2025spikescore,
  title     = {Beyond In-Domain Detection: SpikeScore for Cross-Domain Hallucination Detection},
  author    = {Deng, Yongxin and Fang, Zhen and Li, Yixuan and Chen, Ling},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026}
}

