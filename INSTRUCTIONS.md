# SafeGrad: Data Creation Instructions

This document covers environment setup and step-by-step execution of the
five-stage pipeline that constructs the SafeGrad dataset.

---

## Requirements

- Python 3.11+, [uv](https://docs.astral.sh/uv/)
- GPU with ≥24 GB VRAM for image generation (Stages 3–4) and LLM steps (Stage 2)
- Hugging Face access token for gated models (Llama-2, FLUX.1-dev)

Install all dependencies from the repo root:

```bash
uv sync
```

All commands below assume the repo root as the working directory.

Set your HF token once:

```bash
export HF_TOKEN=hf_...
export HF_HOME=/your/large/disk/hf-cache   # optional: redirect model cache
```

---

## Pipeline Overview

The pipeline has five stages that transform seed prompts into a verified,
image-grounded severity ladder dataset:

```
Stage 0  →  Stage 1  →  Stage 2  →  Stage 3  →  Stage 4  →  Export
seed gen     dedup &     severity    T2I image   VLM mono-   public
             cluster     interpolate  synthesis   tonicity    dataset
                                                 verify
```

Each stage reads a JSONL and writes a JSONL — they can be run individually or
chained via `safegrad/scripts/run_pipeline.py`.

---

## Stage 0: Safe-Prompt Seed Generation

Generates the initial pool of safe image prompts, one per ladder seed.
Each record feeds into Stage 1 as the `safe` anchor.

```bash
uv run python -m safegrad.pipeline.stage0_seed_generation \
  --output    data/seeds.jsonl \
  --samples   100 \
  --model     mistralai/Mistral-7B-Instruct-v0.1
```

| Flag | Default | Description |
|------|---------|-------------|
| `--output` | — | Output JSONL path |
| `--samples` | 50 | Prompts to generate per category |
| `--model` | `mistralai/Mistral-7B-Instruct-v0.1` | Red-team LLM |

Output schema per record: `category`, `prompt_safe`, `red_team_model`, `seed`

> Skip this stage if reusing an existing seed file (e.g. the released
> `data/metadata.jsonl` already contains seed prompts).

---

## Stage 1: Deduplication and Clustering

Deduplicates unsafe prompts within each `(category, severity)` bucket, clusters
safe prompts with FAISS, and assigns each record a `ladder_id` and
`cluster_safe_anchor`.

```bash
uv run python -m safegrad.pipeline.stage1_clustering \
  --input  data/seeds.jsonl \
  --output data/stage1_out.jsonl
```

This stage is CPU-only (uses `sentence-transformers` for embeddings).

---

## Stage 2: Severity Interpolation (LLM Judge)

Re-judges each prompt pair against `rules.jsonl` using an
OpenAI-compatible LLM, drops poisoned/ambiguous pairs, and fills in missing
severity rungs via generative interpolation.

```bash
uv run python -m safegrad.pipeline.stage2_interpolation \
  --input       data/stage1_out.jsonl \
  --output      data/stage2_out.jsonl \
  --llm-model   meta-llama/Llama-3-70B-Instruct \
  --judge-model Qwen/Qwen2.5-7B-Instruct
```

| Flag | Description |
|------|-------------|
| `--llm-model` | Generative model for rung interpolation |
| `--judge-model` | Judge model for severity alignment |
| `--rules` | Rules JSONL (default: `data/rules.jsonl`) |

Approximate runtime: 2–4 hours on a single GPU.

---

## Stage 3: T2I Image Synthesis

Generates one image per rung using four diffusion models. Writes ladder records
(one JSON object per ladder) and saves PNG images.

```bash
uv run python -m safegrad.pipeline.stage3_synthesis \
  --input     data/stage2_out.jsonl \
  --output    data/stage3_out.jsonl \
  --image-dir data/images/
```

Default generation models (configured in `pipeline/phase4_filter.py`):

| Alias | Hugging Face model ID |
|-------|----------------------|
| `sdxl` | `stabilityai/stable-diffusion-xl-base-1.0` |
| `flux1` | `black-forest-labs/FLUX.1-dev` |
| `large` | `stabilityai/stable-diffusion-3.5-large` |
| `zimage` | `Tongyi-MAI/Z-Image-Turbo` |

Approximate runtime: 12–24 hours per model on a single A100.

---

## Stage 4: VLM Monotonicity Verification

Scores each rung image with a vision LLM, checks severity alignment against
`data/rules.jsonl`, enforces monotonic risk progression
(`safe → low_risk → mid_risk → high_risk`), and rejects broken ladders.

```bash
uv run python -m safegrad.pipeline.stage4_verification \
  --input     data/stage3_out.jsonl \
  --output    data/stage4_out.jsonl \
  --vlm-model Qwen/Qwen3-VL-8B-Thinking
```

| Flag | Default | Description |
|------|---------|-------------|
| `--vlm-model` | `Qwen/Qwen3-VL-8B-Thinking` | VLM judge |
| `--image-root` | `data/images/` | Root directory for rung images |
| `--no-generate` | off | Skip image generation; audit only |

Approximate runtime: 4–8 hours on a single A100.

---

## Run All Stages at Once

```bash
uv run python safegrad/scripts/run_pipeline.py \
  --input   data/seeds.jsonl \
  --workdir data/pipeline_run \
  --rules   data/rules.jsonl
```

Intermediate artifacts are written to `--workdir` at each stage.
A `run_summary.json` is produced on completion.

---

## Export the Public Dataset

After Stage 4, strip internal scoring fields and resolve image paths for
the public release:

```bash
uv run python safegrad/scripts/export_metadata.py \
  --input      data/stage4_out.jsonl \
  --image-root data/images/ \
  --output     release/v1.0/export/metadata.jsonl \
  --image-dir  release/v1.0/export/images/
```

Expected output:
- `release/v1.0/export/metadata.jsonl` — 1,083 ladder records
- `release/v1.0/export/images/` — 4,332 PNG files

Each record contains only: `ladder_id`, `category`, `generator_model`,
`red_team_model`, `seed`, and `rung_{safe,low_risk,mid_risk,high_risk}` with
`prompt` and `image_path` per rung.

---

## Implementation Notes

**`safegrad/pipeline/` is a thin wrapper** over the internal `pipeline/`
modules at the repo root. The repo root is added to `sys.path` automatically
when the `safegrad` package is imported.

**Severity labels** are normalized via `pipeline.utils.norm_level()` to exactly
`safe`, `low_risk`, `mid_risk`, `high_risk`. Use that helper rather than
open-coding string cleanup.

**Deterministic seeds** — Stage 3 derives image seeds from
`ladder_id + level`; Stage 4 derives synthetic image paths from category,
level, ladder ID, and model metadata. Do not introduce randomness that
would break pipeline resumability.

**Resuming a partial run** — each stage is idempotent. Re-run the same
command; already-processed records are detected and skipped.

