# SafeGrad: Data Creation Instructions

This document covers environment setup and step-by-step execution of the
four-stage ASL pipeline that constructs the SafeGrad dataset.

---

## Requirements

- Python 3.11+, [uv](https://docs.astral.sh/uv/)
- GPU with ≥24 GB VRAM for image generation (Stage 3) and LLM steps (Stage 2)
- Hugging Face access token for gated models (Llama-3-70B, FLUX.1-dev)

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

The ASL pipeline has four stages (plus an optional seed-generation pre-stage)
that transform seed prompts into a verified, image-grounded severity ladder
dataset:

```
Stage 0  →  Stage 1  →  Stage 2a  →  Stage 2b  →  Stage 3  →  Stage 4  →  Export
seed gen     dedup &     severity      prompt        T2I image   VLM mono-   public
             cluster     judge         synthesis     generation  tonicity    dataset
                                                                 verify
```

Each stage reads a JSONL and writes a JSONL.  They can be run individually or
chained via `safegrad/scripts/run_pipeline.py`.

### Two starting points

**Path A — Extend existing paired data** (e.g. the released `data/metadata.jsonl`)

Each record already contains both `prompt_safe` and `prompt_unsafe` at a
specific severity level.  Stage 1 deduplicates these pairs and clusters the
safe anchors; Stage 2a re-judges the severity labels; Stage 2b fills any
missing rungs.

```
data/metadata.jsonl  →  Stage 1  →  Stage 2a  →  Stage 2b  →  Stage 3  →  Stage 4
(89K existing pairs)    cluster     re-judge      fill gaps     images      verify
```

**Path B — Generate from scratch** (Stage 0 → Stage 2b, skipping Stage 2a)

Stage 0 produces safe-only seed records.  Stage 1 clusters these safe
anchors.  Since no unsafe prompts exist yet, Stage 2a is **skipped** — Stage
2b generates all three unsafe rungs (low/mid/high) from each safe anchor
directly.

```
Stage 0   →  Stage 1  →  Stage 2b only  →  Stage 3  →  Stage 4
safe seeds   cluster     generate all        images      verify
(no unsafe)               3 unsafe rungs
```

> Stage 2a is only useful when paired data already exists and you want to
> re-verify the severity labels.  For fresh generation, go straight from
> Stage 1 to Stage 2b.

---

## Stage 0: Safe-Prompt Seed Generation (optional)

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
| `--output` | `data/seeds.jsonl` | Output JSONL path |
| `--samples` | 50 | Prompts per category |
| `--model` | `mistral` | Red-team LLM (shorthand or full HF ID) |
| `--seed` | 42 | Random seed |
| `--dry-run` | off | Show planned counts without loading the model |

Output schema per record: `category`, `prompt_safe`, `red_team_model`, `seed`

> Skip this stage if reusing an existing seed file — the released
> `data/metadata.jsonl` already contains safe seed prompts in the flat schema
> consumed by Stage 1.

---

## Stage 1: Deduplication and Clustering

Deduplicates prompts within each `(category, severity)` bucket, clusters
safe prompts with FAISS, and assigns each record a `ladder_id` and
`cluster_safe_anchor`.  CPU-only.

```bash
uv run python -m safegrad.pipeline.stage1_clustering \
  --input  data/seeds.jsonl \
  --output data/stage1_out.jsonl
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | `data/metadata.jsonl` | Source JSONL |
| `--output` | `metadata_stage1.jsonl` | Output JSONL |
| `--model` | `all-MiniLM-L6-v2` | Sentence-transformer for embeddings |
| `--threshold` | 0.95 | Cosine similarity cutoff (paper: ≥ 0.95) |
| `--batch-size` | 512 | Embedding batch size |

---

## Stage 2: Severity Judge + Prompt Synthesis

Two sub-stages exposed as subcommands of the same module.

### Sub-stage 2a — Severity Judge

Re-judges each prompt pair against `data/rules.jsonl`, filters poisoned
ladders where the safe anchor is itself unsafe, and annotates records with
severity verdicts.

```bash
uv run python -m safegrad.pipeline.stage2_interpolation judge \
  --input       data/stage1_out.jsonl \
  --output      data/stage2a_out.jsonl \
  --model       Qwen/Qwen2.5-7B-Instruct \
  --rules       data/rules.jsonl
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `Qwen/Qwen2.5-7B-Instruct` | Judge LLM |
| `--backend` | `local` | `local` or `openai` |
| `--base-url` | — | OpenAI-compatible API base (with `--backend openai`) |
| `--concurrency` | 8 | Max concurrent LLM calls |

### Sub-stage 2b — Prompt Synthesis

Groups surviving records by `ladder_id`, fills missing severity rungs via
generative interpolation, and outputs one ladder record per cluster.

```bash
uv run python -m safegrad.pipeline.stage2_interpolation interpolate \
  --input       data/stage2a_out.jsonl \
  --output      data/stage2b_out.jsonl \
  --model       meta-llama/Llama-3-70B-Instruct \
  --rules       data/rules.jsonl
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `meta-llama/Llama-3-70B-Instruct` | Generative LLM |
| `--backend` | `local` | `local` or `openai` |
| `--base-url` | — | OpenAI-compatible API base (with `--backend openai`) |
| `--concurrency` | 4 | Max concurrent LLM calls |

Approximate combined runtime: 2–4 hours on a single GPU.

---

## Stage 3: T2I Image Synthesis

Generates one image per rung using the diffusion model specified in the
ladder's `generator_model` field.  Saves PNG files to `--image-dir`.

```bash
uv run python -m safegrad.pipeline.stage3_synthesis \
  --input     data/stage2b_out.jsonl \
  --output    data/stage3_out.jsonl \
  --image-dir data/images/
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | `data/stage2_out.jsonl` | Source JSONL |
| `--output` | `data/stage3_out.jsonl` | Output JSONL |
| `--image-dir` | `data/images/` | Directory for generated images |
| `--no-generate` | off | Dry run: skip generation |

Generated images are saved as `{image-dir}/{category}/{ladder_id}_{level}.png`
and the `image_path` field in the output JSONL is set to
`images/{category}/{ladder_id}_{level}.png`.

Default generation models (set per-ladder via the `generator_model` field):

| Alias | Hugging Face model ID |
|-------|----------------------|
| `sdxl` | `stabilityai/stable-diffusion-xl-base-1.0` |
| `flux1` | `black-forest-labs/FLUX.1-dev` |
| `large` | `stabilityai/stable-diffusion-3.5-large` |
| `zimage` | `Tongyi-MAI/Z-Image-Turbo` |

Approximate runtime: 12–24 hours per model on a single A100.

---

## Stage 4: VLM Monotonicity Verification

Scores each rung image with a vision LLM on a 0–3 scale, enforces monotonic
risk progression (`safe → low_risk → mid_risk → high_risk`) with a minimum
adjacent gap of 0.4 (paper default), and rejects broken ladders.

```bash
uv run python -m safegrad.pipeline.stage4_verification \
  --input      data/stage3_out.jsonl \
  --output     data/stage4_out.jsonl \
  --image-root data/images/ \
  --vlm-model  Qwen/Qwen3-VL-8B-Thinking
```

| Flag | Default | Description |
|------|---------|-------------|
| `--vlm-model` | `Qwen/Qwen3-VL-8B-Thinking` | Vision LLM |
| `--image-root` | `data/images/` | Root directory for rung images |
| `--min-score-gap` | 0.4 | Minimum score gap between adjacent rungs |
| `--backend` | `local` | `local` or `openai` |
| `--base-url` | — | OpenAI-compatible API base |
| `--no-checkpoint` | off | Ignore existing checkpoint; process from scratch |

Approximate runtime: 4–8 hours on a single A100.

Stage 4 checkpoints progress to `<output>.stage4_progress`.  Re-running
with the same `--output` resumes automatically.

---

## Run All Stages at Once

```bash
uv run python safegrad/scripts/run_pipeline.py \
  --input   data/seeds.jsonl \
  --workdir data/pipeline_run \
  --rules   data/rules.jsonl
```

Intermediate artifacts are written to `--workdir`:

| File | Stage |
|------|-------|
| `metadata_stage1.jsonl` | After FAISS deduplication |
| `metadata_stage2a.jsonl` | After severity judging |
| `metadata_stage2b.jsonl` | After prompt interpolation |
| `metadata_stage3.jsonl` | After image generation |
| `metadata_stage4.jsonl` | Final verified ladders |

A `run_summary.json` is produced on completion.

---

## Export the Public Dataset

After Stage 4, strip internal scoring fields and copy images for distribution:

```bash
uv run python safegrad/scripts/export_metadata.py \
  --input      data/stage4_out.jsonl \
  --image-root data/ \
  --output     release/v1/metadata.jsonl \
  --image-dir  release/v1/
```

Expected output:
- `release/v1/metadata.jsonl` — 1,026 ladder records
- `release/v1/images/` — 4,104 PNG files (4 per ladder)

Each record contains: `ladder_id`, `category`, `generator_model`,
`red_team_model`, `seed`, and `rung_{safe,low_risk,mid_risk,high_risk}` with
`prompt`, `image_path`, and `explanation` per rung.

---

## Implementation Notes

**Severity labels** are normalised via `safegrad.pipeline.utils.norm_level()`
to exactly `safe`, `low_risk`, `mid_risk`, `high_risk`.  Use that helper rather
than open-coding string cleanup.

**Deterministic seeds** — Stage 3 derives image seeds from `ladder_id + level`
via SHA-256 when no explicit seed is present.  Do not introduce randomness that
would break pipeline resumability.

**Resuming a partial run** — Stage 4 is the only stage with automatic
checkpointing (via `.stage4_progress`).  Stages 1–3 are fast enough to re-run
from scratch; if needed, filter already-processed records manually before
re-running.

**Local vs. OpenAI-compatible backends** — Stages 2 and 4 support both a
`local` backend (loads HF weights locally) and an `openai` backend (calls an
OpenAI-compatible API endpoint, e.g. vLLM).  Use `--backend openai --base-url
http://localhost:8000/v1` to serve models via vLLM for better throughput.
