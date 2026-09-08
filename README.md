# SafeGrad

**SafeGrad: A Severity-Graded Benchmark for Safety Evaluation, Defense, and Attack of Text-to-Image Models**

> **Pipeline v1 (v1.0.0)** — the Adversarial Severity Ladders (ASL) pipeline and benchmark described in the SafeGrad paper: 1,026 ladders over four rungs (L0–L3) and 11 risk categories.

SafeGrad is a safety benchmark of **1,026 adversarial severity ladders** spanning 11 risk categories. Each ladder escalates through four severity levels — Safe (L0), Low-risk (L1), Mid-risk (L2), High-risk (L3) — with a paired prompt, reference image, and risk explanation at each rung. SafeGrad is constructed automatically via the **Adversarial Severity Ladders (ASL) pipeline** without manual prompt authoring.

> **Responsible use:** This dataset contains AI-generated images intended for safety research only. By using it you agree to the [Terms of Use](TERMS_OF_USE.md). Do not use for generative model training.

---

## Requirements

- Python **3.11+**
- [`uv`](https://docs.astral.sh/uv/) — used for all dependency management and script execution
- GPU with **≥24 GB VRAM** for image generation (Stage 3) and VLM scoring (Stage 4)
- Hugging Face access token for gated models (`FLUX.1-dev`, `Llama-3-70B-Instruct`)

## Install

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies into the managed virtual environment
uv sync
```

All pipeline commands use `uv run` to execute inside the managed environment:

```bash
uv run python -m safegrad.pipeline.stage1_clustering --help
```

---

## Repository layout

```
safegrad/
  pipeline/       — Four-stage ASL construction pipeline
    stage0_seed_generation.py   — Safe seed prompt generation (optional pre-stage)
    stage1_clustering.py        — Deduplication + FAISS clustering
    stage2_interpolation.py     — Severity judge + prompt synthesis (sub-stages: judge / interpolate)
    stage3_synthesis.py         — T2I reference image generation
    stage4_verification.py      — VLM monotonicity verification
    hf_auth.py / local_llm.py / local_vlm.py / utils.py — Shared utilities
  eval/           — Evaluation suite (Tables 1–6 of the paper)
  scripts/
    run_pipeline.py             — End-to-end pipeline runner
    export_metadata.py          — Export public metadata from pipeline output

data/
  rules.jsonl              — Category-specific escalation rules (used in Stages 2 & 4)
  safegrad_test_input.jsonl — 6-record paired sample for quick testing (Path A)

release/  (created by the export step below — not tracked in git)
```

The released benchmark (1,026 ladders, 4,104 reference images) is produced by
the export step and distributed separately under its access terms.

---

## ASL Pipeline

The pipeline has four stages (plus an optional seed-generation pre-stage). For details see [`safegrad/INSTRUCTIONS.md`](safegrad/INSTRUCTIONS.md).

### Stage 0 — Safe Seed Generation *(optional)*

Generate safe seed prompts. Skip if you already have a seeds or paired-input file.

```bash
uv run python -m safegrad.pipeline.stage0_seed_generation \
  --output data/seeds.jsonl --samples 50
```

### Stage 1 — Deduplication & FAISS Clustering *(CPU only)*

```bash
uv run python -m safegrad.pipeline.stage1_clustering \
  --input  data/seeds.jsonl \
  --output data/stage1_out.jsonl
```

### Stage 2 — Severity Judge + Prompt Synthesis *(GPU / API)*

Run the two sub-stages in sequence:

```bash
# 2a: severity judge
uv run python -m safegrad.pipeline.stage2_interpolation judge \
  --input  data/stage1_out.jsonl \
  --output data/stage2a_out.jsonl \
  --rules  data/rules.jsonl

# 2b: prompt interpolation
uv run python -m safegrad.pipeline.stage2_interpolation interpolate \
  --input  data/stage2a_out.jsonl \
  --output data/stage2b_out.jsonl \
  --rules  data/rules.jsonl
```

Both sub-stages support `--backend openai --base-url http://localhost:8000/v1` to serve models via vLLM.

### Stage 3 — T2I Image Synthesis *(GPU)*

```bash
uv run python -m safegrad.pipeline.stage3_synthesis \
  --input     data/stage2b_out.jsonl \
  --output    data/stage3_out.jsonl \
  --image-dir data/images/
```

### Stage 4 — VLM Monotonicity Verification *(GPU)*

```bash
uv run python -m safegrad.pipeline.stage4_verification \
  --input      data/stage3_out.jsonl \
  --output     data/stage4_out.jsonl \
  --image-root data/images/ \
  --vlm-model  Qwen/Qwen3-VL-8B-Thinking
```

### Run all stages at once

**Path B (default) — generate from scratch:**

```bash
# Stage 0 generates seeds automatically; 2b synthesises all unsafe rungs
uv run python safegrad/scripts/run_pipeline.py \
  --workdir data/pipeline_run/ \
  --rules   data/rules.jsonl

# Or skip Stage 0 if you already have a seeds file:
uv run python safegrad/scripts/run_pipeline.py \
  --seeds   data/seeds.jsonl \
  --workdir data/pipeline_run/ \
  --rules   data/rules.jsonl
```

**Path A — re-process existing paired data (adds severity judge):**

```bash
uv run python safegrad/scripts/run_pipeline.py \
  --paired-input data/safegrad_test_input.jsonl \
  --workdir      data/pipeline_run/ \
  --rules        data/rules.jsonl
```

### Export public dataset

```bash
uv run python safegrad/scripts/export_metadata.py \
  --input      data/stage4_out.jsonl \
  --image-root data/ \
  --output     release/v1/metadata.jsonl \
  --image-dir  release/v1/
```

---

## Running on a GPU cluster (SLURM)

Set environment variables and submit via `sbatch`:

```bash
export HF_TOKEN=hf_...
sbatch run_test.sh
```

The test script runs Stages 1–4 on the bundled 6-record paired sample
(`data/safegrad_test_input.jsonl`, Path A) and writes results to
`data/safegrad_test_run/`. Logs go to `logs/safegrad_test_<jobid>.out`.

For a full production run, adapt the script with your input path and desired workdir.

---

## Evaluation

Run the evaluation suite against the released dataset:

```bash
# CPU metrics (monotonicity, coverage, attack rates)
uv run python -m safegrad.eval.run_eval \
  --dataset release/v1/metadata.jsonl \
  --output-dir data/eval_results/

# HGR/SBS scoring (uses mllm_score_* fields)
uv run python -m safegrad.eval.run_eval \
  --dataset release/v1/metadata.jsonl \
  --eval-hgr
```

See [`safegrad/eval/`](safegrad/eval/) for the full experiment suite.

---

## Environment variables

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | Hugging Face access token (required for gated models) |
| `HF_HOME` | HF model cache directory (default: `~/.cache/huggingface`) |
| `OPENAI_API_KEY` | API key when using `--backend openai` |

---

## License

Dataset and code: [CC BY-NC 4.0](LICENSE) — usage is additionally governed by the
[Terms of Use](TERMS_OF_USE.md).
