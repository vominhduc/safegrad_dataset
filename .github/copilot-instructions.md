# Copilot instructions for SafeGrad

## Commands

This repo uses `uv` and Python 3.11+.

- Install/sync dependencies: `uv sync`
- Run the full pipeline end-to-end (Path B, from scratch):
  `uv run python safegrad/scripts/run_pipeline.py --workdir data/pipeline_run --rules data/rules.jsonl`
- Run the pipeline from existing paired data (Path A):
  `uv run python safegrad/scripts/run_pipeline.py --paired-input data/safegrad_test_input.jsonl --workdir data/pipeline_run --rules data/rules.jsonl`
- Run an individual stage:
  - `uv run python -m safegrad.pipeline.stage1_clustering --input <jsonl> --output <jsonl>`
  - `uv run python -m safegrad.pipeline.stage2_interpolation judge --input <jsonl> --output <jsonl> --rules data/rules.jsonl`
  - `uv run python -m safegrad.pipeline.stage2_interpolation interpolate --input <jsonl> --output <jsonl> --rules data/rules.jsonl`
  - `uv run python -m safegrad.pipeline.stage3_synthesis --input <jsonl> --output <jsonl> --image-dir data/images/`
  - `uv run python -m safegrad.pipeline.stage4_verification --input <jsonl> --output <jsonl> --image-root data/images/`
- Cluster smoke test (SLURM): `sbatch run_test.sh`

There is no repository-defined lint or build command in `pyproject.toml`.

## High-level architecture

`safegrad/scripts/run_pipeline.py` orchestrates the four-stage ASL pipeline and
writes every intermediate artifact into a workdir plus a final `run_summary.json`.

1. `stage0_seed_generation.py` (optional): generates safe seed prompts with a
   pool of aligned/unaligned LLMs.
2. `stage1_clustering.py`: exact + embedding (FAISS) de-duplication of seeds.
3. `stage2_interpolation.py`: two sub-commands. `judge` (2a, Path A) re-judges
   existing prompt pairs against `data/rules.jsonl`; `interpolate` (2b)
   escalates each seed through the four severity rungs with an LLM, then a
   ladder-quality scorer (2c, run by the orchestrator) rejects rungs whose
   adjacent risk gap is below the minimum.
4. `stage3_synthesis.py`: renders reference images per rung with a weighted
   T2I model mix (default `sdxl:55.8 zimage:26.8 flux1:16.3 large:1.1`), using
   the same model and seed for all rungs of a ladder.
5. `stage4_verification.py`: VLM (default Qwen3-VL-8B-Thinking) verifies
   prompt–image alignment and per-rung severity match; failing ladders are
   rejected or regenerated.

`safegrad/scripts/export_metadata.py` strips internal fields and copies images
for the public release. `safegrad/eval/` contains the paper's evaluation suite
(monotonicity, HGR/SBS, defenses, attacks, filters).

## Key conventions

- Severity labels are normalized centrally via
  `safegrad.pipeline.utils.norm_level()` to exactly `safe`, `low_risk`,
  `mid_risk`, `high_risk` (`LEVELS_ORDERED`). Reuse that helper instead of
  open-coding string cleanup.
- Escalation rules live in `data/rules.jsonl` (11 categories × 4 rungs);
  category handling is case-insensitive in pipeline logic, but record data may
  contain mixed-case values — preserve originals unless a stage explicitly
  normalizes them for lookup.
- Inter-stage interchange is JSONL: each stage consumes the previous stage's
  output file, so runs are resumable via the orchestrator's `--start-from`.
