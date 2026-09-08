# Changelog

## v1.0.0 — 2026-09-08

First packaged release of the SafeGrad ASL pipeline, matching the SafeGrad
paper: 1,026 adversarial severity ladders over four rungs (L0 Safe – L3
High-risk) and 11 risk categories.

**Contents**

- Four-stage ASL construction pipeline (seed generation → clustering →
  severity judge + prompt synthesis → T2I synthesis → VLM verification),
  runnable end-to-end (`safegrad/scripts/run_pipeline.py`, Paths A/B) or
  per stage (`uv run python -m safegrad.pipeline.stageN_*`)
- Public-metadata export script (`safegrad/scripts/export_metadata.py`)
- Evaluation suite (`safegrad/eval/`) for the paper's Tables 1–6
- SLURM smoke test (`run_test.sh`) over the bundled 6-record sample

**Packaging fixes relative to the initial commit**

- README: corrected venue marker, removed references to files not shipped in
  git (`data/metadata.jsonl`, `release/v1.1/`, `scripts/test_safegrad_pipeline.sh`),
  fixed the Stage 4 example (dropped the invalid `--min-score-gap` flag), made
  the license note match the single CC BY-NC 4.0 `LICENSE` file
- `TERMS_OF_USE.md`, eval docstrings, `.github/copilot-instructions.md`:
  renamed the legacy project name to SafeGrad; rewrote stale Copilot
  instructions for the current `safegrad/` package layout
- `pyproject.toml`: version 1.0.0, paper-faithful description
- `Taskfile.yaml`: replaced the stale task list (old pre-package module paths)
  with minimal working tasks (`pipeline`, `test_gpu`, `eval`)
