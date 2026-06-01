# Copilot instructions for T2I-Escalate

## Commands

This repo uses `uv` and Python 3.11+.

- Install/sync dependencies: `uv sync`
- Run the full 5-phase pipeline: `uv run python run_pipeline.py --input data/metadata.jsonl --workdir data/pipeline_run --rules data/rules.jsonl`
- Run the end-to-end test script: `uv run python tests/mock_pipeline_test.py`
- Run an individual phase:
  - `uv run python -m pipeline.phase1_filter --input data/metadata.jsonl --output data/pipeline_run/metadata_phase1.jsonl`
  - `uv run python -m pipeline.phase2_filter --input data/pipeline_run/metadata_phase1.jsonl --output data/pipeline_run/metadata_phase2.jsonl --rules data/rules.jsonl`
  - `uv run python -m pipeline.phase3_filter --input data/pipeline_run/metadata_phase2.jsonl --output data/pipeline_run/metadata_phase3.jsonl --rules data/rules.jsonl`
  - `uv run python -m pipeline.phase4_filter --input data/pipeline_run/metadata_phase3.jsonl --output data/pipeline_run/metadata_phase4.jsonl --rules data/rules.jsonl`
  - `uv run python -m pipeline.phase5_filter --input data/pipeline_run/metadata_phase4.jsonl --output data/pipeline_run/metadata_phase5.jsonl`

There is no repository-defined lint or build command in `pyproject.toml`.

## High-level architecture

`run_pipeline.py` orchestrates a five-phase JSONL pipeline and writes every intermediate artifact into a workdir plus a final `run_summary.json`.

1. `phase1_filter.py`: de-duplicates unsafe prompts within `(category, target_severity)` buckets, clusters `prompt_safe` embeddings within each category, and assigns `ladder_id` plus `cluster_safe_anchor`.
2. `phase2_filter.py`: re-judges prompt pairs against `data/rules.jsonl` with an OpenAI-compatible LLM, drops poisoned/ambiguous pairs, and adds `judge_*` metadata. If a category is missing from the rules file, it falls back to the nearest rule category via sentence-transformer embeddings.
3. `phase3_filter.py`: groups Phase 2 records by `ladder_id`, converts the flat pair stream into ladder records, and fills missing unsafe rungs with a generative LLM. This is the schema pivot in the repo: phases 1-2 operate on pair records; phases 3-5 operate on one JSON object per ladder.
4. `phase4_filter.py`: resolves or locally generates rung images, scores each rung with a vision LLM, checks each rung against the category rules from `data/rules.jsonl`, enforces monotonic risk progression across `safe -> low_risk -> mid_risk -> high_risk`, and rejects broken or rule-misaligned ladders. The default Phase 4 model is `qwen3-vl-8b-thinking`. `phase4_isp_runner.py` is a separate bridge for running the external `image_safety_pipline` project in its own environment.
5. `phase5_filter.py`: caps per-category volume, flags/trims demographic over-representation from high-risk prompts, and writes both the final JSONL and a sibling `*_diversity_report.json`.

## Key conventions

- Severity labels are normalized centrally via `pipeline.utils.norm_level()` to exactly `safe`, `low_risk`, `mid_risk`, and `high_risk`. Reuse that helper instead of open-coding string cleanup.
- Category handling is case-insensitive in the pipeline logic, but original records may contain mixed-case values such as `Sexual_Content`. Preserve record data unless a phase explicitly normalizes for lookup.
- Phase 3 ladder records must keep the rung shape used by later phases: `rung_low_risk`, `rung_mid_risk`, and `rung_high_risk` are dicts containing `prompt`, `synthetic`, `source_id`, `image_path`, `generator_model`, `red_team_model`, and `seed`.
- Synthetic rung/image generation is deterministic. Phase 3 derives rung seeds from `ladder_id + level`, and Phase 4 derives synthetic image paths from category, rung level, ladder id, and model metadata. Avoid introducing randomness that would break resumability.
- Phase 4 treats missing images as a partial audit, not an automatic failure, unless score monotonicity is broken or an audited rung fails rule alignment. It first looks under `--image-root`, then `--project-root`, and only generates local images for non-safe synthetic rungs when generation is enabled.
- The only checked-in test is `tests/mock_pipeline_test.py`. It is an end-to-end smoke test that runs real sentence-transformer embedding in Phase 1 and mocks the external LLM/vision calls in Phases 2-4.
- Phase 5 quota balancing prefers ladders with fewer synthetic rungs before trimming, so changes to rung metadata can affect final sampling behavior.
