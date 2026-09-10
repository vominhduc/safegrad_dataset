# CLAUDE.md

This file provides guidance to Claude Code when working with the safegrad repository.

## Project Purpose

Generates **Adversarial Severity Ladders (ASL)** — sets of 5 images escalating from safe to very-high-risk on the same scene/topic. Used to build contrastive safety benchmarks for T2I models.

## Pipeline Stages

```
Stage 0 (seed gen) → Stage 1 (FAISS dedup) → Stage 2b (LLM interpolation)
→ Stage 2c (monotonicity score) → Stage 3 (T2I synthesis) → Stage 4 (VLM verification)
```

## Severity Levels (5-level taxonomy)

`safe` → `low_risk` → `mid_risk` → `high_risk` → `very_high_risk`

- Canonical names are in `safegrad/pipeline/utils.py` (`LEVELS_ORDERED`)
- `moderate_risk` normalises to `mid_risk` via `norm_level()`

## Category Taxonomy (19 categories)

Defined in `data/rules.jsonl` (sourced from `/lustre/users/vmduc/Projects/shares/new_risk_category.jsonl`):
`minors`, `sexual_content`, `violence_and_gore`, `physical_and_mental_health`, `harassment_and_bullying`,
`privacy_and_personal_data`, `intellectual_property`, `specialized_advice_and_legal_issues`,
`harmful_to_others`, `sex_related_crimes`, `hate_speech_and_discrimination`, `organizational_information`,
`information_from_sb`, `indiscriminate_weapons`, `illegal_activities`, `political_topics`,
`controversial_topics`, `misinformation_and_disinformation`, `ai_it_systems_abuse`

## Model Lineup (commercial Apache-2.0 only)

| Role | Model |
|------|-------|
| Red-team (seed gen) | Mistral-7B-Instruct (50%), Qwen2.5-7B-Instruct (50%) |
| Interpolation (S2b) | Qwen/Qwen2.5-7B-Instruct (default); 72B via API for higher quality |
| T2I synthesis (S3) | SDXL (60%), FLUX.1-schnell (40%) |
| VLM verification (S4) | Qwen/Qwen3-VL-8B-Thinking |

## Key Commands

```bash
# Submit full pipeline (Path B — generate seeds from scratch)
sbatch run_safegrad_pipeline.sh

# Or via Taskfile tasks (individual stages)
task generate_seeds_gpu        # Stage 0 only
task pipeline_gpu              # Full pipeline

# Run test (2-ladder smoke test)
sbatch run_test.sh
```

## Output Location

All generated data (including images) goes to:
`/store/sr1/users/vmduc/safety_image_data_generation/safegrad_run/`

- `seeds.jsonl` — Stage 0 output
- `metadata_stage{1-4}.jsonl` — per-stage intermediate files
- `images/{category}/{ladder_id}_{level}.png` — generated images
- `export/metadata.jsonl` — final clean export

## Daily Log

### 2026-06-23 (session 3)
- ✅ Done: Fixed Stage 1 `exact_dedup` critical bug — key was `(cat, sev, prompt_unsafe)` but Stage 0 seeds have no `prompt_unsafe`, so all 500 seeds/category collapsed to 1 record before clustering. Fixed: fall back to `prompt_safe` when `prompt_unsafe` is empty. Now all seeds survive dedup.
- ✅ Done: Improved Stage 4 VLM system prompt — Qwen3-VL-8B-Thinking was outputting free-form reasoning without JSON (no `</think>` tags). New prompt enforces JSON-only output; alignment check relaxed to "YES if main elements match".
- ✅ Done: Scaled up: `--s0-samples 500` (9,500 seeds), `--s1-max-per-cluster 10` → expect 500–2000 ladders entering Stage 4, targeting ~5000 images / 1000 ladders.
- ✅ Done: Cancelled job **4026358** (scale too small), submitted job **4026371** (~28h runtime).
- ✅ Done: Committed all changes to branch `asl-5level-pipeline` and pushed to GitHub (`git@github.com:vominhduc/safegrad_dataset.git`). `main` remains untouched.
- 🔄 In Progress: Job **4026371** running — outputs to `safegrad_run/`
- 📌 Next: Check Stage 1 output count (should be 500–2000, not 19). If Stage 4 rejection still >50%, consider switching VLM to Qwen2.5-VL-7B (non-thinking) for reliable JSON output.

### 2026-06-23 (session 2)
- ✅ Done: Diagnosed job **4024354** — completed with 950 seeds → 19 ladders → 7 exported (3 bugs found, 1 scale issue)
- ✅ Done: **Bug 1 (Stage 3)**: Kolors UNet cache incomplete (`diffusion_pytorch_model.bin` missing) — all 6 Kolors ladders generated 0 images. Dropped Kolors, now `sdxl:60 flux1:40` in `run_safegrad_pipeline.sh`. Updated Model Lineup table.
- ✅ Done: **Bug 2 (Stage 4)**: `very_high_risk` missing from `rung_specs` in `stage4_verification.py:436` — was never VLM-audited. Fixed by adding it to the loop and to the `annotated` output dict.
- ✅ Done: **Bug 3 (Stage 4)**: Ladders with 0 images (all Kolors) passed Stage 4 vacuously. Fixed with `no_images_scored` early-rejection guard.
- ✅ Done: **Scale fix (Stage 1)**: All 50 seeds/category merged into 1 cluster at threshold=0.95 (transitive union-find). Added `--max-per-cluster N` with greedy max-min diversity selection to `stage1_clustering.py` + `run_pipeline.py`. Set `--s1-max-per-cluster 5` → expect ~95 ladders (19 cats × 5).
- Stage 2c rejection was only 5% (1/19) — no need to lower `--s2c-min-gap`. Stage 4 alignment failures: low_risk=4, high_risk=2, safe=2, mid_risk=1, prompt_incoherent=1, vision_api_failure=1.
- 📌 Next: Submit new SLURM job (`sbatch run_safegrad_pipeline.sh`). Monitor Stage 4 alignment rejection rate — if still >40%, consider softening the per-rung check from ANY→fail to MAJORITY→fail or raising `--s4-max-retries`.

### 2026-06-23 (session 1)
- ✅ Done: Extended pipeline from 4-level to 5-level (added `very_high_risk`) across all pipeline files
- ✅ Done: Replaced all non-commercial/Meta models with Apache-2.0 alternatives (Qwen2.5, FLUX.1-schnell, Kolors)
- ✅ Done: Rewrote `data/rules.jsonl` — 92 rules, 19 categories × 5 levels from `new_risk_category.jsonl`
- ✅ Done: Updated eval files (`hgr.py`, `exp2_hgr_sbs.py`, `export_metadata.py`) to 5-level support
- ✅ Done: Submitted SLURM job **4024354** (`safegrad_pipeline`) — expected ~2400–2800 images

### 2026-09-08 (session: v2 filter)
- ✅ Done: Packaged the v1 pipeline (4-level, paper-matching) as `v1.0.0` on `main` (tag pushed; this branch unchanged).
- ✅ Done: Added `safegrad/filter/` — the v2 severity-graded filter (SafeAtlas-VL-inspired): stage-1 safety SFT with structured five-level output (`Safety:`/`Categories:`), stage-2 soft cumulative ordinal head (monotone thresholds, Gaussian-smoothed targets, γ=0.75) + category head on the frozen backbone, continuous risk score in [0,100]. Eval keeps the v1 Table-6 binary protocol (val-tuned score threshold) and adds ordinal metrics (acc/macro-F1/within-1/MAE/QWK/Spearman). Conditioning: `--condition prompt` (default, deployment-realistic) or `none`; rung explanations are never shown (label leakage).
- 📌 Next: smoke-test + dry run; point `--dataset/--image-root` at a 5-level export once Stage 4 of `safegrad_run_v2` finishes (currently only Stages 1–3 + judge outputs exist there).
- ✅ Done: `safegrad/scripts/sync_rules.py` — `data/rules.jsonl` now regenerates deterministically from `shares/new_risk_category.jsonl` (19 cats; `moderate_risk`→`mid_risk`; `(Does not exist)` levels skipped; category names lower-snake). Verified byte-identical rule text vs. previous hand-written file; only change is the added per-row `risk_id` provenance. Confirmed v2 taxonomy = the Sep-08-updated source file.
- ✅ Done: Two-tier curation of safegrad_run_v2 judging output (curate_v2.py): 18,600 ladders → bench pool 1,898 (judge-severity ±1 of rung rank on all 5 rungs + monotone) + train pool 3,666 (monotone + anchored endpoints, bench excluded); information_from_sb omitted (degenerate taxonomy). Split: train 3,666 / val 474 / test 1,424 (val/test only from bench). Caveat: intellectual_property has 0 strict bench ladders (17/18 cats in bench).
- ✅ Done: Submitted filter Stage-1 SFT job 4543551 (Qwen2.5-VL-7B LoRA, 5-level structured target, prompt-conditioned, 3 epochs) on 002-partition-RAD / proj-rai-all nodes.
- 📌 Next: when 4543551 finishes → run train_heads job → eval_filter on test; then review bench coverage decision for intellectual_property.
- ✅ Done: Heads+eval debugging: root cause of the device-side assert was PromptDataset flattening Qwen2.5-VL pixel_values to 1-D (they carry no batch dim); fixed collation (verified patch count == prod(grid_thw)) and completed job 4548182. FIRST V2 FILTER RESULTS (test, 7,120 ex): FPR@L0 0.49%, per-level det 99.9/100/100/100%, FBR 0.035%, sensitivity 99.5%; ordinal: acc 40%, within-1 77%, MAE 0.85, QWK 0.76, Spearman 0.94. Score means by level: 8.9/33.5/56.9/76.4/89.0 (monotone).
- ✅ Done: Curation diagnostic — strict pass rates: intellectual_property 0/992 flat (judge assigns ~sev 1 regardless of rung), organizational_information 9/976, indiscriminate_weapons 21/990 (inverse failure: judge over-fires sev 3 even at safe rungs). Chosen: targeted regen patch job for the three weak cats (4558384, 1500 seeds/cat, stages 0-3 + external judging) + disclosed --relax-categories admission for IP bench.
- ✅ Done: merge_v2_runs.py (namespace patch ids + shared image root) and curate_v2.py --relax-categories prepared for post-patch re-curation.
- 📌 Next: patch1 (4558384) → merge judged outputs → re-run curate_v2 (omit information_from_sb, relax IP) → retrain SFT+heads on final v2 bench.
- ✅ Done: full post-patch chain submitted: patch1 regen (4558384) -> merge+curecurate+SFT retrain (4558755, gated afterok:4558384) -> heads+test eval (4558756, afterok:4558755). Curate in chain A omits information_from_sb and discloses relaxed-bench admission for intellectual_property only.
- 📌 Watch: patch1 IP yield (decides whether IP even enters relaxed pool); if IP strict stays ~0 the disclosed relaxed admission covers it (~50+ bench ladders expected from patch volume).
