#!/bin/bash
#SBATCH --job-name=safegrad_v2_full_sft
#SBATCH --partition=002-partition-RAD
#SBATCH --nodelist=srdgx[00238,00241,00242,00243,00247]
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=/lustre/users/vmduc/Projects/safegrad/logs/v2_full_sft_%j.out
#SBATCH --error=/lustre/users/vmduc/Projects/safegrad/logs/v2_full_sft_%j.err
#SBATCH --container-image=/lustre/users/vmduc/container_cache/image_generation_pipeline:latest.sqsh
#SBATCH --container-mounts=/lustre:/lustre,/store:/store,/home:/home

# Post-patch chain A: merge main v2 + patch1 judged outputs, re-curate the
# two-tier split (omit information_from_sb; disclosed relaxed admission for
# intellectual_property), then retrain filter Stage-1 SFT on the final bench.
# Submit with: sbatch --dependency=afterok:<patch1_job> run_v2_merge_curate_sft.sh

set -euo pipefail

SAFEGRAD=/lustre/users/vmduc/Projects/safegrad
COMMON=/store/sr1/users/vmduc/safety_image_data_generation
MAIN_JUDGED="${COMMON}/safegrad_run_v2/judged_metadata.jsonl"
PATCH_JUDGED="${COMMON}/safegrad_run_v2_patch1/judged_metadata.jsonl"
CUR=data/v2_curation
RUNS=data/filter_runs

cd "$SAFEGRAD"
export HF_HOME=/lustre/users/vmduc/hf-cache
export TRANSFORMERS_CACHE=/lustre/users/vmduc/hf-cache
export HF_TOKEN="${HF_TOKEN:-}"
mkdir -p logs

echo "=================================================="
echo "  SafeGrad v2 — merge + curate + full SFT retrain"
echo "  Job ID  : ${SLURM_JOB_ID:-local}"
echo "  Node    : $(hostname)"
echo "  Started : $(date)"
echo "=================================================="

# --- Gate: patch judging must have completed --------------------------------
test -s "$PATCH_JUDGED" || { echo "FATAL: missing $PATCH_JUDGED"; exit 1; }

source .venv/bin/activate

echo ""
echo "--- Merge judged runs ---"
uv run python -m safegrad.scripts.merge_v2_runs \
  --run "${MAIN_JUDGED}" \
  --run "patch1_:${PATCH_JUDGED}" \
  --common-root "${COMMON}" \
  --output "${CUR}/judged_merged.jsonl"

echo ""
echo "--- Curate (omit information_from_sb; relaxed bench admission for IP) ---"
uv run python -m safegrad.scripts.curate_v2 \
  --input             "${CUR}/judged_merged.jsonl" \
  --image-root        "${COMMON}" \
  --out-dir           "${CUR}" \
  --relax-categories  intellectual_property

echo ""
echo "--- Stage-1 SFT (full retrain on final bench) ---"
uv run python -m safegrad.filter.train_sft \
  --dataset     "${CUR}/curated.jsonl" \
  --image-root  "${COMMON}" \
  --split-file  "${CUR}/split.json" \
  --out         "${RUNS}/sft_v2_full" \
  --epochs      3

echo ""
echo "=================================================="
echo "  Done: $(date)"
echo "=================================================="
