#!/bin/bash
#SBATCH --job-name=safegrad_v2_full_heads
#SBATCH --partition=002-partition-RAD
#SBATCH --nodelist=srdgx[00238,00241,00242,00243,00247]
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/lustre/users/vmduc/Projects/safegrad/logs/v2_full_heads_%j.out
#SBATCH --error=/lustre/users/vmduc/Projects/safegrad/logs/v2_full_heads_%j.err
#SBATCH --container-image=/lustre/users/vmduc/container_cache/image_generation_pipeline:latest.sqsh
#SBATCH --container-mounts=/lustre:/lustre,/store:/store,/home:/home

# Post-patch chain B: ordinal + category heads on the frozen full-SFT
# backbone, then test-split evaluation on the final v2 bench.
# Submit with: sbatch --dependency=afterok:<full_sft_job> run_v2_full_heads_eval.sh

set -euo pipefail

SAFEGRAD=/lustre/users/vmduc/Projects/safegrad
COMMON=/store/sr1/users/vmduc/safety_image_data_generation
CUR=data/v2_curation
RUNS=data/filter_runs

cd "$SAFEGRAD"
export HF_HOME=/lustre/users/vmduc/hf-cache
export TRANSFORMERS_CACHE=/lustre/users/vmduc/hf-cache
export HF_TOKEN="${HF_TOKEN:-}"
mkdir -p logs

echo "=================================================="
echo "  SafeGrad v2 — full heads + test eval"
echo "  Job ID  : ${SLURM_JOB_ID:-local}"
echo "  Node    : $(hostname)"
echo "  Started : $(date)"
echo "=================================================="

test -d "${RUNS}/sft_v2_full/adapter_best" || { echo "FATAL: no full SFT adapter"; exit 1; }

source .venv/bin/activate

echo ""
echo "--- Stage 2: ordinal + category heads ---"
uv run python -m safegrad.filter.train_heads \
  --dataset     "${CUR}/curated.jsonl" \
  --image-root  "${COMMON}" \
  --split-file  "${RUNS}/sft_v2_full/split.json" \
  --adapter     "${RUNS}/sft_v2_full/adapter_best" \
  --out         "${RUNS}/heads_v2_full"

echo ""
echo "--- Evaluation: final v2 bench test split ---"
uv run python -m safegrad.filter.eval_filter \
  --dataset     "${CUR}/curated.jsonl" \
  --image-root  "${COMMON}" \
  --split-file  "${RUNS}/sft_v2_full/split.json" \
  --split       test \
  --adapter     "${RUNS}/sft_v2_full/adapter_best" \
  --heads       "${RUNS}/heads_v2_full/heads.pt" \
  --out         "${RUNS}/eval_v2_full_test.json"

echo ""
echo "=================================================="
echo "  Done: $(date)"
echo "  Metrics: ${RUNS}/eval_v2_full_test.json"
echo "=================================================="
