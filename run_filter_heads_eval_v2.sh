#!/bin/bash
#SBATCH --job-name=safegrad_filter_heads_v2
#SBATCH --partition=002-partition-RAD
#SBATCH --nodelist=srdgx[00238,00241,00242,00243,00247]
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/lustre/users/vmduc/Projects/safegrad/logs/filter_heads_v2_%j.out
#SBATCH --error=/lustre/users/vmduc/Projects/safegrad/logs/filter_heads_v2_%j.err
#SBATCH --container-image=/lustre/users/vmduc/container_cache/image_generation_pipeline:latest.sqsh
#SBATCH --container-mounts=/lustre:/lustre,/store:/store,/home:/home

# SafeGrad v2 filter — Stage 2 (ordinal + category heads on the frozen SFT
# backbone) + evaluation on the strict bench test split.
# Submitted with --dependency=afterok:<sft-job-id>; runs only if Stage-1 SFT
# completed successfully.
#
# Usage: sbatch --dependency=afterok:<sft_job> run_filter_heads_eval_v2.sh

set -euo pipefail
cd /lustre/users/vmduc/Projects/safegrad

export HF_HOME=/lustre/users/vmduc/hf-cache
export TRANSFORMERS_CACHE=/lustre/users/vmduc/hf-cache
export HF_TOKEN="${HF_TOKEN:-}"

WORKDIR=/store/sr1/users/vmduc/safety_image_data_generation/safegrad_run_v2
RUNS=data/filter_runs
mkdir -p logs "${RUNS}"

echo "=================================================="
echo "  SafeGrad v2 filter — Stage 2 heads + test eval"
echo "  Job ID  : ${SLURM_JOB_ID:-local}"
echo "  Node    : $(hostname)"
echo "  Started : $(date)"
echo "=================================================="

source .venv/bin/activate

echo ""
echo "--- Stage 2: ordinal + category heads (frozen backbone) ---"
uv run python -m safegrad.filter.train_heads \
  --dataset     data/v2_curation/curated.jsonl \
  --image-root  "${WORKDIR}" \
  --split-file  "${RUNS}/sft_v2/split.json" \
  --adapter     "${RUNS}/sft_v2/adapter_best" \
  --out         "${RUNS}/heads_v2"

echo ""
echo "--- Evaluation: strict bench test split ---"
uv run python -m safegrad.filter.eval_filter \
  --dataset     data/v2_curation/curated.jsonl \
  --image-root  "${WORKDIR}" \
  --split-file  "${RUNS}/sft_v2/split.json" \
  --split       test \
  --adapter     "${RUNS}/sft_v2/adapter_best" \
  --heads       "${RUNS}/heads_v2/heads.pt" \
  --out         "${RUNS}/eval_v2_test.json"

echo ""
echo "=================================================="
echo "  Done: $(date)"
echo "  Metrics: ${RUNS}/eval_v2_test.json"
echo "=================================================="
