#!/bin/bash
#SBATCH --job-name=safegrad_filter_sft_v2
#SBATCH --partition=002-partition-RAD
#SBATCH --nodelist=srdgx[00238,00241,00242,00243,00247]
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=/lustre/users/vmduc/Projects/safegrad/logs/filter_sft_v2_%j.out
#SBATCH --error=/lustre/users/vmduc/Projects/safegrad/logs/filter_sft_v2_%j.err
#SBATCH --container-image=/lustre/users/vmduc/container_cache/image_generation_pipeline:latest.sqsh
#SBATCH --container-mounts=/lustre:/lustre,/store:/store,/home:/home

# SafeGrad v2 filter — Stage 1: safety instruction tuning (LoRA, Qwen2.5-VL-7B)
# Data: two-tier curation of safegrad_run_v2 (data/v2_curation/).
# Train = relaxed pool only; bench (strict) ladders are never trained on.
#
# Usage: sbatch run_filter_sft_v2.sh

set -euo pipefail
cd /lustre/users/vmduc/Projects/safegrad

export HF_HOME=/lustre/users/vmduc/hf-cache
export TRANSFORMERS_CACHE=/lustre/users/vmduc/hf-cache
export HF_TOKEN="${HF_TOKEN:-}"

WORKDIR=/store/sr1/users/vmduc/safety_image_data_generation/safegrad_run_v2
mkdir -p logs data/filter_runs

echo "=================================================="
echo "  SafeGrad v2 filter — Stage 1 SFT (5-level ordinal)"
echo "  Job ID  : ${SLURM_JOB_ID:-local}"
echo "  Node    : $(hostname)"
echo "  Started : $(date)"
echo "=================================================="

source .venv/bin/activate

uv run python -m safegrad.filter.train_sft \
  --dataset     data/v2_curation/curated.jsonl \
  --image-root  "${WORKDIR}" \
  --split-file  data/v2_curation/split.json \
  --out         data/filter_runs/sft_v2 \
  --epochs      3

echo ""
echo "=================================================="
echo "  Done: $(date)"
echo "  Output: data/filter_runs/sft_v2"
echo "=================================================="
