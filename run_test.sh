#!/bin/bash
#SBATCH --job-name=safegrad_test
#SBATCH --partition=002-partition-default
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/safegrad_test_%j.out
#SBATCH --error=logs/safegrad_test_%j.err

# End-to-end smoke test: Stages 1-4 on the bundled 6-record paired sample
# (Path A). Adjust HF_HOME / container flags to your cluster environment.
#
# Usage:
#   export HF_TOKEN=hf_...
#   sbatch run_test.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

export HF_HOME="${HF_HOME:-~/.cache/huggingface}"
export TRANSFORMERS_CACHE="${HF_HOME}"
export HF_TOKEN="${HF_TOKEN:-}"

WORKDIR=data/safegrad_test_run
mkdir -p "${WORKDIR}" logs

echo "=================================================="
echo "  SafeGrad Pipeline Smoke Test (Path A)"
echo "  Job ID  : ${SLURM_JOB_ID:-local}"
echo "  Node    : $(hostname)"
echo "  Started : $(date)"
echo "  Workdir : ${WORKDIR}"
echo "=================================================="

uv run python safegrad/scripts/run_pipeline.py \
  --paired-input data/safegrad_test_input.jsonl \
  --workdir      "${WORKDIR}" \
  --rules        data/rules.jsonl

echo "=================================================="
echo "  Finished: $(date)"
echo "  Final output: ${WORKDIR}/metadata_stage4.jsonl"
echo "=================================================="
