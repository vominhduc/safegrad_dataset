#!/bin/bash
#SBATCH --job-name=safegrad_v2_patch1
#SBATCH --partition=002-partition-RAD
#SBATCH --nodelist=srdgx[00238,00241,00242,00243,00247]
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH --time=30:00:00
#SBATCH --output=/lustre/users/vmduc/Projects/safegrad/logs/v2_patch1_%j.out
#SBATCH --error=/lustre/users/vmduc/Projects/safegrad/logs/v2_patch1_%j.err
#SBATCH --container-image=/lustre/users/vmduc/container_cache/image_generation_pipeline:latest.sqsh
#SBATCH --container-mounts=/lustre:/lustre,/store:/store,/home:/home

# Targeted regeneration for the strict-bench-weak categories of the v2 run
# (bench counts from the first curation: intellectual_property 0,
# organizational_information 9, indiscriminate_weapons 21).
# Stages 0-3 only; judging via the shared judge repo, as in the main v2 run.
#
# Usage: sbatch run_v2_patch1.sh

set -euo pipefail

SAFEGRAD=/lustre/users/vmduc/Projects/safegrad
WORKDIR=/store/sr1/users/vmduc/safety_image_data_generation/safegrad_run_v2_patch1

cd "$SAFEGRAD"
export HF_HOME=/lustre/users/vmduc/hf-cache
export TRANSFORMERS_CACHE=/lustre/users/vmduc/hf-cache
export HF_TOKEN="${HF_TOKEN:-}"

mkdir -p "$WORKDIR" logs

echo "=================================================="
echo "  SafeGrad v2 patch1 — targeted regen (3 weak categories)"
echo "  Job ID  : ${SLURM_JOB_ID:-local}"
echo "  Node    : $(hostname)"
echo "  Started : $(date)"
echo "  Workdir : ${WORKDIR}"
echo "=================================================="

source .venv/bin/activate

uv run python safegrad/scripts/run_pipeline.py \
  --workdir         "$WORKDIR" \
  --rules           data/rules.jsonl \
  --stop-after      3 \
  --s0-categories   intellectual_property organizational_information indiscriminate_weapons \
  --s0-models       mistral:50 qwen25:50 \
  --s0-samples      1500 \
  --s1-threshold        0.95 \
  --s1-max-per-cluster  10 \
  --s2b-model           Qwen/Qwen2.5-7B-Instruct \
  --s2b-backend         local \
  --s2b-concurrency     4 \
  --s2c-model           Qwen/Qwen2.5-7B-Instruct \
  --s2c-backend         local \
  --s3-t2i-models       sdxl:60 flux1:40

echo ""
echo "--- Summary ---"
uv run python - << 'PYEOF'
import pathlib
workdir = pathlib.Path("/store/sr1/users/vmduc/safety_image_data_generation/safegrad_run_v2_patch1")
for fname in ["seeds.jsonl", "metadata_stage1.jsonl", "metadata_stage2b.jsonl",
              "metadata_stage2c.jsonl", "metadata_stage3.jsonl"]:
    p = workdir / fname
    if p.exists():
        n = sum(1 for l in p.open() if l.strip())
        print(f"  {fname}: {n} records")
PYEOF

deactivate || true

echo ""
echo "=================================================="
echo "  Judging — judge_pending.py (same container, separate venv)"
echo "  Started : $(date)"
echo "=================================================="

JUDGE_REPO=/lustre/users/vmduc/Projects/safety_image_data_generation
cd "$JUDGE_REPO"
PYTHONPATH="$JUDGE_REPO" "$JUDGE_REPO/.venv/bin/python" judge_pending.py \
  --metadata_jsonl "${WORKDIR}/metadata_stage3.jsonl" \
  --base_dir       "${WORKDIR}" \
  --use_vllm

echo ""
echo "=================================================="
echo "  Done (generate + judge): $(date)"
echo "  Output: ${WORKDIR}/judged_metadata.jsonl"
echo "=================================================="
