#!/bin/bash
#SBATCH --job-name=safegrad_run_v2
#SBATCH --partition=002-partition-default
#SBATCH --exclude=srdgx00095,srdgx00096,srdgx00113
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH --output=/lustre/users/vmduc/Projects/safegrad/logs/safegrad_run_v2_%j.out
#SBATCH --error=/lustre/users/vmduc/Projects/safegrad/logs/safegrad_run_v2_%j.err
#SBATCH --container-image=/lustre/users/vmduc/container_cache/image_generation_pipeline:latest.sqsh
#SBATCH --container-mounts=/lustre:/lustre,/store:/store,/home:/home

set -euo pipefail

WORKDIR=/store/sr1/users/vmduc/safety_image_data_generation/safegrad_run_v2
SAFEGRAD=/lustre/users/vmduc/Projects/safegrad

cd "$SAFEGRAD"

export HF_HOME=/lustre/users/vmduc/hf-cache
export TRANSFORMERS_CACHE=/lustre/users/vmduc/hf-cache
export HF_TOKEN="${HF_TOKEN:-}"

mkdir -p "$WORKDIR" logs

echo "=================================================="
echo "  SafeGrad 5-Level Pipeline — v2 (Stages 0-3, no VLM verify)"
echo "  Job ID  : ${SLURM_JOB_ID:-local}"
echo "  Node    : $(hostname)"
echo "  Started : $(date)"
echo "  Workdir : ${WORKDIR}"
echo "=================================================="

source .venv/bin/activate

# Stages 0-3 only (--stop-after 3). Stage 4 VLM verification is intentionally
# skipped; images are judged separately and faster with judge_pending.py.
# Safe seeds raised to 1000/category (19 categories => ~19,000 seeds).
uv run python safegrad/scripts/run_pipeline.py \
  --workdir         "$WORKDIR" \
  --rules           data/rules.jsonl \
  --stop-after      3 \
  --s0-models       mistral:50 qwen25:50 \
  --s0-samples      1000 \
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

workdir = pathlib.Path("/store/sr1/users/vmduc/safety_image_data_generation/safegrad_run_v2")
for fname in ["seeds.jsonl", "metadata_stage1.jsonl", "metadata_stage2b.jsonl",
              "metadata_stage2c.jsonl", "metadata_stage3.jsonl"]:
    p = workdir / fname
    if p.exists():
        n = sum(1 for l in p.open() if l.strip())
        print(f"  {fname}: {n} records")
PYEOF

# Leave the safegrad (py3.11) venv before invoking the judge repo's own venv.
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
