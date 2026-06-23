#!/bin/bash
#SBATCH --job-name=safegrad_pipeline
#SBATCH --partition=002-partition-default
#SBATCH --exclude=srdgx00095,srdgx00096,srdgx00113
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH --output=/lustre/users/vmduc/Projects/safegrad/logs/safegrad_pipeline_%j.out
#SBATCH --error=/lustre/users/vmduc/Projects/safegrad/logs/safegrad_pipeline_%j.err
#SBATCH --container-image=/lustre/users/vmduc/container_cache/image_generation_pipeline:latest.sqsh
#SBATCH --container-mounts=/lustre:/lustre,/store:/store,/home:/home

set -euo pipefail

WORKDIR=/store/sr1/users/vmduc/safety_image_data_generation/safegrad_run
SAFEGRAD=/lustre/users/vmduc/Projects/safegrad

cd "$SAFEGRAD"

export HF_HOME=/lustre/users/vmduc/hf-cache
export TRANSFORMERS_CACHE=/lustre/users/vmduc/hf-cache
export HF_TOKEN="${HF_TOKEN:-}"

mkdir -p "$WORKDIR" logs

echo "=================================================="
echo "  SafeGrad 5-Level Pipeline"
echo "  Job ID  : ${SLURM_JOB_ID:-local}"
echo "  Node    : $(hostname)"
echo "  Started : $(date)"
echo "  Workdir : ${WORKDIR}"
echo "=================================================="

source .venv/bin/activate

# Full pipeline: Stage 0 (seed gen) → Stage 1 (FAISS) → Stage 2b (interpolation)
# → Stage 2c (quality score) → Stage 3 (T2I synthesis) → Stage 4 (VLM verification)
uv run python safegrad/scripts/run_pipeline.py \
  --workdir         "$WORKDIR" \
  --rules           data/rules.jsonl \
  --s0-models       mistral:50 qwen25:50 \
  --s0-samples      500 \
  --s1-threshold        0.95 \
  --s1-max-per-cluster  10 \
  --s2b-model           Qwen/Qwen2.5-7B-Instruct \
  --s2b-backend         local \
  --s2b-concurrency     4 \
  --s2c-model           Qwen/Qwen2.5-7B-Instruct \
  --s2c-backend         local \
  --s3-t2i-models       sdxl:60 flux1:40 \
  --s4-model        Qwen/Qwen3-VL-8B-Thinking \
  --s4-backend      local

echo ""
echo "--- Export ---"
uv run python safegrad/scripts/export_metadata.py \
  --input      "${WORKDIR}/metadata_stage4.jsonl" \
  --image-root "${WORKDIR}" \
  --output     "${WORKDIR}/export/metadata.jsonl" \
  --image-dir  "${WORKDIR}/export/"

echo ""
echo "--- Summary ---"
uv run python - << 'PYEOF'
import json, pathlib

workdir = pathlib.Path("/store/sr1/users/vmduc/safety_image_data_generation/safegrad_run")
for fname in ["seeds.jsonl", "metadata_stage1.jsonl", "metadata_stage2b.jsonl",
              "metadata_stage2c.jsonl", "metadata_stage3.jsonl", "metadata_stage4.jsonl"]:
    p = workdir / fname
    if p.exists():
        n = sum(1 for l in p.open() if l.strip())
        print(f"  {fname}: {n} records")

export_path = workdir / "export" / "metadata.jsonl"
if export_path.exists():
    ladders = [json.loads(l) for l in export_path.open() if l.strip()]
    cats = {}
    for ldr in ladders:
        cats[ldr["category"]] = cats.get(ldr["category"], 0) + 1
    print(f"\n  Exported ladders: {len(ladders)}")
    for cat, n in sorted(cats.items()):
        print(f"    {cat}: {n}")
PYEOF

echo ""
echo "=================================================="
echo "  Done: $(date)"
echo "=================================================="
