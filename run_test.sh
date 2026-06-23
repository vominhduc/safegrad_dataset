#!/bin/bash
#SBATCH --job-name=safegrad_verify
#SBATCH --partition=002-partition-default
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/users/vmduc/Projects/safegrad/logs/verify_%j.out
#SBATCH --error=/lustre/users/vmduc/Projects/safegrad/logs/verify_%j.err
#SBATCH --container-image=/lustre/users/vmduc/container_cache/image_generation_pipeline:latest.sqsh
#SBATCH --container-mounts=/lustre:/lustre,/store:/store,/home:/home

# End-to-end pipeline verification on 2 sample ladders (Path B with pre-made seeds).
#
# Stages run:
#   1  — FAISS dedup/clustering         (CPU)
#   2b — Prompt synthesis               (GPU: Llama-3-70B)
#   2c — Ladder quality scoring         (GPU: Qwen2.5-7B)
#   3  — T2I image synthesis            (GPU: SDXL / FLUX.1-dev)
#   4  — VLM verification               (GPU: Qwen3-VL-8B-Thinking)
#
# Usage:
#   sbatch run_test.sh

set -euo pipefail
cd /lustre/users/vmduc/Projects/safegrad

export HF_HOME=/lustre/users/vmduc/hf-cache
export TRANSFORMERS_CACHE=/lustre/users/vmduc/hf-cache
export HF_TOKEN="${HF_TOKEN:-}"

WORKDIR=data/verify_run
mkdir -p "${WORKDIR}" logs

echo "=================================================="
echo "  SafeGrad Pipeline Verification"
echo "  Job ID  : ${SLURM_JOB_ID:-local}"
echo "  Node    : $(hostname)"
echo "  Started : $(date)"
echo "  Workdir : ${WORKDIR}"
echo "=================================================="

uv run python safegrad/scripts/run_pipeline.py \
  --seeds       data/safegrad_test_seeds.jsonl \
  --workdir     "${WORKDIR}" \
  --rules       data/rules.jsonl \
  --s1-threshold 0.90 \
  --s2b-model   Qwen/Qwen2.5-7B-Instruct \
  --s2b-backend local \
  --s2c-model   Qwen/Qwen2.5-7B-Instruct \
  --s2c-backend local \
  --s3-t2i-models sdxl:56 flux1:44 \
  --s4-model    Qwen/Qwen3-VL-8B-Thinking \
  --s4-backend  local

echo ""
echo "--- Export ---"
uv run python safegrad/scripts/export_metadata.py \
  --input      "${WORKDIR}/metadata_stage4.jsonl" \
  --image-root "${WORKDIR}" \
  --output     "${WORKDIR}/export/metadata.jsonl" \
  --image-dir  "${WORKDIR}/export/"

echo ""
echo "--- Results ---"
uv run python - << 'PYEOF'
import json, pathlib, sys

workdir = pathlib.Path("data/verify_run")
export_path = workdir / "export" / "metadata.jsonl"

# Show intermediate counts
for fname in ["metadata_stage1.jsonl", "metadata_stage2b.jsonl",
              "metadata_stage2c.jsonl", "metadata_stage3.jsonl",
              "metadata_stage4.jsonl"]:
    p = workdir / fname
    if p.exists():
        n = sum(1 for l in p.open() if l.strip())
        print(f"  {fname}: {n} records")

print()
if not export_path.exists():
    print("ERROR: export not produced")
    sys.exit(1)

ladders = [json.loads(l) for l in export_path.open() if l.strip()]
print(f"  Exported: {len(ladders)} ladder(s)")
for ldr in ladders:
    print(f"\n  Ladder {ldr['ladder_id']} ({ldr['category']}) — model: {ldr['generator_model']}")
    for rung in ("rung_safe", "rung_low_risk", "rung_mid_risk", "rung_high_risk"):
        r = ldr.get(rung, {})
        prompt = r.get("prompt", "")[:70]
        img    = r.get("image_path", "MISSING")
        expl   = (r.get("explanation") or "")[:60]
        print(f"    {rung:<18} | {prompt}")
        print(f"    {'':18} | img={img}")
        print(f"    {'':18} | explanation={expl}")
PYEOF

echo ""
echo "=================================================="
echo "  Done: $(date)"
echo "=================================================="
