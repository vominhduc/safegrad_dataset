"""SafeGrad pipeline runner — four-stage Automated Severity Ladder construction.

Executes all four stages in sequence:

  Stage 1: Safe-Anchor Extraction and FAISS Clustering
  Stage 2: Severity Judge + Prompt Ladder Interpolation
  Stage 3: T2I Image Synthesis
  Stage 4: Visual Monotonicity Verification

Output files are written to --workdir:
  metadata_stage1.jsonl   — after FAISS deduplication
  metadata_stage2a.jsonl  — after severity judging
  metadata_stage2b.jsonl  — after prompt interpolation
  metadata_stage3.jsonl   — after image generation + VLM scoring
  metadata_stage4.jsonl   — final verified ladders

Paper models (Section 3):
  Stage 2 judge:         Qwen/Qwen2.5-7B-Instruct
  Stage 2 interpolation: meta-llama/Llama-3-70B-Instruct
  Stage 3 T2I:           SDXL / Z-Turbo / FLUX.1 / SD 3.5
  Stage 4 VLM:           Qwen/Qwen3-VL-8B-Thinking

Usage
-----
    uv run python safegrad/scripts/run_pipeline.py \\
        --input  data/metadata.jsonl \\
        --workdir data/pipeline_run/ \\
        --rules  data/rules.jsonl \\
        --p2-backend openai --p2-base-url http://localhost:8000 \\
        --p3-model meta-llama/Llama-3-70B-Instruct
"""

import sys

# Delegate to the full pipeline runner which handles all phase orchestration.
# The full run_pipeline.py at the repo root supports the same four stages
# and all model configuration flags.

# Adjust sys.path so the import resolves correctly from any working directory.
import os
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

if __name__ == "__main__":
    # Dynamically execute the top-level run_pipeline.py
    import runpy
    runpy.run_path(
        os.path.join(_repo_root, "run_pipeline.py"),
        run_name="__main__",
    )
