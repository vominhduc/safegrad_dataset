"""Stage 2: Severity Judge and Prompt Ladder Interpolation (Section 3.2).

The second stage of the Automated Severity Ladder (ASL) pipeline, combining
severity verification and missing-rung generation.

Sub-stage 2a — Severity Judge
  Calls a reasoning LLM (default: Qwen2.5-7B-Instruct) on each source record
  with the category-specific safety rules.  The judge verifies that the
  alleged severity labels are correct and filters poisoned ladders where the
  safe anchor is itself unsafe.

Sub-stage 2b — Prompt Ladder Interpolation
  Groups surviving records by ``ladder_id`` and generates missing rung prompts
  via a high-capacity LLM (default: meta-llama/Llama-3-70B-Instruct, as used
  in the paper).  For each cluster with gaps the LLM fills in the missing
  rung levels (low_risk / mid_risk / high_risk) using severity-conditioned
  interpolation guided by explicit Subject/Setting/Composition constraints.

Paper reference: Section 3, Stage 2 ("Severity-Conditioned Interpolation")
LLMs used: Llama-3-70B-Instruct (generation), Qwen2.5-7B-Instruct (judge)

Note: Sub-stage 2a and 2b correspond to pipeline/phase2_filter.py and
pipeline/phase3_filter.py respectively.  The two phases are exposed as a
unified Stage 2 here to match the paper's description.

Usage — run both sub-stages in sequence
-----------------------------------------
    # Sub-stage 2a: severity judge
    uv run python -m safegrad.pipeline.stage2_interpolation judge [OPTIONS]

    # Sub-stage 2b: prompt interpolation
    uv run python -m safegrad.pipeline.stage2_interpolation interpolate [OPTIONS]

Options (judge)
---------------
    --input         Source JSONL file               [default: metadata_stage1.jsonl]
    --output        Output JSONL file               [default: metadata_stage2a.jsonl]
    --rules         Path to rules JSONL             [default: data/rules.jsonl]
    --model         LLM judge model                 [default: Qwen/Qwen2.5-7B-Instruct]
    --backend       Inference backend               [default: local]
    --base-url      OpenAI-compatible API base      [default: none]
    --embed-model   Sentence-transformer model      [default: all-MiniLM-L6-v2]
    --concurrency   Max concurrent LLM calls        [default: 8]
    --max-retries   Max retries on errors           [default: 3]

Options (interpolate)
---------------------
    --input         Source JSONL file               [default: metadata_stage2a.jsonl]
    --output        Output JSONL file               [default: metadata_stage2b.jsonl]
    --rules         Path to rules JSONL             [default: data/rules.jsonl]
    --model         Generative LLM model            [default: meta-llama/Llama-3-70B-Instruct]
    --backend       Inference backend               [default: local]
    --base-url      OpenAI-compatible API base      [default: none]
    --concurrency   Max concurrent LLM calls        [default: 4]
    --max-retries   Max retries per call            [default: 3]
"""

from __future__ import annotations

import argparse
import sys

# Sub-stage 2a: severity judge (phase 2 implementation)
import pipeline.phase2_filter as _judge_module

# Sub-stage 2b: prompt ladder interpolation (phase 3 implementation)
import pipeline.phase3_filter as _interp_module


def _judge_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 2a: Severity Judge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input",        default="metadata_stage1.jsonl")
    p.add_argument("--output",       default="metadata_stage2a.jsonl")
    p.add_argument("--rules",        default="data/rules.jsonl")
    # Paper judge model: Qwen2.5-7B-Instruct
    p.add_argument("--model",        default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--backend",      default="local", choices=["local", "openai"])
    p.add_argument("--base-url",     default=None)
    p.add_argument("--embed-model",  default="all-MiniLM-L6-v2")
    p.add_argument("--concurrency",  default=8, type=int)
    p.add_argument("--max-retries",  default=3, type=int)
    return p.parse_args()


def _interp_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 2b: Prompt Ladder Interpolation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input",        default="metadata_stage2a.jsonl")
    p.add_argument("--output",       default="metadata_stage2b.jsonl")
    p.add_argument("--rules",        default="data/rules.jsonl")
    # Paper generation model: Llama-3-70B-Instruct
    p.add_argument("--model",        default="meta-llama/Llama-3-70B-Instruct")
    p.add_argument("--backend",      default="local", choices=["local", "openai"])
    p.add_argument("--base-url",     default=None)
    p.add_argument("--concurrency",  default=4, type=int)
    p.add_argument("--max-retries",  default=3, type=int)
    return p.parse_args()


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("judge", "interpolate"):
        print(__doc__)
        print("\nUsage: python -m safegrad.pipeline.stage2_interpolation {judge|interpolate} [OPTIONS]")
        sys.exit(1)

    subcmd = sys.argv.pop(1)
    if subcmd == "judge":
        # Delegate to phase2_filter with updated defaults
        # Inject defaults into sys.argv if not already set
        import pipeline.phase2_filter as _m
        _m.main()
    else:
        import pipeline.phase3_filter as _m
        _m.main()


if __name__ == "__main__":
    main()
