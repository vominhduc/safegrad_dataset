"""SafeGrad ASL pipeline runner.

Two operating modes:

  Path B — Generate from scratch (DEFAULT)
  ─────────────────────────────────────────
  Stage 0 generates safe seed prompts; Stage 2b synthesises all three unsafe
  rungs (low/mid/high) directly from each safe anchor.  No pre-existing paired
  data is required.

    uv run python safegrad/scripts/run_pipeline.py \\
        --workdir data/pipeline_run \\
        --rules   data/rules.jsonl

  Skip Stage 0 if you already have a seeds file:

    uv run python safegrad/scripts/run_pipeline.py \\
        --seeds   data/seeds.jsonl \\
        --workdir data/pipeline_run \\
        --rules   data/rules.jsonl

  Path A — Re-process existing paired data
  ─────────────────────────────────────────
  Start from a JSONL that already contains both ``prompt_safe`` and
  ``prompt_unsafe`` fields (e.g. the released ``data/metadata.jsonl``).
  Adds a severity-judge step (Stage 2a) between clustering and interpolation.

    uv run python safegrad/scripts/run_pipeline.py \\
        --paired-input data/metadata.jsonl \\
        --workdir      data/pipeline_run \\
        --rules        data/rules.jsonl

Intermediate files written to --workdir:

  Path B                           Path A
  ─────────────────────────────    ──────────────────────────────────
  seeds.jsonl         (stage 0)    <skipped>
  metadata_stage1.jsonl            metadata_stage1.jsonl
  <skipped>                        metadata_stage2a.jsonl  (judge)
  metadata_stage2b.jsonl           metadata_stage2b.jsonl
  metadata_stage3.jsonl            metadata_stage3.jsonl
  metadata_stage4.jsonl            metadata_stage4.jsonl
  run_summary.json                 run_summary.json

Paper models (Section 4):
  Stage 0 seeds:          Mistral-7B-Instruct (50%), Qwen2.5-7B-Instruct (50%)
  Stage 2a judge:         Qwen/Qwen2.5-7B-Instruct  (Path A only)
  Stage 2b interpolation: Qwen/Qwen2.5-72B-Instruct
  Stage 2c quality score: Qwen/Qwen2.5-7B-Instruct
  Stage 3 T2I:            SDXL (50%), FLUX.1-schnell (30%), Kolors (20%)
  Stage 4 VLM:            Qwen/Qwen3-VL-8B-Thinking (chain-of-thought)
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    stream=sys.stderr,
)
log = logging.getLogger(__name__)

DIVIDER = "=" * 64


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open() as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _run(cmd: list[str], label: str) -> None:
    log.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        log.error("%s failed (exit code %d). Aborting.", label, result.returncode)
        sys.exit(result.returncode)


def _uv(*args: str) -> list[str]:
    return ["uv", "run", "python", "-m", *args]


# ---------------------------------------------------------------------------
# Stage functions
# ---------------------------------------------------------------------------

def run_stage0(args: argparse.Namespace, workdir: Path) -> Path:
    """Stage 0: generate safe seed prompts."""
    output = workdir / "seeds.jsonl"
    cmd = _uv(
        "safegrad.pipeline.stage0_seed_generation",
        "--output",     str(output),
        "--samples",    str(args.s0_samples),
        "--batch-size", str(args.s0_batch_size),
        "--seed",       str(args.s0_seed),
        "--models",     *args.s0_models,
    )
    if args.s0_categories:
        cmd += ["--categories"] + args.s0_categories
    _run(cmd, "Stage 0")
    return output


def run_stage1(args: argparse.Namespace, workdir: Path, input_path: Path) -> Path:
    """Stage 1: FAISS deduplication and safe-anchor clustering."""
    output = workdir / "metadata_stage1.jsonl"
    _run(_uv(
        "safegrad.pipeline.stage1_clustering",
        "--input",           str(input_path),
        "--output",          str(output),
        "--model",           args.s1_model,
        "--threshold",       str(args.s1_threshold),
        "--batch-size",      str(args.s1_batch_size),
        "--max-per-cluster", str(args.s1_max_per_cluster),
    ), "Stage 1")
    return output


def run_stage2a(args: argparse.Namespace, workdir: Path, input_path: Path) -> Path:
    """Stage 2a: severity judge (Path A only)."""
    output = workdir / "metadata_stage2a.jsonl"
    cmd = _uv(
        "safegrad.pipeline.stage2_interpolation", "judge",
        "--input",       str(input_path),
        "--output",      str(output),
        "--rules",       str(args.rules),
        "--model",       args.s2a_model,
        "--backend",     args.s2a_backend,
        "--embed-model", args.s2a_embed_model,
        "--concurrency", str(args.s2a_concurrency),
        "--max-retries", str(args.s2a_max_retries),
        "--batch-size",  str(args.s2a_batch_size),
    )
    if args.s2a_base_url:
        cmd += ["--base-url", args.s2a_base_url]
    _run(cmd, "Stage 2a")
    return output


def run_stage2b(args: argparse.Namespace, workdir: Path, input_path: Path) -> Path:
    """Stage 2b: prompt interpolation — generates missing unsafe rungs."""
    output = workdir / "metadata_stage2b.jsonl"
    cmd = _uv(
        "safegrad.pipeline.stage2_interpolation", "interpolate",
        "--input",       str(input_path),
        "--output",      str(output),
        "--rules",       str(args.rules),
        "--model",       args.s2b_model,
        "--backend",     args.s2b_backend,
        "--concurrency", str(args.s2b_concurrency),
        "--max-retries", str(args.s2b_max_retries),
        "--batch-size",  str(args.s2b_batch_size),
    )
    if args.s2b_base_url:
        cmd += ["--base-url", args.s2b_base_url]
    _run(cmd, "Stage 2b")
    return output


def run_stage2c(args: argparse.Namespace, workdir: Path, input_path: Path) -> Path:
    """Stage 2c: text-level ladder quality scoring (paper Appendix E.3)."""
    output = workdir / "metadata_stage2c.jsonl"
    cmd = _uv(
        "safegrad.pipeline.stage2_interpolation", "verify",
        "--input",       str(input_path),
        "--output",      str(output),
        "--rules",       str(args.rules),
        "--model",       args.s2c_model,
        "--backend",     args.s2c_backend,
        "--concurrency", str(args.s2c_concurrency),
        "--max-retries", str(args.s2c_max_retries),
        "--batch-size",  str(args.s2c_batch_size),
        "--min-gap",     str(args.s2c_min_gap),
    )
    if args.s2c_base_url:
        cmd += ["--base-url", args.s2c_base_url]
    _run(cmd, "Stage 2c")
    return output


def run_stage3(args: argparse.Namespace, workdir: Path, input_path: Path) -> Path:
    """Stage 3: T2I reference image generation."""
    output    = workdir / "metadata_stage3.jsonl"
    image_dir = workdir / "images"
    cmd = _uv(
        "safegrad.pipeline.stage3_synthesis",
        "--input",      str(input_path),
        "--output",     str(output),
        "--image-dir",  str(image_dir),
        "--t2i-models", *args.s3_t2i_models,
    )
    if args.s3_no_generate:
        cmd += ["--no-generate"]
    _run(cmd, "Stage 3")
    return output


def run_stage4(args: argparse.Namespace, workdir: Path, input_path: Path) -> Path:
    """Stage 4: VLM visual verification."""
    output = workdir / "metadata_stage4.jsonl"
    # image_path fields are stored as "images/{category}/..." relative to workdir,
    # so --image-root must be workdir itself, not workdir/images.
    cmd = _uv(
        "safegrad.pipeline.stage4_verification",
        "--input",           str(input_path),
        "--output",          str(output),
        "--rules",           str(args.rules),
        "--image-root",      str(workdir),
        "--vlm-model",       args.s4_model,
        "--backend",         args.s4_backend,
        "--thinking-budget", str(args.s4_thinking_budget),
        "--max-side",        str(args.s4_max_side),
        "--batch-size",      str(args.s4_batch_size),
        "--concurrency",     str(args.s4_concurrency),
        "--max-retries",     str(args.s4_max_retries),
    )
    if args.s4_base_url:
        cmd += ["--base-url", args.s4_base_url]
    _run(cmd, "Stage 4")
    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Mode ─────────────────────────────────────────────────────────────────
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--seeds", metavar="FILE", default=None,
        help=(
            "Path B: skip Stage 0 and use FILE as the seeds input to Stage 1. "
            "FILE should contain records with at least 'category' and 'prompt_safe'. "
            "Omit to generate seeds via Stage 0."
        ),
    )
    mode.add_argument(
        "--paired-input", metavar="FILE", default=None,
        help=(
            "Path A: start from a JSONL with existing 'prompt_safe' + 'prompt_unsafe' "
            "pairs (e.g. data/metadata.jsonl). Automatically enables Stage 2a (judge) "
            "between Stage 1 and Stage 2b."
        ),
    )

    # ── Shared ───────────────────────────────────────────────────────────────
    p.add_argument("--workdir", default="data/pipeline_run",
                   help="Directory for intermediate and final files (default: data/pipeline_run)")
    p.add_argument("--rules",   default="data/rules.jsonl",
                   help="Safety rules JSONL file (default: data/rules.jsonl)")

    # ── Stage 0: seed generation (Path B, no --seeds) ────────────────────────
    g0 = p.add_argument_group("Stage 0 — safe seed generation (Path B only)")
    g0.add_argument(
        "--s0-models", nargs="+", default=["mistral:50", "qwen25:50"],
        metavar="MODEL[:WEIGHT]",
        help=(
            "Red-team LLMs for seed generation as KEY[:WEIGHT] entries. "
            "Examples: --s0-models mistral  |  --s0-models mistral:50 qwen25:50. "
            "Default: equal split between mistral and qwen25."
        ),
    )
    g0.add_argument("--s0-samples",    default=50, type=int,
                    help="Safe prompts per category (default: 50)")
    g0.add_argument("--s0-batch-size", default=5,  type=int,
                    help="Prompts per LLM call (default: 5)")
    g0.add_argument("--s0-seed",       default=42, type=int,
                    help="Random seed (default: 42)")
    g0.add_argument("--s0-categories", nargs="+", default=None, metavar="CAT",
                    help="Restrict to these categories (default: all 19)")

    # ── Stage 1: clustering ───────────────────────────────────────────────────
    g1 = p.add_argument_group("Stage 1 — deduplication and FAISS clustering")
    g1.add_argument("--s1-model",           default="all-MiniLM-L6-v2",
                    help="Sentence-transformer for embeddings (default: all-MiniLM-L6-v2)")
    g1.add_argument("--s1-threshold",       default=0.95, type=float,
                    help="Cosine similarity cutoff (paper: 0.95, default: 0.95)")
    g1.add_argument("--s1-batch-size",      default=512,  type=int,
                    help="Embedding batch size (default: 512)")
    g1.add_argument("--s1-max-per-cluster", default=1,    type=int,
                    help="Max diverse representatives per cluster (default: 1)")

    # ── Stage 2a: judge (Path A only) ────────────────────────────────────────
    g2a = p.add_argument_group("Stage 2a — severity judge (Path A / --paired-input only)")
    g2a.add_argument("--s2a-model",       default="Qwen/Qwen2.5-7B-Instruct",
                     help="Judge LLM (paper: Qwen2.5-7B-Instruct)")
    g2a.add_argument("--s2a-backend",     choices=("local", "openai"), default="local")
    g2a.add_argument("--s2a-base-url",    default="",
                     help="OpenAI-compatible API base URL for judge LLM")
    g2a.add_argument("--s2a-embed-model", default="all-MiniLM-L6-v2")
    g2a.add_argument("--s2a-concurrency", default=8,  type=int)
    g2a.add_argument("--s2a-max-retries", default=3,  type=int)
    g2a.add_argument("--s2a-batch-size",  default=8,  type=int)

    # ── Stage 2b: interpolation ───────────────────────────────────────────────
    g2b = p.add_argument_group("Stage 2b — prompt synthesis (all missing unsafe rungs)")
    g2b.add_argument("--s2b-model",       default="Qwen/Qwen2.5-72B-Instruct",
                     help="Generative LLM for prompt interpolation (default: Qwen2.5-72B-Instruct)")
    g2b.add_argument("--s2b-backend",     choices=("local", "openai"), default="local")
    g2b.add_argument("--s2b-base-url",    default="",
                     help="OpenAI-compatible API base URL for generative LLM")
    g2b.add_argument("--s2b-concurrency", default=4, type=int)
    g2b.add_argument("--s2b-max-retries", default=3, type=int)
    g2b.add_argument("--s2b-batch-size",  default=8, type=int)

    # ── Stage 2c: ladder quality scoring ─────────────────────────────────────
    g2c = p.add_argument_group("Stage 2c — ladder quality scoring (paper E.3)")
    g2c.add_argument("--s2c-model",       default="Qwen/Qwen2.5-7B-Instruct",
                     help="Judge LLM for ladder scoring (paper: Qwen2.5-7B-Instruct)")
    g2c.add_argument("--s2c-backend",     choices=("local", "openai"), default="local")
    g2c.add_argument("--s2c-base-url",    default="",
                     help="OpenAI-compatible API base URL for Stage 2c judge LLM")
    g2c.add_argument("--s2c-concurrency", default=8,   type=int)
    g2c.add_argument("--s2c-max-retries", default=3,   type=int)
    g2c.add_argument("--s2c-batch-size",  default=8,   type=int)
    g2c.add_argument("--s2c-min-gap",     default=0.4, type=float,
                     help="Min score gap between adjacent rungs (paper: 0.4)")

    # ── Stage 3: T2I synthesis ────────────────────────────────────────────────
    g3 = p.add_argument_group("Stage 3 — T2I reference image generation")
    g3.add_argument(
        "--s3-t2i-models", nargs="+",
        default=["sdxl:50", "flux1:30", "kolors:20"],
        metavar="MODEL[:WEIGHT]",
        help=(
            "T2I models to assign to ladders as KEY[:WEIGHT] entries. "
            "Keys: sdxl, flux1, kolors (or full HF IDs). "
            "Examples: --s3-t2i-models sdxl  |  --s3-t2i-models sdxl:60 flux1:40. "
            "Use 'none' to disable auto-assignment. "
            "Default: sdxl:50 flux1:30 kolors:20 (commercial Apache-2.0 models only)."
        ),
    )
    g3.add_argument("--s3-no-generate", action="store_true",
                    help="Dry run: skip image generation")

    # ── Stage 4: VLM verification ─────────────────────────────────────────────
    g4 = p.add_argument_group("Stage 4 — VLM visual verification (paper E.4)")
    g4.add_argument("--s4-model",            default="Qwen/Qwen3-VL-8B-Thinking",
                    help="Vision LLM (paper: Qwen3-VL-8B-Thinking, chain-of-thought enabled)")
    g4.add_argument("--s4-backend",          choices=("local", "openai"), default="local")
    g4.add_argument("--s4-base-url",         default=None)
    g4.add_argument("--s4-thinking-budget",  default=0,    type=int,
                    help="Thinking token budget (0 = disabled)")
    g4.add_argument("--s4-max-side",         default=512,  type=int)
    g4.add_argument("--s4-batch-size",       default=4,    type=int)
    g4.add_argument("--s4-concurrency",      default=8,    type=int)
    g4.add_argument("--s4-max-retries",      default=3,    type=int)

    # ── Resume control ────────────────────────────────────────────────────────
    p.add_argument("--start-from", default=None, metavar="STAGE",
                   choices=("0", "1", "2a", "2b", "2c", "3", "4"),
                   help="Resume from this stage (0, 1, 2a, 2b, 2c, 3, 4)")
    p.add_argument("--stop-after", default=None, metavar="STAGE",
                   choices=("0", "1", "2a", "2b", "2c", "3", "4"),
                   help="Stop after this stage (inclusive). E.g. '3' skips Stage 4 "
                        "VLM verification so images can be judged separately.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    # Determine mode and build the ordered list of (stage_id, label, fn, input_path)
    path_a = args.paired_input is not None

    if path_a:
        mode_label = "Path A (paired input + judge)"
        initial_input = Path(args.paired_input)
    elif args.seeds is not None:
        mode_label = "Path B (pre-existing seeds)"
        initial_input = Path(args.seeds)
    else:
        mode_label = "Path B (generate seeds via Stage 0)"
        initial_input = None  # will be set after stage0

    # Validate inputs exist
    if initial_input is not None and not initial_input.exists():
        log.error("Input file not found: %s", initial_input)
        sys.exit(1)

    # Build ordered pipeline as list of (stage_id, fn_or_None)
    # stage_id is used for --start-from comparison and output file naming
    pipeline: list[tuple[str, str]] = []

    if not path_a and args.seeds is None:
        pipeline.append(("0",  "Stage 0: Safe Seed Generation"))
    pipeline.append(("1",  "Stage 1: Deduplication & FAISS Clustering"))
    if path_a:
        pipeline.append(("2a", "Stage 2a: Severity Judge"))
    pipeline.append(("2b", "Stage 2b: Prompt Synthesis"))
    pipeline.append(("2c", "Stage 2c: Ladder Quality Scoring"))
    pipeline.append(("3",  "Stage 3: T2I Image Synthesis"))
    pipeline.append(("4",  "Stage 4: VLM Verification"))

    # Truncate the pipeline if --stop-after is given (inclusive).
    if args.stop_after is not None:
        stop_ids = [sid for sid, _ in pipeline]
        if args.stop_after not in stop_ids:
            log.error("--stop-after %s is not part of this run's stages: %s",
                      args.stop_after, " → ".join(stop_ids))
            sys.exit(1)
        cut = stop_ids.index(args.stop_after) + 1
        pipeline = pipeline[:cut]

    _STAGE_FN_MAP = {
        "0":  lambda a, w, i: run_stage0(a, w),
        "1":  run_stage1,
        "2a": run_stage2a,
        "2b": run_stage2b,
        "2c": run_stage2c,
        "3":  run_stage3,
        "4":  run_stage4,
    }
    _STAGE_OUTPUT = {
        "0":  "seeds.jsonl",
        "1":  "metadata_stage1.jsonl",
        "2a": "metadata_stage2a.jsonl",
        "2b": "metadata_stage2b.jsonl",
        "2c": "metadata_stage2c.jsonl",
        "3":  "metadata_stage3.jsonl",
        "4":  "metadata_stage4.jsonl",
    }

    # Determine start point
    start_from = args.start_from
    stage_ids  = [sid for sid, _ in pipeline]

    print(DIVIDER)
    print("  SAFEGRAD PIPELINE RUN")
    print(DIVIDER)
    print(f"  Mode    : {mode_label}")
    if initial_input:
        print(f"  Input   : {initial_input}  ({_count_lines(initial_input)} records)")
    print(f"  Workdir : {workdir}")
    print(f"  Stages  : {' → '.join(sid for sid, _ in pipeline)}")
    if start_from:
        print(f"  Resuming from stage {start_from}")
    print()

    summary: dict[str, dict] = {}
    current_input: Path | None = initial_input

    for stage_id, label in pipeline:
        # Determine the input for this stage
        if stage_id == "0":
            # Stage 0 takes no file input — it generates its own
            stage_input = None
        elif current_input is not None:
            stage_input = current_input
        else:
            # Reconstruct input from previous stage's expected output
            prev_idx = stage_ids.index(stage_id) - 1
            stage_input = workdir / _STAGE_OUTPUT[stage_ids[prev_idx]]

        # Skip if resuming from a later stage
        if start_from and stage_ids.index(stage_id) < stage_ids.index(start_from):
            log.info("Skipping %s (resume from %s)", label, start_from)
            current_input = workdir / _STAGE_OUTPUT[stage_id]
            continue

        print(f"\n{DIVIDER}")
        print(f"  {label}")
        print(DIVIDER)

        before = _count_lines(stage_input) if stage_input else 0
        t0 = time.perf_counter()

        output_path = _STAGE_FN_MAP[stage_id](args, workdir, stage_input)

        elapsed = time.perf_counter() - t0
        after   = _count_lines(output_path)
        removed = before - after
        pct     = 100.0 * removed / before if before else 0.0

        summary[stage_id] = {
            "label":   label,
            "input":   before,
            "output":  after,
            "removed": removed,
            "elapsed": round(elapsed, 1),
        }

        log.info(
            "%s done in %.1fs — %d → %d  (removed %d, %.1f%%)",
            label, elapsed, before, after, removed, pct,
        )

        if stage_id != "0" and before > 0 and after == 0:
            log.error("%s produced 0 records from %d inputs. Aborting.", label, before)
            sys.exit(1)

        current_input = output_path

    # Write run summary
    summary_path = workdir / "run_summary.json"
    with summary_path.open("w") as f:
        json.dump({"mode": mode_label, "stages": summary}, f, indent=2)
    log.info("Run summary: %s", summary_path)

    # Print final table
    print(f"\n{DIVIDER}")
    print("  FINAL SUMMARY")
    print(DIVIDER)
    print(f"  {'Stage':<40}  {'In':>7}  {'Out':>7}  {'Time':>8}")
    print(f"  {'-'*40}  {'-'*7}  {'-'*7}  {'-'*8}")
    for sid, info in summary.items():
        print(
            f"  {info['label']:<40}  {info['input']:>7}  {info['output']:>7}"
            f"  {info['elapsed']:>7.1f}s"
        )

    if summary:
        last_id   = list(summary.keys())[-1]
        last      = summary[last_id]
        print(f"\n  Final output  : {workdir / _STAGE_OUTPUT[last_id]}")
        print(f"  Final ladders : {last['output']}")

    print(f"\n{DIVIDER}\n")


if __name__ == "__main__":
    main()
