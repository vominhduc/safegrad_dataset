"""Export clean public metadata from the internal pipeline output.

Produces a distribution-ready ``metadata.jsonl`` from the internal
``metadata_subset.jsonl``, stripping pipeline-internal fields that are not
suitable for public distribution:

Stripped top-level fields:
  mllm_score_*     — VLM conservatism issue (94% are 0.0); internal curation metric
  clip_align_*     — CLIP quality-control metric used during curation
  mllm_reasoning   — verbose internal VLM reasoning text
  mllm_rule_level  — internal consistency check labels
  mllm_rule_alignment
  visual_ladder_valid
  broken_rung_pairs
  rule_mismatch_levels
  partial_audit
  cluster_safe_anchor
  judge_*          — Phase 2 judge metadata (internal)

Stripped per-rung fields:
  verdict          — internal pipeline label; not distributed
  synthetic        — confusing name (means "prompt was LLM-generated", not "image is synthetic")
  source_id        — internal record identifier

Kept top-level fields:
  ladder_id, category, generator_model, red_team_model, seed

Kept per-rung fields:
  prompt, image_path
  explanation      — VLM justification (LLaVA-NeXT) for the image content; null where absent.
                     Run scripts/generate_explanations.py to populate all missing values before
                     exporting the final dataset.

When --image-dir is provided, real PNG files are copied (resolving symlinks) so
the export folder is self-contained and can be zipped for distribution.

Usage
-----
    uv run python safegrad/scripts/export_metadata.py \\
        --input      release/v1.0/metadata_subset.jsonl \\
        --image-root release/v1.0/ \\
        --output     release/v1.0/export/metadata.jsonl \\
        --image-dir  release/v1.0/export/

    # Images are copied to --image-dir / image_path, so if image_path is
    # "images/Disturbing_Content/...", the destination is:
    #   release/v1.0/export/images/Disturbing_Content/...
    # which matches how metadata.jsonl references them.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    stream=sys.stderr,
)
log = logging.getLogger(__name__)

_KEEP_TOP = {"ladder_id", "category", "generator_model", "red_team_model", "seed"}
_LEVELS = ("safe", "low_risk", "mid_risk", "high_risk")


def _strip_record(rec: dict) -> dict:
    out = {k: v for k, v in rec.items() if k in _KEEP_TOP}
    for lvl in _LEVELS:
        rung = rec.get(f"rung_{lvl}", {})
        if isinstance(rung, dict):
            rung_out: dict = {}
            for k in ("prompt", "image_path"):
                if k in rung:
                    rung_out[k] = rung[k]
            # explanation: null when absent; populated after generate_explanations.py runs
            rung_out["explanation"] = rung.get("explanation")
            out[f"rung_{lvl}"] = rung_out
        else:
            out[f"rung_{lvl}"] = {"explanation": None}
    return out


def _copy_images(
    records: list[dict],
    image_root: Path,
    image_dir: Path,
) -> tuple[int, int]:
    """Copy real PNG files from image_root into image_dir, resolving symlinks.

    Returns (n_copied, n_missing).
    """
    n_copied = n_missing = 0
    for rec in records:
        for lvl in _LEVELS:
            rung = rec.get(f"rung_{lvl}", {})
            img_path = rung.get("image_path")
            if not img_path:
                continue

            src: Path | None = None
            for root in (image_root, Path(".")):
                candidate = root / img_path
                if candidate.exists():
                    src = candidate.resolve()
                    break

            if src is None or not src.exists():
                n_missing += 1
                continue

            dst = image_dir / img_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(src, dst)
            n_copied += 1

    return n_copied, n_missing


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input",      required=True,
                   help="Path to internal metadata_subset.jsonl")
    p.add_argument("--output",     required=True,
                   help="Output path for public metadata.jsonl")
    p.add_argument("--image-root", default=".",
                   help="Root directory for resolving image_path fields")
    p.add_argument("--image-dir",  default=None,
                   help="If set, copy real PNG files here (resolving symlinks)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)
    image_root  = Path(args.image_root)
    image_dir   = Path(args.image_dir) if args.image_dir else None

    log.info("Loading %s …", input_path)
    records: list[dict] = []
    with input_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    log.info("Loaded %d records.", len(records))

    # Detect which top-level fields are being stripped
    if records:
        all_keys = set(records[0].keys())
        stripped_top = sorted(all_keys - _KEEP_TOP - {f"rung_{l}" for l in _LEVELS})
        log.info("Stripping top-level fields: %s", stripped_top)
        _kept_rung = {"prompt", "image_path", "explanation"}
        sample_rung = {k for lvl in _LEVELS for k in records[0].get(f"rung_{lvl}", {}).keys()}
        stripped_rung = sorted(sample_rung - _kept_rung)
        log.info("Stripping per-rung fields: %s", stripped_rung)

    # Strip records
    stripped = [_strip_record(r) for r in records]

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for rec in stripped:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log.info("Wrote %d records to %s", len(stripped), output_path)

    # Copy images
    if image_dir is not None:
        log.info("Copying images to %s …", image_dir)
        n_copied, n_missing = _copy_images(stripped, image_root, image_dir)
        log.info("  Copied: %d  Missing: %d", n_copied, n_missing)
        if n_missing:
            log.warning("  %d image paths could not be resolved — check --image-root", n_missing)

    # Summary
    log.info("=" * 60)
    log.info("Export complete")
    log.info("  Records: %d", len(stripped))
    log.info("  Output:  %s", output_path)
    if image_dir:
        log.info("  Images:  %s", image_dir)


if __name__ == "__main__":
    main()
