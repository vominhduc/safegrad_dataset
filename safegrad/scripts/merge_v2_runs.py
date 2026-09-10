"""Merge the main v2 run with targeted patch-run judged metadata into one file.

Problem: patch runs re-number ladder ids per category (``<category>_0001``
collides with the main run's ids) and keep their own image root.  This script
rewrites the patch records (id prefix + prefixed ``image_path``) and the main
records (prefixed ``image_path``), and writes one merged file where every
``image_path`` is relative to ``--common-root``.

Usage:
  uv run python -m safegrad.scripts.merge_v2_runs \
    --run /store/.../safegrad_run_v2/judged_metadata.jsonl \
    --run patch1_:/store/.../safegrad_run_v2_patch1/judged_metadata.jsonl \
    --common-root /store/sr1/users/vmduc/safety_image_data_generation \
    --output data/v2_curation/judged_merged.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from safegrad.pipeline.utils import LEVELS_ORDERED


def parse_run(spec: str) -> tuple[str, Path]:
    """Parse a run spec of the form ``[IDPREFIX:]ABS_PATH``."""
    head, sep, tail = spec.partition(":")
    if not sep or tail.startswith("//"):
        # no prefix
        if spec.startswith("/"):
            return "", Path(spec)
        raise SystemExit(f"run spec needs an absolute path or PREFIX:path, got: {spec}")
    if spec.startswith("/"):
        return "", Path(spec)
    return head + ("" if head.endswith("_") else "_"), Path(tail)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True,
                    help="[IDPREFIX:]/abs/path/to/judged_metadata.jsonl, repeatable. "
                         "IDs in the file get IDPREFIX prepended (patch runs must "
                         "carry a prefix to avoid collisions with the main run).")
    ap.add_argument("--common-root", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    common_root = Path(args.common_root)
    out_records: list[dict] = []
    for spec in args.run:
        id_prefix, path = parse_run(spec)
        recs = [json.loads(l) for l in path.open() if l.strip()]
        run_dir = path.parent
        rel = run_dir.relative_to(common_root) if run_dir.is_relative_to(common_root) else run_dir
        seen = set()
        for rec in recs:
            rec["ladder_id"] = f"{id_prefix}{rec['ladder_id']}"
            if rec["ladder_id"] in seen:
                raise SystemExit(f"duplicate ladder_id in {path}: {rec['ladder_id']}")
            seen.add(rec["ladder_id"])
            for lv in LEVELS_ORDERED:
                rung = rec.get(f"rung_{lv}") or {}
                ip = rung.get("image_path")
                if ip:
                    rung["image_path"] = str(rel / ip)
        out_records.extend(recs)
        print(f"  {path}: {len(recs)} records (id_prefix={id_prefix!r}, rel={rel})")

    all_ids = [r["ladder_id"] for r in out_records]
    if len(all_ids) != len(set(all_ids)):
        raise SystemExit("ladder_id collision across runs; give each run an id prefix")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for rec in out_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"merged {len(out_records)} records -> {out}")


if __name__ == "__main__":
    main()
