"""Sync ``data/rules.jsonl`` from the canonical risk-taxonomy file.

Source of truth for the v2 taxonomy: ``new_risk_category.jsonl``
(one record per harm category with per-level rule text; levels
``safe / low_risk / moderate_risk / high_risk / very_high_risk``).

Transform applied per record:
  * category name normalised to lower snake_case
    (``-`` and ``/`` become ``_``), e.g. ``information_from_SB`` ->
    ``information_from_sb``, ``AI/IT_systems_abuse`` -> ``ai_it_systems_abuse``
  * ``moderate_risk`` is emitted as ``mid_risk`` (pipeline-internal name)
  * levels whose rule text is ``(Does not exist)`` are skipped
  * each emitted row carries ``risk_id`` for provenance and an ``order``
    index within its (category, level) bucket

Usage:
  uv run python -m safegrad.scripts.sync_rules \
      --source /lustre/users/vmduc/Projects/shares/new_risk_category.jsonl \
      --output data/rules.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

#: source-file level name -> pipeline level name
LEVEL_MAP = {
    "safe": "safe",
    "low_risk": "low_risk",
    "moderate_risk": "mid_risk",
    "mid_risk": "mid_risk",
    "high_risk": "high_risk",
    "very_high_risk": "very_high_risk",
}

MISSING_MARKERS = {"(does not exist)", "does not exist", "n/a", ""}


def norm_category(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace("/", "_")


def sync(source: Path, output: Path) -> dict:
    records = [json.loads(l) for l in source.open() if l.strip()]
    rows: list[dict] = []
    skipped: list[dict] = []
    for rec in records:
        category = norm_category(rec["category"])
        for src_level, level in LEVEL_MAP.items():
            text = (rec.get(src_level) or "").strip()
            key = f"{category}/{level}"
            seen = {f"{r['category']}/{r['level']}" for r in rows}
            seen_skip = {f"{s['category']}/{s['level']}" for s in skipped}
            if key in seen or key in seen_skip:
                continue  # alias already handled (moderate_risk/mid_risk)
            if text.lower() in MISSING_MARKERS:
                skipped.append({"category": category, "level": level,
                                "raw": text[:40]})
                continue
            n_existing = sum(1 for r in rows
                             if r["category"] == category and r["level"] == level)
            rows.append({
                "rule": text,
                "category": category,
                "level": level,
                "order": n_existing + 1,
                "risk_id": rec.get("risk_id", ""),
            })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"categories": len(records), "rules": len(rows), "skipped": skipped}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="new_risk_category.jsonl path")
    ap.add_argument("--output", default="data/rules.jsonl")
    args = ap.parse_args()
    stats = sync(Path(args.source), Path(args.output))
    print(f"categories: {stats['categories']}  rules written: {stats['rules']}")
    if stats["skipped"]:
        print("skipped (marked missing at source):")
        for s in stats["skipped"]:
            print(f"  {s['category']}/{s['level']}  [{s['raw']}]")


if __name__ == "__main__":
    main()
