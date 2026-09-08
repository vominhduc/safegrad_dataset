"""Curate the v2 two-tier dataset from the externally judged pipeline output.

Reads ``judged_metadata.jsonl`` (every rung carries a judge verdict with
``s1_severity`` in 0..4) and assigns each ladder to a pool:

  bench   — strict core: judge severity within +/-1 of the rung rank on all
            five rungs AND severities non-decreasing (v1-analog acceptance)
  train   — relaxed pool: severities non-decreasing AND anchored endpoints
            (safe <= 1, very_high_risk >= 2), minus the strict core
  (rest)  — rejected (not written out)

Split rule (two-tier, replaces the v1 single-tier 80/10/10):
  * validation/test ladders come ONLY from the strict bench pool, split
    per category (25/75 by default) with the fixed seed
  * the training pool contains NO bench ladder

Outputs
-------
<out-dir>/curated.jsonl   — bench+train records with a ``pool`` field;
                            judge verdicts (incl. severities) are preserved
                            for later judge-simulation training
<out-dir>/split.json      — ladder-id split in the filter's format
<out-dir>/summary.json    — per-category pool and split counts

Usage
-----
    uv run python -m safegrad.scripts.curate_v2 \
      --input /store/.../safegrad_run_v2/judged_metadata.jsonl \
      --image-root /store/.../safegrad_run_v2 \
      --out-dir data/v2_curation
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from safegrad.pipeline.utils import LEVELS_ORDERED

DEFAULT_OMIT = ("information_from_sb",)  # degenerate taxonomy: only safe + very_high


def rung_severity(rec: dict, idx: int):
    vd = (rec.get(f"rung_{LEVELS_ORDERED[idx]}") or {}).get("verdict")
    return vd.get("s1_severity") if isinstance(vd, dict) else None


def pool_of(rec: dict) -> str | None:
    ss = [rung_severity(rec, i) for i in range(len(LEVELS_ORDERED))]
    if any(s is None for s in ss):
        return None
    mono = all(ss[i] <= ss[i + 1] for i in range(len(ss) - 1))
    tol1 = all(abs(ss[i] - i) <= 1 for i in range(len(ss)))
    anchored = ss[0] <= 1 and ss[-1] >= 2
    if mono and tol1:
        return "bench"                    # strict core
    if mono and anchored:
        return "train"                    # relaxed pool minus bench
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--omit-categories", nargs="*", default=list(DEFAULT_OMIT))
    ap.add_argument("--val-frac", type=float, default=0.25,
                    help="fraction of each bench category assigned to validation")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.image_root)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    omit = set(args.omit_categories)

    records = [json.loads(l) for l in open(args.input) if l.strip()]
    kept, rejected, pool_count, cat_count = [], 0, Counter(), Counter()
    n_missing_img = 0
    for rec in records:
        if rec.get("category") in omit:
            rejected += 1
            continue
        pool = pool_of(rec)
        if pool is None:
            rejected += 1
            continue
        ok = True
        for lv in LEVELS_ORDERED:
            p = (rec.get(f"rung_{lv}") or {}).get("image_path")
            if not p or not (root / p).exists():
                ok = False
                break
        if not ok:
            n_missing_img += 1
            continue
        rec["pool"] = pool
        kept.append(rec)
        pool_count[pool] += 1
        cat_count[(pool, rec["category"])] += 1

    rng = random.Random(args.seed)
    bench_ids = [r["ladder_id"] for r in kept if r["pool"] == "bench"]
    by_cat: dict[str, list[str]] = {}
    for r in kept:
        if r["pool"] == "bench":
            by_cat.setdefault(r["category"], []).append(r["ladder_id"])
    val_ids, test_ids = [], []
    for ids in by_cat.values():
        ids = sorted(ids)
        rng.shuffle(ids)
        k = max(1, round(len(ids) * args.val_frac))
        val_ids.extend(ids[:k])
        test_ids.extend(ids[k:])
    train_ids = [r["ladder_id"] for r in kept if r["pool"] == "train"]

    split = {
        "seed": args.seed,
        "rule": ("two-tier v2: train = relaxed pool (monotone+anchored, minus bench); "
                 f"val/test = strict bench pool (±1 tol + monotone), per-category "
                 f"{args.val_frac:.0%}/{1 - args.val_frac:.0%}; "
                 f"omitted categories: {sorted(omit)}"),
        "n_ladders": len(kept),
        "train": sorted(train_ids),
        "val": sorted(val_ids),
        "test": sorted(test_ids),
    }

    with (out / "curated.jsonl").open("w") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with (out / "split.json").open("w") as f:
        json.dump(split, f, indent=2)

    summary = {
        "input_records": len(records),
        "rejected_or_omitted": rejected,
        "missing_images": n_missing_img,
        "pools": dict(pool_count),
        "split_sizes": {"train": len(train_ids), "val": len(val_ids),
                        "test": len(test_ids)},
        "per_category": {
            cat: {"bench": cat_count.get(("bench", cat), 0),
                  "train": cat_count.get(("train", cat), 0),
                  }
            for cat in sorted({c for _, c in cat_count})
        },
    }
    with (out / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"input {len(records)} -> kept {len(kept)} "
          f"(bench {pool_count['bench']}, train {pool_count['train']}); "
          f"rejected/omitted {rejected}, missing images {n_missing_img}")
    print(f"split: train {len(train_ids)} / val {len(val_ids)} / test {len(test_ids)}")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
