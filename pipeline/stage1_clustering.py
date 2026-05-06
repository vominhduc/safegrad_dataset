"""Stage 1: Safe-Anchor Extraction and FAISS Clustering (Section 3.1).

The first stage of the Automated Severity Ladder (ASL) pipeline.

Steps
-----
1. Exact deduplication — remove records with duplicate ``prompt_unsafe``
   within the same (category, target_severity) bucket.  Most-descriptive
   record is kept when duplicates exist.

2. Safe-anchor clustering — embed ``prompt_safe`` per category using a
   sentence-transformer model and group records whose safe-anchor cosine
   similarity exceeds ``--threshold`` (default 0.95) via FAISS HNSW.
   Only records within the same category are clustered together.

3. Ladder-ID assignment — each cluster becomes one escalation ladder.
   Records are annotated with ``ladder_id`` ({CATEGORY}_{index:04d}) and
   ``cluster_safe_anchor`` (canonical safe prompt for the ladder).

Paper reference: Section 3, Stage 1 ("Safe-Anchor Extraction")
FAISS threshold: cosine similarity ≥ 0.95 (Section 3.1)

Usage
-----
    uv run python -m safegrad.pipeline.stage1_clustering [OPTIONS]

Options
-------
    --input      Source JSONL file                   [default: data/metadata.jsonl]
    --output     Output JSONL file                   [default: metadata_stage1.jsonl]
    --model      Sentence-transformer model          [default: all-MiniLM-L6-v2]
    --threshold  Safe-anchor cosine similarity       [default: 0.95]
    --batch-size Embedding batch size                [default: 512]
"""

from pipeline.phase1_filter import (  # noqa: F401
    exact_dedup,
    assign_ladder_ids,
    _build_clusters,
    _union_find_init,
    _find,
    _union,
)

import argparse
import json
import logging
import sys
from pathlib import Path

from pipeline.utils import descriptiveness_score, norm_level  # noqa: F401

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    stream=sys.stderr,
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input",      default="data/metadata.jsonl",       help="Source JSONL file")
    p.add_argument("--output",     default="metadata_stage1.jsonl",     help="Output JSONL file")
    p.add_argument("--model",      default="all-MiniLM-L6-v2",          help="Sentence-transformer model")
    p.add_argument("--threshold",  default=0.95, type=float,            help="Cosine similarity threshold (paper: 0.95)")
    p.add_argument("--batch-size", default=512,  type=int,              help="Embedding batch size")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_path  = Path(args.input)
    output_path = Path(args.output)

    log.info("Stage 1 — Safe-Anchor Extraction & FAISS Clustering")
    log.info("Loading %s …", input_path)
    records: list[dict] = []
    with input_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    log.info("Loaded %d records.", len(records))

    original_count = len(records)
    records, n_exact = exact_dedup(records)

    records = assign_ladder_ids(
        records,
        model_name=args.model,
        threshold=args.threshold,
        batch_size=args.batch_size,
    )

    n_ladders = len({r["ladder_id"] for r in records})
    log.info("Output: %d records, %d ladder clusters (removed %d duplicates)",
             len(records), n_ladders, n_exact)

    with output_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log.info("Wrote %s", output_path)


if __name__ == "__main__":
    main()
