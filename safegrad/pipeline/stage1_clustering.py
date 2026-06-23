"""Stage 1: Safe-Anchor Extraction and FAISS Clustering (Section 3, Stage 1).

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
FAISS threshold: cosine similarity >= 0.95 (Section 3.1)

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

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from safegrad.pipeline.utils import descriptiveness_score, md5_pair, norm_level  # noqa: F401

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    stream=sys.stderr,
)
log = logging.getLogger(__name__)


def _release_torch_memory() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Step 1 — Exact de-duplication (per category + severity bucket)
# ---------------------------------------------------------------------------

def exact_dedup(records: list[dict]) -> tuple[list[dict], int]:
    """Remove duplicate ``prompt_unsafe`` entries within each
    (category, target_severity) bucket.

    Deduplicating per-bucket (not globally) ensures that the same unsafe
    prompt text cannot appear at multiple severity levels of the same
    category.  When duplicates exist the most-descriptive record is kept.
    """
    seen: dict[tuple[str, str, str], dict] = {}
    for rec in records:
        cat = rec.get("category", "").lower()
        sev = norm_level(rec.get("target_severity", ""))
        unsafe = rec.get("prompt_unsafe", "").strip()
        key = (cat, sev, unsafe or rec.get("prompt_safe", "").strip())
        if key not in seen or descriptiveness_score(rec) > descriptiveness_score(seen[key]):
            seen[key] = rec

    kept = list(seen.values())
    removed = len(records) - len(kept)
    log.info("Exact dedup: %d -> %d  (removed %d)", len(records), len(kept), removed)
    return kept, removed


# ---------------------------------------------------------------------------
# Step 2 — Safe-anchor clustering (within category)
# ---------------------------------------------------------------------------

def _union_find_init(n: int) -> list[int]:
    return list(range(n))


def _find(parent: list[int], x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _union(parent: list[int], x: int, y: int) -> None:
    px, py = _find(parent, x), _find(parent, y)
    if px != py:
        parent[px] = py


def _build_clusters(
    embeddings: np.ndarray,
    threshold: float,
    batch_size: int = 1024,
) -> dict[int, list[int]]:
    """Return {cluster_root: [member_local_indices]} using FAISS HNSW.

    All vectors are L2-normalised so inner-product equals cosine similarity.
    """
    n = embeddings.shape[0]
    dim = embeddings.shape[1]

    vecs = embeddings.copy().astype("float32")
    faiss.normalize_L2(vecs)

    index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 64
    index.add(vecs)

    parent = _union_find_init(n)
    k = min(16, n)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        sims, nbrs = index.search(vecs[start:end], k)
        for local_i, (sim_row, nbr_row) in enumerate(zip(sims, nbrs)):
            global_i = start + local_i
            for sim, nbr in zip(sim_row[1:], nbr_row[1:]):
                if nbr < 0:
                    break
                if sim >= threshold:
                    _union(parent, global_i, nbr)

    clusters: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        clusters[_find(parent, i)].append(i)

    return clusters


def _greedy_diverse_select(
    local_indices: list[int],
    embeddings: np.ndarray,
    records: list[dict],
    k: int,
) -> list[int]:
    """Return up to k indices from local_indices that maximise diversity.

    Uses greedy max-min: start with the most descriptive record, then
    repeatedly pick the one with maximum minimum cosine distance to all
    already-selected records.  Embeddings must be L2-normalised so that
    inner product equals cosine similarity.
    """
    if len(local_indices) <= k:
        return local_indices

    # Seed with the most descriptive record
    seed = max(local_indices, key=lambda i: descriptiveness_score(records[i]))
    selected = [seed]
    remaining = [i for i in local_indices if i != seed]

    while len(selected) < k and remaining:
        # max over remaining: min cosine distance to any selected
        best = max(
            remaining,
            key=lambda i: min(
                1.0 - float(np.dot(embeddings[i], embeddings[s]))
                for s in selected
            ),
        )
        selected.append(best)
        remaining.remove(best)

    return selected


def assign_ladder_ids(
    records: list[dict],
    model_name: str,
    threshold: float,
    batch_size: int,
    max_per_cluster: int = 1,
) -> list[dict]:
    """Embed ``prompt_safe`` per category, cluster similar safe anchors, and
    return one representative record per ladder with ``ladder_id`` and
    ``cluster_safe_anchor`` fields set.

    With ``max_per_cluster > 1``, up to that many diverse representatives are
    selected per cluster using greedy max-min distance, each becoming its own
    ladder.  Categories are processed independently.
    """
    log.info("Loading embedding model: %s", model_name)
    model = SentenceTransformer(model_name)

    by_category: dict[str, list[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        by_category[rec.get("category", "unknown").lower()].append(i)

    selected_records: list[dict] = []
    ladder_counter = 0

    for category, cat_indices in sorted(by_category.items()):
        cat_records = [records[i] for i in cat_indices]
        safe_prompts = [r["prompt_safe"] for r in cat_records]

        log.info("  [%s] Encoding %d safe anchors ...", category, len(safe_prompts))
        embeddings = model.encode(
            safe_prompts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        clusters = _build_clusters(embeddings, threshold=threshold, batch_size=batch_size)
        log.info("  [%s] %d records -> %d clusters", category, len(cat_indices), len(clusters))

        for cluster_local_indices in clusters.values():
            representatives = _greedy_diverse_select(
                cluster_local_indices, embeddings, cat_records, max_per_cluster
            )
            for rep_local_i in representatives:
                ladder_counter += 1
                lid = f"{category}_{ladder_counter:04d}"
                canonical_safe = cat_records[rep_local_i]["prompt_safe"]
                global_i = cat_indices[rep_local_i]
                selected_records.append({
                    **records[global_i],
                    "ladder_id": lid,
                    "cluster_safe_anchor": canonical_safe,
                    "cluster_size": len(cluster_local_indices),
                })

    try:
        model.to("cpu")
    except Exception:
        pass
    del model
    _release_torch_memory()

    log.info(
        "Assigned %d ladder IDs across %d categories (max_per_cluster=%d)",
        ladder_counter, len(by_category), max_per_cluster,
    )
    return selected_records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input",           default="data/metadata.jsonl",   help="Source JSONL file")
    p.add_argument("--output",          default="metadata_stage1.jsonl", help="Output JSONL file")
    p.add_argument("--model",           default="all-MiniLM-L6-v2",     help="Sentence-transformer model")
    p.add_argument("--threshold",       default=0.95, type=float,       help="Cosine similarity threshold (paper: 0.95)")
    p.add_argument("--batch-size",      default=512,  type=int,         help="Embedding batch size")
    p.add_argument("--max-per-cluster", default=1,    type=int,
                   help="Max diverse representatives per cluster (default: 1)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_path  = Path(args.input)
    output_path = Path(args.output)

    log.info("Stage 1 — Safe-Anchor Extraction & FAISS Clustering")
    log.info("Loading %s ...", input_path)
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
        max_per_cluster=args.max_per_cluster,
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
