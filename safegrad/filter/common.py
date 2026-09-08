"""Shared data plumbing for the SafeGrad v2 severity-graded filter.

Ladder JSONL schema (one record per ladder, produced by the ASL pipeline)::

    {"ladder_id": ..., "category": ...,
     "rung_safe": {"image_path", "explanation", "prompt", ...},
     "rung_low_risk": {...}, ..., "rung_very_high_risk": {...}}

The split rule (80/10/10 by ``ladder_id``, fixed seed) matches the v1
remediation protocol so that v2 numbers remain protocol-comparable with the
paper's Table 6 tuned row.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from safegrad.pipeline.utils import LEVELS_ORDERED, norm_level

K_LEVELS: int = len(LEVELS_ORDERED)
NONE_CATEGORY: str = "none"

#: v1 emphasised the boundary rungs (L1/L2 x2); keep the analogue for five
#: levels and expose ``--oversample`` to override.
DEFAULT_OVERSAMPLE: dict[str, int] = {
    "safe": 1,
    "low_risk": 2,
    "mid_risk": 2,
    "high_risk": 1,
    "very_high_risk": 1,
}

CONDITION_MODES: tuple[str, ...] = ("prompt", "none")


def rung_keys() -> list[str]:
    return [f"rung_{level}" for level in LEVELS_ORDERED]


def load_records(dataset_path: str | Path) -> list[dict]:
    with open(dataset_path) as f:
        return [json.loads(line) for line in f if line.strip()]


def ladder_split(
    records: list[dict], seed: int, ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
) -> dict:
    """80/10/10 split with ladder integrity, identical rule to v1."""
    ids = [r["ladder_id"] for r in records]
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(ratios[0] * n)
    n_val = int(ratios[1] * n)
    return {
        "seed": seed,
        "rule": "80/10/10 by ladder_id, shuffled with random.Random(seed)",
        "n_ladders": n,
        "train": ids[:n_train],
        "val": ids[n_train : n_train + n_val],
        "test": ids[n_train + n_val :],
    }


def parse_oversample(spec: str | None) -> dict[str, int]:
    """Parse ``--oversample low_risk:2,mid_risk:2`` style overrides."""
    if not spec:
        return dict(DEFAULT_OVERSAMPLE)
    out = {level: 1 for level in LEVELS_ORDERED}
    for part in spec.split(","):
        level, _, reps = part.partition(":")
        level = norm_level(level.strip())
        out[level] = int(reps)
    return out


def build_examples(
    records: list[dict],
    ladder_ids: set,
    oversample: dict[str, int] | None = None,
) -> list[dict]:
    """Flatten ladder records into per-rung (image, level, category) examples."""
    oversample = oversample or {level: 1 for level in LEVELS_ORDERED}
    out: list[dict] = []
    for rec in records:
        if rec["ladder_id"] not in ladder_ids:
            continue
        for level_idx, (key, level) in enumerate(zip(rung_keys(), LEVELS_ORDERED)):
            rung = rec.get(key) or {}
            if not rung.get("image_path"):
                continue
            ex = {
                "ladder_id": rec["ladder_id"],
                "category": rec["category"],
                "level": level,
                "level_idx": level_idx,
                "image_path": rung["image_path"],
                "explanation": rung.get("explanation", ""),
                "prompt": rung.get("prompt", ""),
            }
            out.extend([ex] * int(oversample.get(level, 1)))
    return out


def prune_missing(examples: list[dict], image_root: str | Path) -> list[dict]:
    root = Path(image_root)
    keep = [ex for ex in examples if (root / ex["image_path"]).exists()]
    dropped = len(examples) - len(keep)
    if dropped:
        print(f"WARNING: dropped {dropped}/{len(examples)} examples with missing images",
              flush=True)
    return keep


def system_prompt(categories: list[str]) -> str:
    cat_list = "\n".join(f"  - {c}" for c in [*categories, NONE_CATEGORY])
    level_list = "\n".join(f"  - {lv}" for lv in LEVELS_ORDERED)
    return (
        "You are a multimodal safety auditor specializing in AI-generated images.\n"
        "Given an image (and, when provided, the text-to-image request that\n"
        "produced it), assess its safety severity on the ordered scale:\n"
        f"{level_list}\n"
        "Also assign the harm category that best describes the risk\n"
        f"(use '{NONE_CATEGORY}' for safe content):\n"
        f"{cat_list}\n"
        "Respond in exactly this format:\n"
        "Safety: <level>\n"
        "Categories: <category>"
    )


def render_messages(ex: dict, categories: list[str], condition: str = "prompt") -> list[dict]:
    """Build the chat messages for one example.

    ``condition='prompt'`` shows the rung's T2I prompt (deployment-realistic,
    matches the v1 prompt-aware inference finding); ``'none'`` judges the
    image alone.  The rung *explanation* is never shown: it describes the
    target rung and would leak the label.
    """
    if condition not in CONDITION_MODES:
        raise ValueError(f"unknown condition mode: {condition}")
    user_content: list[dict] = [{"type": "image"}]
    if condition == "prompt" and ex.get("prompt"):
        user_content.append({"type": "text", "text": f"T2I request: {ex['prompt']}"})
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt(categories)}]},
        {"role": "user", "content": user_content},
    ]


def target_text(ex: dict, safe_category_none: bool = True) -> str:
    cat = NONE_CATEGORY if (safe_category_none and ex["level"] == "safe") else ex["category"]
    return f"Safety: {ex['level']}\nCategories: {cat}"
