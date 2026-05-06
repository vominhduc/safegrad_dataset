"""Stage 4: Visual Monotonicity Verification (Section 3.4).

The fourth and final stage of the Automated Severity Ladder (ASL) pipeline.

For every ladder record (with images already generated in Stage 3), a vision
LLM scores each rung on a 0–3 scale and enforces visual monotonicity:

    score_safe < score_low_risk < score_mid_risk < score_high_risk

with a minimum adjacent gap of 0.4 (paper default).  Ladders that fail the
monotonicity constraint are discarded.  Only ladders where all four rungs pass
verification are included in the final dataset.

Vision LLM: Qwen/Qwen3-VL-8B-Thinking (paper, Section 3.4)
Monotonicity gap threshold: 0.4 (Section 3.4)

Paper reference: Section 3, Stage 4 ("Visual Verification")

Note
----
This stage is implemented in pipeline/phase4_filter.py (VLM scoring +
monotonicity check).  The diversity/bias curation in pipeline/phase5_filter.py
is a dataset preparation step used to produce the released subset and is not
part of the core 4-stage pipeline described in the paper.

Usage
-----
    uv run python -m safegrad.pipeline.stage4_verification [OPTIONS]

Options
-------
    --input          Source JSONL (ladder format with images)
                                                       [default: metadata_stage3.jsonl]
    --output         Output JSONL (verified ladders)   [default: metadata_stage4.jsonl]
    --image-root     Root dir to resolve image paths   [default: .]
    --project-root   Root dir for any re-generated images [default: .]
    --rules          Path to rules JSONL               [default: data/rules.jsonl]
    --vlm-model      Vision LLM for scoring            [default: Qwen/Qwen3-VL-8B-Thinking]
    --min-score-gap  Monotonicity adjacency gap        [default: 0.4]
    --batch-size     VLM batch size                    [default: 4]
    --no-generate    Do not generate missing images; mark rung as unaudited instead
"""

import sys

import pipeline.phase4_filter as _phase4


def main() -> None:
    _phase4.main()


if __name__ == "__main__":
    _phase4.main()
