"""Severity-graded safety filter (SafeGrad v2).

Trains a ladder-aware multimodal filter that replaces v1's binary YES/NO
verdict with an ordered five-level judgment plus a continuous risk score:

  Stage 1 (``train_sft.py``)    — safety instruction tuning (LoRA) producing a
                                  compact structured judgment (level + category)
  Stage 2 (``train_heads.py``)  — frozen backbone; soft cumulative ordinal head
                                  (monotone thresholds, Gaussian-smoothed
                                  cumulative targets) + category head
  Eval    (``eval_filter.py``)  — Table-6 binary readout (per-level detection,
                                  FPR@L0, FBR, sensitivity) via a validation-
                                  tuned score threshold, plus ordinal metrics
                                  (acc, macro-F1, within-1, MAE, QWK, Spearman)

The ordinal scoring formulation follows the soft cumulative ordinal head of
SafeAtlas-VL (Wang et al., 2026), adapted to ladder supervision.
"""

from safegrad.filter.common import (  # noqa: F401
    DEFAULT_OVERSAMPLE,
    NONE_CATEGORY,
    build_examples,
    ladder_split,
    load_records,
    parse_oversample,
    render_messages,
    rung_keys,
)
from safegrad.filter.heads import (  # noqa: F401
    CategoryHead,
    OrdinalHead,
    OrdinalThresholds,
    expected_risk_score,
    gaussian_soft_targets,
    level_probabilities,
    ordinal_bce_loss,
)
