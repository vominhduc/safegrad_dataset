"""Attack Success Rate (ASR) Evaluation (Section 4.3, Table 6).

Measures how effectively SafeGrad escalation ladders function as
adversarial attacks against T2I safety guardrails.

Two evaluation modes:
  CPU (default): uses pre-computed mllm_score_* fields from the dataset for
    VLM-confirmed ASR. No GPU required.
  GPU (with_filters=True): also runs image-based safety classifiers on the
    pre-generated high-risk images for per-image filter-bypass ASR.

Key metrics (Table 6):
  asr_vlm_L3       : fraction of L3 images with VLM score ≥ threshold
  asr_by_category  : per-category ASR at L3
  escalation_rate  : fraction of ladders with monotonically increasing scores
  filter_bypass_asr: (GPU) per-filter bypass rate at L3

Note on attack method implementations
--------------------------------------
The paper (Table 6) reports ASR for five attack strategies:
SneakyPrompt, Ring-A-Bell, UnlearnDiffAtk, MMA-Diffusion, and P4D.
These are specialized adversarial methods that require separate codebases
and are not re-implemented here.  The ASR metric computed by this module
treats SafeGrad ladders themselves as an attack strategy (Section 4.3).

Paper reference: Section 4.3, Table 6
"""

from eval.benchmark.attack_eval import compute  # noqa: F401

__all__ = ["compute"]
