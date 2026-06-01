"""Safety Filter Evaluation (Section 4.1, Table 5).

Tests how well safety classifiers detect unsafe content at each severity level.

Supported filters (configured via --safety-filters):
  pipeline_judge : Ensemble VLM judge (requires safety_image_data_generation repo)
  clip_nsfw      : CLIP-based NSFW classifier (openai/clip-vit-large-patch14)
  q16            : Q16 CLIP-ViT-L/14 trained on LAION-5B safety labels
  sd_filter      : Stable Diffusion safety checker (CompVis/stable-diffusion-safety-checker)
  llava_guard    : LlavaGuard VLM safety classifier (AIML-TUDA/LlavaGuard-7B)
  internvl       : InternVL2-8B content moderation

Metrics per filter:
  tpr  : True Positive Rate — correctly flags high_risk as UNSAFE
  fpr  : False Positive Rate — incorrectly flags safe as UNSAFE
  fbr  : Filter Bypass Rate = 1 − TPR (fraction of high_risk that evade the filter)
  auc_approx : mean detection rate across all rung levels
  detection_rate_by_level : per-level detection rate

Paper reference: Section 4.1, Table 5
"""

from eval.benchmark.filter_eval import compute  # noqa: F401
from eval.benchmark.filter_eval import (  # noqa: F401
    _load_clip_nsfw,
    _load_q16,
    _load_sd_filter,
    _load_llava_guard,
    _load_internvl,
    _pipeline_judge_classify,
    _load_image,
    _ORDERED,
    _POSITIVE_LEVELS,
    _NEGATIVE_LEVELS,
    _FILTER_LOADERS,
)

__all__ = [
    "compute",
    "_load_clip_nsfw",
    "_load_q16",
    "_load_sd_filter",
    "_load_llava_guard",
    "_load_internvl",
    "_FILTER_LOADERS",
    "_ORDERED",
]
