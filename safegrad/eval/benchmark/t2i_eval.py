"""T2I Model Safety Evaluation (Section 4.1, Table 3).

Generates images with six T2I models evaluated in the paper and scores them
with a VLM judge (Qwen/Qwen3-VL-8B-Thinking, default).

T2I models evaluated (Table 3):
  sd14        : Stable Diffusion 1.4 (CompVis/stable-diffusion-v1-4)
  sd21        : Stable Diffusion 2.1 (stabilityai/stable-diffusion-2-1)
  pixart_alpha: PixArt-α (PixArt-alpha/PixArt-XL-2-1024-MS)
  cogview4    : CogView4 (THUDM/CogView4-6B)
  janus_pro   : Janus-Pro-7B (deepseek-ai/Janus-Pro-7B)
  hidream     : HiDream-I1-Full (HiDream-ai/HiDream-I1-Full)

Metrics:
  HGR (Harmful Generation Rate) per model and category — see hgr.py
  mean_vlm_score_by_level : mean VLM score per rung level per model

Paper reference: Section 4.1, Table 3

Note: Requires GPU (~16–80 GB VRAM depending on model).
"""

from eval.benchmark.t2i_eval import compute, MODEL_MAP  # noqa: F401

__all__ = ["compute", "MODEL_MAP"]
