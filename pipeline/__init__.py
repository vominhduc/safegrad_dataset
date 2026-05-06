"""SafeGrad pipeline — four stages of Automated Severity Ladder construction.

Stages (Section 3)
------------------
  stage1_clustering   — Safe-anchor extraction + FAISS deduplication
  stage2_interpolation — Severity judge + prompt ladder interpolation (LLM)
  stage3_synthesis    — T2I image synthesis (SDXL / Z-Turbo / FLUX.1 / SD 3.5)
  stage4_verification — Visual monotonicity verification (Qwen3-VL-8B)
"""
