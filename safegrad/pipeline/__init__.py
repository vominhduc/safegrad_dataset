"""SafeGrad pipeline — four-stage Automated Severity Ladder construction.

Stages (Section 4 of the paper)
---------------------------------
  stage0_seed_generation  — Safe seed prompt generation (pre-stage)
  stage1_clustering       — Deduplication and FAISS clustering (Stage 1)
  stage2_interpolation    — Severity judge + prompt synthesis (Stage 2)
  stage3_synthesis        — Reference image generation (Stage 3)
  stage4_verification     — Visual monotonicity verification (Stage 4)
"""
