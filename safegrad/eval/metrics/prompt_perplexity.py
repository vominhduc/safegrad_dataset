"""Prompt perplexity evaluation (Section 4.3).

Measures GPT-2 perplexity of prompts at each severity level.  A low
perplexity at high_risk relative to safe indicates that harmful prompts
are linguistically fluent and stealthy — they do not stand out to
language-model-based filters.

Key finding (paper, Section 4.3): PPL ratio L3/L0 = 0.57, meaning
high-risk prompts are *more* fluent than safe prompts.

Usage
-----
    from safegrad.eval.metrics.prompt_perplexity import compute
    result = compute(ladders, model_name="gpt2")
    print(result["ppl_ratio_L3_L0"])   # e.g. 0.572
"""

from eval.metrics.prompt_perplexity import compute  # noqa: F401

__all__ = ["compute"]
