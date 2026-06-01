"""Attack Strategy Analysis (Section 4.3).

Analyzes the effectiveness of jailbreak strategies stored in the
``rewrite_strategy`` field of rung records.

Strategies: surreal, medical, news, movie, legal, dream, negation, historical

Metrics:
  asr_per_strategy       : Attack Success Rate per strategy (VLM score ≥ 2.0)
  strategy_distribution  : frequency of each strategy in the dataset
  strategy_bypass_per_filter : (optional) per-strategy filter bypass rate

Paper reference: Section 4.3, Table 6
"""

from eval.benchmark.strategy_eval import compute  # noqa: F401

__all__ = ["compute"]
