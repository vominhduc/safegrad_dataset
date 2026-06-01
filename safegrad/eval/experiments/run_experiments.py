"""Runner for NeurIPS paper experiments.

Delegates to eval/experiments/run_experiments.py — see that module for
full documentation.  The safegrad namespace adds exp2_hgr_sbs which
is the canonical Table 3 reproduction.
"""
from eval.experiments.run_experiments import main  # noqa: F401
from eval.experiments.run_experiments import parse_args  # noqa: F401

__all__ = ["main", "parse_args"]
