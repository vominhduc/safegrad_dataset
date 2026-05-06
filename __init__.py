"""SafeGrad: Automated Severity Ladder (ASL) construction pipeline and evaluation suite.

Paper: "SafeGrad: Benchmarking Text-to-Image Models with Automatically Generated
       Severity Ladders" (NeurIPS 2026)

Repository layout
-----------------
  safegrad/pipeline/    — Four-stage ASL construction pipeline (Section 3)
  safegrad/eval/        — Evaluation suite matching paper Tables 1–6 (Section 4)
  safegrad/scripts/     — Utility scripts (metadata export, pipeline runner)
"""

import sys as _sys
from pathlib import Path as _Path

# Ensure the repo root (parent of this package) is on sys.path so that the
# bare `pipeline.*` and `eval.*` imports in wrapper modules resolve correctly.
_repo_root = str(_Path(__file__).resolve().parents[1])
if _repo_root not in _sys.path:
    _sys.path.insert(0, _repo_root)
