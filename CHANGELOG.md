# Changelog

## Unreleased (v2)

- Taxonomy: `safegrad.scripts.sync_rules` regenerates `data/rules.jsonl`
  deterministically from the canonical risk-taxonomy file
  (`shares/new_risk_category.jsonl`: 19 categories, five-level rules; levels
  marked "(Does not exist)" are skipped). Rule texts are unchanged; each row
  now carries its source `risk_id` for provenance.

- `safegrad.filter`: severity-graded safety filter replacing the v1 binary
  YES/NO filter — two-stage training (safety SFT producing a five-level
  structured judgment; then a soft cumulative ordinal head with monotone
  thresholds + Gaussian-smoothed cumulative targets, and a harm-category
  head, on the frozen backbone). Continuous risk score in [0, 100].
  Ordinal scoring follows SafeAtlas-VL (Wang et al., 2026) adapted to ladder
  supervision. Evaluation keeps the v1 Table-6 binary protocol
  (validation-tuned score threshold) and adds ordinal metrics
  (acc / macro-F1 / within-1 / MAE / QWK / Spearman).
- `pyproject.toml`: `filter` extra (`peft`) and `safegrad-filter-*` entry points.
