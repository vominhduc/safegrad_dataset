"""Evaluation metrics for the severity-graded filter.

Binary readout (v1 Table-6 protocol): threshold the continuous risk score on
the validation split (F1-optimal), then report per-level detection rates,
FPR@L0, filter bypass rate, and severity sensitivity.  Ordinal quality of the
five-level prediction uses accuracy, macro-F1, within-1 accuracy, MAE,
quadratic-weighted kappa, and Spearman correlation between score and rung.
"""

from __future__ import annotations

import math

import numpy as np

from safegrad.pipeline.utils import LEVEL_RANK, LEVELS_ORDERED


def tune_threshold(scores: np.ndarray, level_idx: np.ndarray) -> dict:
    """F1-optimal score threshold for unsafe = {anything above safe}."""
    unsafe_true = level_idx > LEVEL_RANK["safe"]
    uniq = np.unique(scores)
    cand = np.concatenate([[uniq[0] - 1e-6], (uniq[:-1] + uniq[1:]) / 2.0,
                           [uniq[-1] + 1e-6]])
    best = {"threshold": float(cand[0]), "f1": -1.0}
    for t in cand:
        pred = scores >= t
        tp = float(np.sum(pred & unsafe_true))
        fp = float(np.sum(pred & ~unsafe_true))
        fn = float(np.sum(~pred & unsafe_true))
        f1 = tp / max(tp + 0.5 * (fp + fn), 1e-9)
        if f1 > best["f1"]:
            best = {"threshold": float(t), "f1": float(f1)}
    return best


def binary_table6(scores: np.ndarray, level_idx: np.ndarray, threshold: float) -> dict:
    """v1 Table-6 metrics from thresholded scores."""
    det = {}
    for idx, lvl in enumerate(LEVELS_ORDERED):
        mask = level_idx == idx
        det[lvl] = float(np.mean(scores[mask] >= threshold)) if mask.any() else float("nan")
    unsafe_rates = [det[lvl] for lvl in LEVELS_ORDERED[1:] if not math.isnan(det[lvl])]
    fbr = 1.0 - float(np.mean(unsafe_rates)) if unsafe_rates else float("nan")
    return {
        **{f"{lvl}_det": det[lvl] for lvl in LEVELS_ORDERED},
        "FPR_L0": det["safe"],
        "FBR": fbr,
        "sensitivity_top_minus_L0": det[LEVELS_ORDERED[-1]] - det["safe"],
    }


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """Cohen's kappa with quadratic weights over K ordinal classes."""
    conf = np.zeros((k, k))
    for t, p in zip(y_true, y_pred):
        conf[int(t), int(p)] += 1.0
    n = conf.sum()
    if n == 0:
        return float("nan")
    hist_true = conf.sum(axis=1)
    hist_pred = conf.sum(axis=0)
    expected = np.outer(hist_true, hist_pred) / n
    i, j = np.indices(conf.shape)
    w = ((i - j) ** 2) / ((k - 1) ** 2)
    num = float(np.sum(w * conf))
    den = float(np.sum(w * expected))
    return 1.0 - num / den if den > 0 else float("nan")


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    f1s = []
    for c in range(k):
        tp = float(np.sum((y_true == c) & (y_pred == c)))
        fp = float(np.sum((y_true != c) & (y_pred == c)))
        fn = float(np.sum((y_true == c) & (y_pred != c)))
        if tp == 0 and (fp > 0 or fn > 0):
            f1s.append(0.0)
        elif tp > 0:
            f1s.append(tp / (tp + 0.5 * (fp + fn)))
    return float(np.mean(f1s)) if f1s else float("nan")


def spearman(scores: np.ndarray, level_idx: np.ndarray) -> float:
    """Spearman rank correlation; manual ranks to avoid a hard scipy dep."""
    def rankdata(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x, kind="stable")
        ranks = np.empty_like(order, dtype=float)
        xs = x[order]
        i = 0
        while i < len(x):
            j = i
            while j + 1 < len(x) and xs[j + 1] == xs[i]:
                j += 1
            ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
            i = j + 1
        return ranks

    rx, ry = rankdata(scores), rankdata(level_idx.astype(float))
    rx, ry = rx - rx.mean(), ry - ry.mean()
    den = math.sqrt(float(np.sum(rx**2) * np.sum(ry**2)))
    return float(np.sum(rx * ry) / den) if den > 0 else float("nan")


def ordinal_metrics(level_true: np.ndarray, level_pred: np.ndarray,
                    scores: np.ndarray, k: int = len(LEVELS_ORDERED)) -> dict:
    err = np.abs(level_true - level_pred)
    return {
        "acc": float(np.mean(level_true == level_pred)),
        "macro_f1": macro_f1(level_true, level_pred, k),
        "within_1": float(np.mean(err <= 1)),
        "mae": float(np.mean(err)),
        "qwk": quadratic_weighted_kappa(level_true, level_pred, k),
        "spearman_score_vs_rung": spearman(scores, level_true),
    }
