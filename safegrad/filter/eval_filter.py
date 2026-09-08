"""Evaluate a trained severity-graded filter on a ladder split.

Binary readout follows the v1 Table-6 protocol: the continuous risk score is
thresholded (F1-optimal on the *validation* split unless ``--threshold`` is
given), then per-level detection rates, FPR@L0, FBR and severity sensitivity
are reported.  Ordinal quality of the five-level prediction is reported via
accuracy, macro-F1, within-1 accuracy, MAE, QWK and Spearman(score, rung).

Usage (tuned model, test split):
  uv run python -m safegrad.filter.eval_filter \
      --dataset data/export/metadata.jsonl --image-root data/export \
      --split-file runs/sft/split.json --split test \
      --adapter runs/sft/adapter_best --heads runs/heads/heads.pt \
      --out runs/heads/test_eval.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoProcessor

from safegrad.filter.common import build_examples, load_records, prune_missing
from safegrad.filter.heads import CategoryHead, OrdinalHead
from safegrad.filter.inference import extract_hidden, score_features, true_level_idx
from safegrad.filter.metrics import binary_table6, ordinal_metrics, tune_threshold
from safegrad.filter.train_heads import load_sft_model
from safegrad.pipeline.utils import LEVELS_ORDERED


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--split-file", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--adapter", default="none")
    ap.add_argument("--heads", required=True, help="heads.pt from stage 2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--condition", default="prompt", choices=["prompt", "none"])
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the F1-optimal threshold tuned on the val split")
    ap.add_argument("--feat-batch", type=int, default=8)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    records = load_records(args.dataset)
    categories = sorted({r["category"] for r in records})
    split = json.load(open(args.split_file))

    ckpt = torch.load(args.heads, map_location="cpu", weights_only=True)
    cfg = json.load(open(Path(args.heads).parent / "heads_config.json"))
    cat_list = cfg["categories"]

    processor = AutoProcessor.from_pretrained(args.model)
    model = load_sft_model(args.model, args.adapter, device)
    ord_head = OrdinalHead(cfg["hidden_dim"], cfg["mlp_dim"]).to(device)
    cat_head = CategoryHead(cfg["hidden_dim"], len(cat_list)).to(device)
    ord_head.load_state_dict(ckpt["ord_head"])
    cat_head.load_state_dict(ckpt["cat_head"])

    def run_split(name: str):
        exs = prune_missing(build_examples(records, set(split[name])), args.image_root)
        feats = extract_hidden(model, processor, exs, args.image_root, categories,
                               args.condition, device, args.feat_batch)
        return exs, true_level_idx(exs), score_features(ord_head, cat_head, feats, device)

    if args.threshold is None:
        print("tuning threshold on val split...", flush=True)
        _, yv, svres = run_split("val")
        thr = tune_threshold(svres["score"], yv)["threshold"]
    else:
        thr = args.threshold
    print(f"score threshold: {thr:.2f}", flush=True)

    print(f"scoring {args.split} split...", flush=True)
    exs, y_true, res = run_split(args.split)
    metrics = {
        "threshold": thr,
        "n_examples": len(exs),
        "binary": binary_table6(res["score"], y_true, thr),
        "ordinal": ordinal_metrics(y_true, res["pred_level_idx"], res["score"]),
        "score_stats_by_level": {
            lvl: {"mean": float(res["score"][y_true == i].mean()),
                  "std": float(res["score"][y_true == i].std())}
            for i, lvl in enumerate(LEVELS_ORDERED) if (y_true == i).any()
        },
    }
    print(json.dumps(metrics, indent=2), flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"adapter": args.adapter, "heads": args.heads, "split": args.split,
                   "metrics": metrics}, f, indent=2)
    csv_path = str(args.out).replace(".json", "_predictions.csv")
    with open(csv_path, "w") as f:
        f.write("ladder_id,category,level,pred_level,pred_category,score\n")
        for ex, pi, ci, s in zip(exs, res["pred_level_idx"], res["pred_cat_idx"],
                                 res["score"]):
            f.write(f"{ex['ladder_id']},{ex['category']},{ex['level']},"
                    f"{LEVELS_ORDERED[int(pi)]},{cat_list[int(ci)]},{float(s):.2f}\n")
    print(f"saved {args.out} and {csv_path}", flush=True)


if __name__ == "__main__":
    main()
