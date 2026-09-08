"""Stage 2: train the ordinal and category heads on the frozen SFT backbone.

Freezes the stage-1 SFT model (backbone + LoRA adapter), extracts the final
hidden state per example, and trains:

  * the soft cumulative ordinal head (monotone thresholds + Gaussian-smoothed
    cumulative BCE targets) -> five-level distribution + risk score in [0,100]
  * the harm-category head (CE over categories + 'none')

Only head and threshold parameters are updated.  Hyperparameters follow the
SafeAtlas-VL head stage (head lr 1e-4, threshold lr 1e-3, gamma 0.75,
lambda_ord 1.0, lambda_cat 0.2, 1 epoch, cosine, warmup 0.03).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, set_seed

from safegrad.filter.common import (
    NONE_CATEGORY,
    build_examples,
    ladder_split,
    load_records,
    prune_missing,
)
from safegrad.filter.heads import (
    CategoryHead,
    OrdinalHead,
    ordinal_bce_loss,
)
from safegrad.filter.inference import extract_hidden, score_features, true_level_idx
from safegrad.filter.metrics import binary_table6, ordinal_metrics, tune_threshold
from safegrad.pipeline.utils import LEVELS_ORDERED


def load_sft_model(model_name: str, adapter: str | None, device: str):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map=device,
        attn_implementation="sdpa")
    if adapter and adapter != "none":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()     # fold LoRA in; everything stays frozen
        print(f"loaded and merged adapter {adapter}", flush=True)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--split-file", default=None,
                    help="split.json from stage 1; recomputed from --dataset/--seed if absent")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--adapter", default="none", help="stage-1 adapter dir (e.g. adapter_best)")
    ap.add_argument("--condition", default="prompt", choices=["prompt", "none"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--feat-batch", type=int, default=8)
    ap.add_argument("--head-lr", type=float, default=1e-4)
    ap.add_argument("--threshold-lr", type=float, default=1e-3)
    ap.add_argument("--mlp-dim", type=int, default=512)
    ap.add_argument("--gamma", type=float, default=0.75)
    ap.add_argument("--lambda-ord", type=float, default=1.0)
    ap.add_argument("--lambda-cat", type=float, default=0.2)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    records = load_records(args.dataset)
    categories = sorted({r["category"] for r in records})
    cat_list = [*categories, NONE_CATEGORY]
    cat_index = {c: i for i, c in enumerate(cat_list)}
    split = (json.load(open(args.split_file)) if args.split_file
             else ladder_split(records, args.seed))

    unit = {lv: 1 for lv in LEVELS_ORDERED}
    train_ex = prune_missing(build_examples(records, set(split["train"]), unit), args.image_root)
    val_ex = prune_missing(build_examples(records, set(split["val"]), unit), args.image_root)
    print(f"train {len(train_ex)} / val {len(val_ex)} examples", flush=True)

    processor = AutoProcessor.from_pretrained(args.model)
    model = load_sft_model(args.model, args.adapter, device)
    hidden_dim = getattr(model.config, "hidden_size", None)
    if hidden_dim is None:
        hidden_dim = model.config.text_config.hidden_size

    print("extracting features (train)...", flush=True)
    f_train = extract_hidden(model, processor, train_ex, args.image_root, categories,
                             args.condition, device, args.feat_batch)
    print("extracting features (val)...", flush=True)
    f_val = extract_hidden(model, processor, val_ex, args.image_root, categories,
                           args.condition, device, args.feat_batch)

    ord_head = OrdinalHead(hidden_dim, args.mlp_dim).to(device)
    cat_head = CategoryHead(hidden_dim, len(cat_list)).to(device)
    opt = torch.optim.AdamW([
        {"params": [p for n, p in ord_head.named_parameters() if "thresholds" not in n],
         "lr": args.head_lr},
        {"params": list(ord_head.thresholds.parameters()), "lr": args.threshold_lr},
        {"params": cat_head.parameters(), "lr": args.head_lr},
    ])

    y_train = torch.tensor(true_level_idx(train_ex), dtype=torch.long)
    c_train = torch.tensor(
        [cat_index[NONE_CATEGORY if ex["level"] == "safe" else ex["category"]]
         for ex in train_ex], dtype=torch.long)

    n = f_train.shape[0]
    steps_per_epoch = math.ceil(n / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[args.head_lr, args.threshold_lr, args.head_lr],
        total_steps=total_steps, pct_start=0.03)

    g = torch.Generator().manual_seed(args.seed)
    for epoch in range(args.epochs):
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for i in range(0, n, args.batch_size):
            idx = perm[i : i + args.batch_size]
            h = f_train[idx].to(device)
            l_ord = ordinal_bce_loss(ord_head.cumulative_probs(h),
                                     y_train[idx].to(device), args.gamma)
            l_cat = F.cross_entropy(cat_head(h), c_train[idx].to(device))
            loss = args.lambda_ord * l_ord + args.lambda_cat * l_cat
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(ord_head.parameters()) + list(cat_head.parameters()), 1.0)
            opt.step()
            sched.step()
            tot += float(loss)
        print(f"epoch {epoch}: mean loss {tot / steps_per_epoch:.4f}", flush=True)

    # ---- validation: tune threshold, report metric bundle -------------------
    val_true = true_level_idx(val_ex)
    sv = score_features(ord_head, cat_head, f_val, device)
    thr = tune_threshold(sv["score"], val_true)
    m_ord = ordinal_metrics(val_true, sv["pred_level_idx"], sv["score"])
    m_bin = binary_table6(sv["score"], val_true, thr["threshold"])
    print(f"val: threshold {thr['threshold']:.2f} (F1 {thr['f1']:.4f})", flush=True)
    print(f"val ordinal: {m_ord}", flush=True)
    print(f"val binary : {m_bin}", flush=True)

    torch.save({"ord_head": ord_head.state_dict(), "cat_head": cat_head.state_dict()},
               out / "heads.pt")
    with open(out / "heads_config.json", "w") as f:
        json.dump(vars(args) | {
            "levels": list(LEVELS_ORDERED),
            "categories": cat_list,
            "hidden_dim": hidden_dim,
            "val_threshold": thr,
            "val_ordinal": m_ord,
            "val_binary": m_bin,
        }, f, indent=2)
    with open(out / "val_predictions.csv", "w") as f:
        f.write("ladder_id,category,level,pred_level,score\n")
        for ex, pi, s in zip(val_ex, sv["pred_level_idx"], sv["score"]):
            f.write(f"{ex['ladder_id']},{ex['category']},{ex['level']},"
                    f"{LEVELS_ORDERED[int(pi)]},{float(s):.2f}\n")
    print("DONE.", flush=True)


if __name__ == "__main__":
    main()
