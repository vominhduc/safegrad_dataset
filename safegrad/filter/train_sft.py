"""Stage 1: safety instruction tuning for the severity-graded filter.

Trains a VLM (default Qwen2.5-VL-7B-Instruct, LoRA) to emit the compact
structured judgment ``Safety: <level>`` / ``Categories: <category>`` for each
(image, rung) pair of the ladder dataset.

Protocol differences from the v1 filter (``SafeGrad_paper`` scripts):
  * supervision is the five-level ordinal target instead of binary YES/NO
  * conditioning defaults to the rung's T2I prompt (deployment-realistic;
    the rung explanation is never shown to avoid label leakage)
  * per-epoch checkpoint selection uses within-1 level accuracy on the
    validation split, ranked by mean log-probability of each
    ``"Safety: <level>"`` candidate (no generation needed)

LoRA recipe, split rule (80/10/10 by ladder, seed 42), per-epoch Trainer
rebuild and adapter-per-epoch auditability mirror the v1 script exactly.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
    set_seed,
)

from safegrad.filter.common import (
    build_examples,
    ladder_split,
    load_records,
    parse_oversample,
    prune_missing,
    render_messages,
)
from safegrad.filter.metrics import ordinal_metrics
from safegrad.filter.sft_data import PIXEL_KW, CausalCollator, SFTDataset
from safegrad.pipeline.utils import LEVELS_ORDERED


def rank_level_logprobs(model, processor, example, image_root, categories, condition, device):
    """Rank levels by teacher-forced mean log-prob of each 'Safety: <level>' candidate."""
    img = Image.open(Path(image_root) / example["image_path"]).convert("RGB")
    msgs = render_messages(example, categories, condition)
    prompt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc_p = processor(text=[prompt], images=[img], return_tensors="pt", **PIXEL_KW)
    n_prompt = enc_p["input_ids"].shape[1]
    means = []
    with torch.no_grad():
        for level in LEVELS_ORDERED:
            enc = processor(text=[prompt + f"Safety: {level}"], images=[img],
                            return_tensors="pt", **PIXEL_KW)
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits[0]                      # (seq, vocab)
            ids = enc["input_ids"][0]
            lp = torch.log_softmax(logits.float(), dim=-1)
            token_lp = lp[n_prompt - 1 : -1].gather(-1, ids[n_prompt:][:, None]).squeeze(-1)
            means.append(float(token_lp.mean()))
    return int(np.argmax(means))


@torch.no_grad()
def level_predictions(model, processor, examples, image_root, categories, condition,
                      device, max_n: int = 200):
    preds, trues = [], []
    for ex in examples[:max_n]:
        preds.append(rank_level_logprobs(model, processor, ex, image_root,
                                         categories, condition, device))
        trues.append(ex["level_idx"])
    return np.array(trues), np.array(preds)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="Ladder JSONL (export of the ASL pipeline)")
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--condition", default="prompt", choices=["prompt", "none"])
    ap.add_argument("--oversample", default=None,
                    help="e.g. 'low_risk:2,mid_risk:2'; default emphasises boundary rungs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--per-device-batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--val-rank-n", type=int, default=200,
                    help="Val examples used for within-1 selection ranking")
    args = ap.parse_args()

    set_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    records = load_records(args.dataset)
    categories = sorted({r["category"] for r in records})
    split = ladder_split(records, args.seed)
    with open(out / "split.json", "w") as f:
        json.dump(split, f, indent=2)
    print(f"ladders: {split['n_ladders']} -> train {len(split['train'])}, "
          f"val {len(split['val'])}, test {len(split['test'])}", flush=True)

    oversample = parse_oversample(args.oversample)
    unit = {lv: 1 for lv in LEVELS_ORDERED}
    train_ex = prune_missing(build_examples(records, set(split["train"]), oversample),
                             args.image_root)
    val_ex = prune_missing(build_examples(records, set(split["val"]), unit), args.image_root)
    random.Random(args.seed).shuffle(train_ex)
    print(f"train pairs (oversampled): {len(train_ex)}; val pairs: {len(val_ex)}",
          flush=True)

    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa")
    model.config.use_cache = False

    from peft import LoraConfig, get_peft_model
    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=r".*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj).*")
    model = get_peft_model(model, lora)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    train_ds = SFTDataset(train_ex, args.image_root, processor, categories, args.condition)
    collator = CausalCollator(processor.tokenizer.pad_token_id)

    best = {"score": -1e9, "epoch": None, "metrics": None}
    history = []
    for epoch in range(args.epochs):
        print(f"=== epoch {epoch} ===", flush=True)
        targs = TrainingArguments(
            output_dir=str(out / f"trainer_epoch{epoch}"),
            per_device_train_batch_size=args.per_device_batch,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=1.0,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            weight_decay=0.0,
            logging_steps=10,
            save_strategy="no",
            bf16=True,
            report_to=[],
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            dataloader_num_workers=2,
            remove_unused_columns=False,
            seed=args.seed,
        )
        Trainer(model=model, args=targs, train_dataset=train_ds,
                data_collator=collator).train()

        model.eval()
        rng = random.Random(args.seed + epoch)
        val_sub = rng.sample(val_ex, min(args.val_rank_n, len(val_ex)))
        trues, preds = level_predictions(model, processor, val_sub, args.image_root,
                                         categories, args.condition, "cuda")
        m = ordinal_metrics(trues, preds, np.zeros(len(trues)))
        # selection: within-1 accuracy first, exact accuracy as tiebreak (SafeAtlas:
        # adjacent-level confusion dominates, so within-1 is the robust criterion)
        score = m["within_1"] + 0.25 * m["acc"]
        print(f"val epoch {epoch}: {m} | select-score {score:.4f}", flush=True)
        history.append({"epoch": epoch, **m})
        ck = out / f"adapter_epoch{epoch}"
        model.save_pretrained(ck)
        processor.save_pretrained(ck)
        with open(ck / "val_metrics.json", "w") as f:
            json.dump(m, f, indent=2)
        if score > best["score"]:
            best = {"score": score, "epoch": epoch, "metrics": m}
        model.train()

    best_dir = out / "adapter_best"
    if best_dir.exists():
        shutil.rmtree(best_dir)
    shutil.copytree(out / f"adapter_epoch{best['epoch']}", best_dir)

    with open(out / "training_config.json", "w") as f:
        json.dump(vars(args) | {
            "levels": list(LEVELS_ORDERED),
            "categories": categories,
            "oversample": oversample,
            "selection": "max(within_1_acc + 0.25*acc) via Safety-token logprob ranking",
            "best_epoch": best["epoch"], "best_val": best,
        }, f, indent=2)
    with open(out / "val_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("DONE. best:", best, flush=True)


if __name__ == "__main__":
    main()
