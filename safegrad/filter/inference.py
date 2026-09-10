"""Frozen-backbone feature extraction and scoring helpers for the filter heads."""

from __future__ import annotations

import numpy as np
import torch

from safegrad.filter.sft_data import FeatureCollator, PromptDataset


@torch.no_grad()
def extract_hidden(model, processor, examples, image_root, categories, condition,
                   device, batch_size: int = 8) -> torch.Tensor:
    """Last-layer hidden state at the final (left-padded) position, ``(N, d)``.

    Batches through the processor once; returns CPU fp32 features.
    """
    from torch.utils.data import DataLoader

    ds = PromptDataset(examples, image_root, processor, categories, condition)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=FeatureCollator(processor.tokenizer.pad_token_id),
                        num_workers=0)
    feats = []
    model.eval()
    for bi, batch in enumerate(loader):
        seq_lens = batch.pop("seq_lens")
        level_idx = batch.pop("level_idx")
        batch = {k: v.to(device) for k, v in batch.items()}
        hs = model(**batch, output_hidden_states=True).hidden_states[-1]
        last_idx = (seq_lens - 1).to(device)                    # last non-pad token per row
        row_idx = torch.arange(hs.shape[0], device=device)
        feats.append(hs[row_idx, last_idx, :].float().cpu())
        if (bi + 1) % 25 == 0:
            print(f"  features {bi + 1}/{len(loader)}", flush=True)
    return torch.cat(feats)


@torch.no_grad()
def score_features(ord_head, cat_head, features: torch.Tensor, device,
                   batch_size: int = 256) -> dict:
    """Ordinal + category predictions and continuous risk scores from features."""
    from safegrad.filter.heads import expected_risk_score

    ord_head.eval()
    if cat_head is not None:
        cat_head.eval()
    probs, scores, cat_pred = [], [], []
    for i in range(0, features.shape[0], batch_size):
        h = features[i : i + batch_size].to(device)
        p = ord_head.level_probs(h)
        probs.append(p.cpu())
        scores.append(expected_risk_score(p).cpu())
        if cat_head is not None:
            cat_pred.append(cat_head(h).argmax(dim=-1).cpu())
    probs = torch.cat(probs).numpy()
    return {
        "level_probs": probs,
        "pred_level_idx": probs.argmax(axis=1).astype(int),
        "score": torch.cat(scores).numpy(),
        "pred_cat_idx": torch.cat(cat_pred).numpy() if cat_pred else None,
    }


def true_level_idx(examples: list[dict]) -> np.ndarray:
    return np.array([ex["level_idx"] for ex in examples], dtype=int)
