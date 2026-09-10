"""Datasets and collators for the severity-graded filter (stage 1 SFT & stage 2)."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from safegrad.filter.common import render_messages, target_text

PIXEL_KW = dict(min_pixels=256 * 28 * 28, max_pixels=768 * 28 * 28)


class SFTDataset(Dataset):
    """Supervises the compact structured target ``Safety: <level>\\nCategories: <cat>``.

    Prompt tokens are masked out of the loss; only the target (plus the
    end-of-turn token) is supervised.
    """

    def __init__(self, examples: list[dict], image_root: str | Path, processor,
                 categories: list[str], condition: str = "prompt",
                 safe_category_none: bool = True):
        self.examples = examples
        self.image_root = Path(image_root)
        self.processor = processor
        self.categories = categories
        self.condition = condition
        self.safe_category_none = safe_category_none

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        img = Image.open(self.image_root / ex["image_path"]).convert("RGB")
        msgs = render_messages(ex, self.categories, self.condition)
        prompt = self.processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        tgt = target_text(ex, self.safe_category_none) + self.processor.tokenizer.eos_token

        enc_p = self.processor(text=[prompt], images=[img], return_tensors="pt", **PIXEL_KW)
        enc_f = self.processor(text=[prompt + tgt], images=[img], return_tensors="pt", **PIXEL_KW)
        ids = enc_f["input_ids"][0]
        labels = ids.clone()
        labels[: enc_p["input_ids"].shape[1]] = -100
        return {
            "input_ids": ids,
            "labels": labels,
            "attention_mask": enc_f["attention_mask"][0],
            "pixel_values": enc_f["pixel_values"],
            "image_grid_thw": enc_f["image_grid_thw"][0],
        }


class PromptDataset(Dataset):
    """Prompt-only encoding used for stage-2 feature extraction and ranking."""

    def __init__(self, examples: list[dict], image_root: str | Path, processor,
                 categories: list[str], condition: str = "prompt"):
        self.examples = examples
        self.image_root = Path(image_root)
        self.processor = processor
        self.categories = categories
        self.condition = condition

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        img = Image.open(self.image_root / ex["image_path"]).convert("RGB")
        msgs = render_messages(ex, self.categories, self.condition)
        prompt = self.processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        enc = self.processor(text=[prompt], images=[img], return_tensors="pt", **PIXEL_KW)
        return {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            # Qwen2.5-VL vision tensors have no batch dim: pixel_values is a flat
            # (total_patches, feat) patch sequence; keep it 2-D.
            "pixel_values": enc["pixel_values"],
            "image_grid_thw": enc["image_grid_thw"][0],
            "level_idx": torch.tensor(ex["level_idx"], dtype=torch.long),
        }


class CausalCollator:
    """Right-pads token tensors and concatenates vision tensors (v1 protocol)."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, feats: list[dict]) -> dict:
        maxlen = max(f["input_ids"].shape[0] for f in feats)
        ids, labels, att = [], [], []
        for f in feats:
            pad = maxlen - f["input_ids"].shape[0]
            ids.append(torch.cat([f["input_ids"], torch.full((pad,), self.pad_id)]))
            labels.append(torch.cat([f["labels"], torch.full((pad,), -100)]))
            att.append(torch.cat([f["attention_mask"], torch.zeros(pad, dtype=torch.long)]))
        return {
            "input_ids": torch.stack(ids),
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(att),
            "pixel_values": torch.cat([f["pixel_values"] for f in feats]),
            "image_grid_thw": torch.stack([f["image_grid_thw"] for f in feats]),
        }


class FeatureCollator:
    """Batch for head training/feature extraction.

    Right-padded (same geometry the stage-1 Trainer used — left padding was
    found to trigger device-side asserts in Qwen2.5-VL's vision tower).
    Returns ``seq_lens`` so callers can index each row's last non-pad token.
    """

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, feats: list[dict]) -> dict:
        maxlen = max(f["input_ids"].shape[0] for f in feats)
        ids, att = [], []
        for f in feats:
            pad = maxlen - f["input_ids"].shape[0]
            ids.append(torch.cat([f["input_ids"], torch.full((pad,), self.pad_id)]))
            att.append(torch.cat([f["attention_mask"], torch.zeros(pad, dtype=torch.long)]))
        att_t = torch.stack(att)
        return {
            "input_ids": torch.stack(ids),
            "attention_mask": att_t,
            "seq_lens": att_t.sum(dim=1),          # per-row real length
            "pixel_values": torch.cat([f["pixel_values"] for f in feats]),
            "image_grid_thw": torch.stack([f["image_grid_thw"] for f in feats]),
            "level_idx": torch.stack([f["level_idx"] for f in feats]),
        }
