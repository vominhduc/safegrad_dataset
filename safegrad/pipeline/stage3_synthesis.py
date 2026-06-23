"""Stage 3: T2I Image Synthesis (Section 3, Stage 3).

The third stage of the Automated Severity Ladder (ASL) pipeline.

For each rung that lacks an image, generates one using the T2I model
specified in the ladder's ``generator_model`` field.  Supports three
commercially-licensed T2I models (Apache-2.0):

  - ``sdxl``   : Stable Diffusion XL (stabilityai/stable-diffusion-xl-base-1.0)
  - ``flux1``  : FLUX.1-schnell (black-forest-labs/FLUX.1-schnell)
  - ``kolors`` : Kolors (Kwai-Kolors/Kolors)

Note: Image generation requires a GPU and ~10-20 GB VRAM per model.
The VLM verification (monotonicity scoring) runs in Stage 4.

Usage
-----
    uv run python -m safegrad.pipeline.stage3_synthesis [OPTIONS]

Options
-------
    --input      Source JSONL (ladder format)     [default: data/stage2_out.jsonl]
    --output     Output JSONL with image paths    [default: data/stage3_out.jsonl]
    --image-dir  Directory to save images         [default: data/images/]
    --no-generate  Skip generation (dry run)
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

from safegrad.pipeline.hf_auth import resolve_hf_token

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    stream=sys.stderr,
)
log = logging.getLogger(__name__)

# Mapping from metadata generator_model shorthand → HuggingFace model ID
_MODEL_MAP: dict[str, str] = {
    "sdxl":   "stabilityai/stable-diffusion-xl-base-1.0",
    "flux1":  "black-forest-labs/FLUX.1-schnell",
    "kolors": "Kwai-Kolors/Kolors",
}

# Per-model generation defaults (steps and guidance scale).
# FLUX.1-schnell is a guidance-distilled model (4 steps, CFG=0); SDXL needs
# more steps; Kolors uses a SDXL-style architecture with moderate CFG.
_MODEL_GENERATION_DEFAULTS: dict[str, dict] = {
    "sdxl":   {"num_inference_steps": 40, "guidance_scale": 7.5},
    "flux1":  {"num_inference_steps":  4, "guidance_scale": 0.0},
    "kolors": {"num_inference_steps": 50, "guidance_scale": 5.0},
}

# Default T2I model distribution (commercial Apache-2.0 licensed models only):
#   SDXL 50%, FLUX.1-schnell 30%, Kolors 20%
PAPER_T2I_PROPORTIONS: list[tuple[str, float]] = [
    ("sdxl",   0.50),
    ("flux1",  0.30),
    ("kolors", 0.20),
]

# Default CLI value
_DEFAULT_T2I_MODELS_ARG: list[str] = [
    "sdxl:50", "flux1:30", "kolors:20",
]


def parse_t2i_weights(specs: list[str]) -> list[tuple[str, float]]:
    """Parse ``KEY[:WEIGHT]`` T2I model specifications into normalised (key, weight) pairs.

    Accepts shorthands (sdxl, flux1, kolors) or full HF model IDs.
    If no weights are given, all models share the quota equally.
    """
    parsed: list[tuple[str, float]] = []
    for spec in specs:
        if ":" in spec:
            head, tail = spec.rsplit(":", 1)
            try:
                w = float(tail)
                parsed.append((head.strip(), w))
                continue
            except ValueError:
                pass
        parsed.append((spec.strip(), 1.0))
    total = sum(w for _, w in parsed)
    if total <= 0:
        raise ValueError(f"T2I model weights must be positive; got {specs}")
    return [(k, w / total) for k, w in parsed]


def assign_generator_models(
    ladders: list[dict],
    model_weights: list[tuple[str, float]] | None = None,
) -> tuple[list[dict], int]:
    """Assign T2I generator models to ladders that have none.

    Assignment is deterministic (SHA-256 of ``ladder_id``) for reproducibility.

    Parameters
    ----------
    ladders:
        Input ladder records.
    model_weights:
        List of (model_key, normalised_weight) pairs. Defaults to the paper
        distribution (SDXL 55.8%, Z-Turbo 26.8%, FLUX.1 16.3%, SD3.5 1.1%).

    Returns
    -------
    (updated_ladders, n_assigned)
    """
    if model_weights is None:
        model_weights = PAPER_T2I_PROPORTIONS

    # Build weighted selection list for O(1) lookup
    weights: list[str] = []
    for key, frac in model_weights:
        weights.extend([key] * max(1, round(frac * 1000)))

    result: list[dict] = []
    n_assigned = 0
    for ladder in ladders:
        gm = ladder.get("generator_model", "")
        if gm and (gm in _MODEL_MAP or "/" in gm):  # valid key or full HF ID
            result.append(ladder)
        else:
            h = int(hashlib.sha256(ladder.get("ladder_id", "").encode()).hexdigest(), 16)
            model_key = weights[h % len(weights)]
            updated = dict(ladder)
            updated["generator_model"] = model_key
            result.append(updated)
            n_assigned += 1
    return result, n_assigned


# ---------------------------------------------------------------------------
# Pipeline cache for diffusers models
# ---------------------------------------------------------------------------

class PipelineCache:
    """Lazy-loading cache of local diffusers pipelines, one per HF model ID.

    All GPU work is serialised through a single asyncio.Semaphore to avoid
    OOM errors when multiple coroutines try to generate images concurrently.
    """

    def __init__(self) -> None:
        self._pipes: dict[str, object] = {}
        self._load_lock = asyncio.Lock()
        self._gpu_sem   = asyncio.Semaphore(1)

    @staticmethod
    def _clear_torch_cache() -> None:
        try:
            import torch
        except ImportError:
            return
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @classmethod
    def _unload_pipe_sync(cls, pipe: object) -> None:
        try:
            if hasattr(pipe, "to"):
                pipe.to("cpu")  # type: ignore[operator]
        except Exception:
            pass
        cls._clear_torch_cache()

    def _unload_all_except_sync(self, keep_model_id: str | None) -> None:
        stale_ids = [model_id for model_id in self._pipes if model_id != keep_model_id]
        for model_id in stale_ids:
            pipe = self._pipes.pop(model_id)
            log.info("Unloading diffusers pipeline: %s", model_id)
            self._unload_pipe_sync(pipe)

    async def _load_sync_in_thread(self, hf_model_id: str) -> object:
        if hf_model_id in self._pipes:
            return self._pipes[hf_model_id]
        async with self._load_lock:
            if hf_model_id in self._pipes:
                return self._pipes[hf_model_id]
            await asyncio.to_thread(self._unload_all_except_sync, hf_model_id)
            pipe = await asyncio.to_thread(self._load_sync, hf_model_id)
            self._pipes[hf_model_id] = pipe
            return pipe

    @staticmethod
    def _load_sync(hf_model_id: str) -> object:
        import torch
        from diffusers import (
            FluxPipeline,
            StableDiffusionPipeline,
            StableDiffusionXLPipeline,
            DiffusionPipeline,
        )

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        device = "cuda" if torch.cuda.is_available() else "cpu"

        cls_map = {
            "black-forest-labs/FLUX.1-schnell":                FluxPipeline,
            "stabilityai/stable-diffusion-xl-base-1.0":        StableDiffusionXLPipeline,
            "CompVis/stable-diffusion-v1-4":                   StableDiffusionPipeline,
        }
        pipeline_cls = cls_map.get(hf_model_id, DiffusionPipeline)
        log.info("Loading diffusers pipeline: %s (cls=%s) ...", hf_model_id, pipeline_cls.__name__)
        load_kwargs: dict = {"torch_dtype": dtype}
        token = resolve_hf_token()
        if token:
            load_kwargs["token"] = token
        if pipeline_cls is StableDiffusionPipeline:
            load_kwargs["safety_checker"] = None
            load_kwargs["requires_safety_checker"] = False
        pipe = pipeline_cls.from_pretrained(hf_model_id, **load_kwargs)
        if hasattr(pipe, "safety_checker"):
            pipe.safety_checker = None

        pipe = pipe.to(device)
        log.info("Pipeline ready: %s", hf_model_id)
        return pipe

    async def generate(
        self,
        model_key: str,
        prompt: str,
        seed: int,
        height: int = 1024,
        width: int = 1024,
    ) -> Image.Image:
        """Generate an image synchronously in a thread, serialised through GPU semaphore.

        Uses per-model defaults for inference steps and guidance scale to match
        each model's recommended settings (see ``_MODEL_GENERATION_DEFAULTS``).
        """
        hf_model_id = _MODEL_MAP.get(model_key, model_key)
        model_defaults = _MODEL_GENERATION_DEFAULTS.get(model_key, {})
        num_steps     = model_defaults.get("num_inference_steps", 30)
        guidance_scale = model_defaults.get("guidance_scale", 7.5)

        async with self._gpu_sem:
            pipe = await self._load_sync_in_thread(hf_model_id)

            for attr in ("safety_checker", "nsfw_classifier"):
                if hasattr(pipe, attr):
                    setattr(pipe, attr, None)

            def _run() -> Image.Image:
                import torch
                generator = torch.Generator().manual_seed(seed)
                kwargs: dict = dict(
                    prompt=prompt,
                    generator=generator,
                    height=height,
                    width=width,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                )
                result = pipe(**kwargs)  # type: ignore[operator]
                img = result.images[0]
                import numpy as np
                arr = np.array(img)
                if arr.mean() < 10 or arr.std() < 5:
                    log.warning("Generated image appears black/uniform — likely blocked")
                return img

            return await asyncio.to_thread(_run)

    async def close(self) -> None:
        async with self._gpu_sem:
            await asyncio.to_thread(self._unload_all_except_sync, None)


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

async def _generate_image_local(
    pipeline_cache: PipelineCache,
    model_key: str,
    prompt: str,
    seed: int,
    save_path: Path,
) -> Image.Image | None:
    """Generate an image locally, save it to save_path, and return it."""
    try:
        img = await pipeline_cache.generate(model_key, prompt, seed)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(save_path, format="PNG")
        log.debug("Generated image saved: %s", save_path)
        return img
    except Exception as exc:
        log.warning("Local T2I generation failed (%s, seed=%d): %s", model_key, seed, exc)
        return None


def _image_path(category: str, level: str, ladder_id: str) -> str:
    """Canonical relative path for any rung image (public-friendly format)."""
    return f"images/{category}/{ladder_id}_{level}.png"


def _synthetic_seed(ladder_id: str, level: str) -> int:
    """Deterministic seed derived from ladder_id + level."""
    digest = hashlib.sha256(f"{ladder_id}:{level}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


# ---------------------------------------------------------------------------
# Pre-generation pass
# ---------------------------------------------------------------------------

async def _pregenerate_all_images(
    ladders: list[dict],
    pipeline_cache: PipelineCache,
    image_dir: Path,
) -> None:
    """Generate images for all five rungs of every ladder, grouped by generator model.

    The same T2I model and seed are used for all five rungs within a ladder,
    isolating severity differences from generative variance.
    This includes the safe rung — all four rungs receive an image.

    Rungs are processed model-by-model so each T2I model loads exactly once.
    Rung ``image_path`` fields are updated in-place on success.
    """
    by_model: dict[str, list[tuple[dict, str, Path, str, int, str]]] = defaultdict(list)

    for ladder in ladders:
        category  = ladder.get("category", "unknown")
        top_gen   = ladder.get("generator_model")
        top_seed  = ladder.get("seed")
        ladder_id = ladder.get("ladder_id", "?")

        for level in ("safe", "low_risk", "mid_risk", "high_risk", "very_high_risk"):
            rung = ladder.get(f"rung_{level}", {})
            if not rung:
                continue

            # Skip rungs that already have an image on disk
            if rung.get("image_path"):
                continue

            gen_model = rung.get("generator_model") or top_gen
            if gen_model not in _MODEL_MAP:
                gen_model = "flux1"
            prompt = rung.get("prompt", "")
            if not prompt:
                continue

            # Paper: the same model and seed are used for all four rungs within a
            # ladder, isolating severity differences from generative variance.
            # Use the ladder-level seed for every rung; fall back to a deterministic
            # hash only when no seed is available at all.
            effective_seed = top_seed if top_seed is not None else _synthetic_seed(ladder_id, level)

            rel_path = _image_path(category, level, ladder_id)
            save_path = image_dir / category / f"{ladder_id}_{level}.png"

            # Already exists on disk — just wire up the path and skip generation
            if save_path.exists():
                rung["image_path"] = rel_path
                continue

            by_model[gen_model].append(
                (rung, prompt, save_path, rel_path, effective_seed, ladder_id)
            )

    if not by_model:
        log.info("Pre-generation: no synthetic images to generate.")
        return

    total = sum(len(v) for v in by_model.values())
    log.info(
        "Pre-generating %d synthetic image(s) across %d model(s): %s",
        total, len(by_model), ", ".join(sorted(by_model)),
    )

    for model_key, items in sorted(by_model.items()):
        log.info("  [%s] Generating %d image(s) ...", model_key, len(items))
        for rung, prompt, save_path, rel_path, effective_seed, ladder_id in items:
            gen_img = await _generate_image_local(pipeline_cache, model_key, prompt, effective_seed, save_path)
            if gen_img is not None:
                rung["image_path"] = rel_path
                log.debug("[%s] Pre-generated: %s", ladder_id, rel_path)
            else:
                log.warning("[%s] Pre-generation failed for %s", ladder_id, rel_path)

    log.info("Pre-generation complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input",      default="data/stage2_out.jsonl",
                   help="Source JSONL file (ladder format)")
    p.add_argument("--output",     default="data/stage3_out.jsonl",
                   help="Output JSONL file with image paths")
    p.add_argument("--image-dir",  default="data/images/",
                   help="Root directory to save generated images")
    p.add_argument("--no-generate", action="store_true",
                   help="Skip image generation (dry run — write JSONL as-is)")
    p.add_argument(
        "--t2i-models", nargs="+", default=_DEFAULT_T2I_MODELS_ARG,
        metavar="MODEL[:WEIGHT]",
        help=(
            "T2I models to assign to ladders that have no generator_model. "
            "Each entry is a model key (sdxl|flux1|kolors) or a full "
            "HuggingFace model ID, with an optional :WEIGHT suffix. "
            "Examples:\n"
            "  --t2i-models sdxl                     (single model, all ladders)\n"
            "  --t2i-models sdxl flux1               (two models, equal weight)\n"
            "  --t2i-models sdxl:60 flux1:40         (custom weights)\n"
            "Ladders that already have a valid generator_model keep their value. "
            "Use 'none' to disable auto-assignment entirely. "
            "Default: sdxl:50 flux1:30 kolors:20 (commercial Apache-2.0 models only)."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)
    image_dir   = Path(args.image_dir)

    log.info("Stage 3 — T2I Image Synthesis")
    log.info("Loading %s ...", input_path)
    ladders: list[dict] = []
    with input_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                ladders.append(json.loads(line))
    log.info("Loaded %d ladders.", len(ladders))

    # Assign T2I generator models to ladders that lack one
    if args.t2i_models and args.t2i_models != ["none"]:
        try:
            model_weights = parse_t2i_weights(args.t2i_models)
        except ValueError as exc:
            log.error("Invalid --t2i-models specification: %s", exc)
            sys.exit(1)
        ladders, n_assigned = assign_generator_models(ladders, model_weights=model_weights)
        if n_assigned:
            weight_str = ", ".join(
                f"{k} {w*100:.1f}%" for k, w in model_weights
            )
            log.info(
                "Assigned generator_model to %d/%d ladders: %s",
                n_assigned, len(ladders), weight_str,
            )

    if not args.no_generate:
        image_dir.mkdir(parents=True, exist_ok=True)
        pipeline_cache = PipelineCache()
        log.info("=" * 60)
        log.info("T2I generation enabled — image_dir: %s", image_dir)
        log.info("=" * 60)
        asyncio.run(_pregenerate_all_images(ladders, pipeline_cache, image_dir))
        asyncio.run(pipeline_cache.close())
        log.info("T2I pipeline closed.")
    else:
        log.info("--no-generate: skipping image generation, writing ladders as-is.")

    log.info("Writing %s ...", output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for ladder in ladders:
            f.write(json.dumps(ladder, ensure_ascii=False) + "\n")
    log.info("Done. Wrote %d ladders.", len(ladders))


if __name__ == "__main__":
    main()
