"""Stage 3: T2I Image Synthesis (Section 3, Stage 3).

The third stage of the Automated Severity Ladder (ASL) pipeline.

For each rung that lacks an image, generates one using the T2I model
specified in the ladder's ``generator_model`` field.  Supports all four
T2I models evaluated in the paper:

  - ``sdxl``   : Stable Diffusion XL (stabilityai/stable-diffusion-xl-base-1.0)
  - ``zimage`` : Z-Turbo (Tongyi-MAI/Z-Image-Turbo)
  - ``flux1``  : FLUX.1-dev (black-forest-labs/FLUX.1-dev)
  - ``large``  : Stable Diffusion 3.5 Large (stabilityai/stable-diffusion-3.5-large)

Paper reference: Section 3, Stage 3 ("T2I Image Synthesis")
Models evaluated: SDXL, Z-Turbo, FLUX.1, SD 3.5 Large (Table 3)

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
    "flux1":  "black-forest-labs/FLUX.1-dev",
    "large":  "stabilityai/stable-diffusion-3.5-large",
    "zimage": "Tongyi-MAI/Z-Image-Turbo",
}

# Paper T2I model distribution (Section 4, Stage 3):
#   SDXL 55.8%, Z-Turbo 26.8%, FLUX.1-dev 16.3%, SD3.5-Large 1.1%
PAPER_T2I_PROPORTIONS: list[tuple[str, float]] = [
    ("sdxl",   0.558),
    ("zimage", 0.268),
    ("flux1",  0.163),
    ("large",  0.011),
]

# Default CLI value that reproduces the paper distribution
_DEFAULT_T2I_MODELS_ARG: list[str] = [
    "sdxl:55.8", "zimage:26.8", "flux1:16.3", "large:1.1",
]


def parse_t2i_weights(specs: list[str]) -> list[tuple[str, float]]:
    """Parse ``KEY[:WEIGHT]`` T2I model specifications into normalised (key, weight) pairs.

    Accepts shorthands (sdxl, flux1, large, zimage) or full HF model IDs.
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
            StableDiffusion3Pipeline,
            StableDiffusionPipeline,
            StableDiffusionXLPipeline,
            DiffusionPipeline,
        )

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        device = "cuda" if torch.cuda.is_available() else "cpu"

        cls_map = {
            "black-forest-labs/FLUX.1-dev":                    FluxPipeline,
            "stabilityai/stable-diffusion-3.5-large":          StableDiffusion3Pipeline,
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
        """Generate an image synchronously in a thread, serialised through GPU semaphore."""
        hf_model_id = _MODEL_MAP.get(model_key, model_key)
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
                    num_inference_steps=20,
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
    """Pre-generate all missing synthetic images, grouped by generator model.

    By processing each model's work items in one batch before moving to the next,
    every T2I model is loaded exactly once — eliminating the repeated load/unload
    cycles that occur when concurrent ladder tasks interleave different models.

    Rung ``image_path`` fields are updated in-place on success.
    """
    by_model: dict[str, list[tuple[dict, str, Path, str, int, str]]] = defaultdict(list)

    for ladder in ladders:
        category  = ladder.get("category", "unknown")
        top_gen   = ladder.get("generator_model")
        top_seed  = ladder.get("seed")
        ladder_id = ladder.get("ladder_id", "?")

        for level in ("safe", "low_risk", "mid_risk", "high_risk"):
            rung = ladder.get(f"rung_{level}", {})
            if not rung:
                continue

            # Skip rungs that already have an image
            if rung.get("image_path"):
                continue

            # Only generate for synthetic rungs (non-safe)
            if not rung.get("synthetic", False) or level == "safe":
                continue

            gen_model = rung.get("generator_model") or top_gen
            if gen_model not in _MODEL_MAP:
                gen_model = "flux1"
            prompt = rung.get("prompt", "")
            if not prompt:
                continue

            rung_seed = rung.get("seed")
            seed = rung_seed if rung_seed is not None else top_seed
            effective_seed = seed if seed is not None else _synthetic_seed(ladder_id, level)

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
            "Each entry is a model key (sdxl|flux1|large|zimage) or a full "
            "HuggingFace model ID, with an optional :WEIGHT suffix. "
            "Examples:\n"
            "  --t2i-models sdxl                     (single model, all ladders)\n"
            "  --t2i-models sdxl flux1               (two models, equal weight)\n"
            "  --t2i-models sdxl:56 flux1:44         (custom weights)\n"
            "Ladders that already have a valid generator_model keep their value. "
            "Use 'none' to disable auto-assignment entirely. "
            "Default: paper distribution "
            "(sdxl:55.8 zimage:26.8 flux1:16.3 large:1.1)."
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
