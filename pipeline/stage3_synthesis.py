"""Stage 3: T2I Image Synthesis (Section 3.3).

The third stage of the Automated Severity Ladder (ASL) pipeline.

For each rung that lacks an image, generates one using the T2I model
specified in the ladder's ``generator_model`` field.  Supports all four
T2I models evaluated in the paper:

  - ``sdxl``   : Stable Diffusion XL (stabilityai/stable-diffusion-xl-base-1.0)
  - ``zimage`` : Z-Turbo (Lykon/dreamshaper-xl-turbo / Z-Turbo adapter)
  - ``flux1``  : FLUX.1-schnell (black-forest-labs/FLUX.1-schnell)
  - ``large``  : Stable Diffusion 3.5 Large (stabilityai/stable-diffusion-3.5-large)

Paper reference: Section 3, Stage 3 ("T2I Image Synthesis")
Models evaluated: SDXL, Z-Turbo, FLUX.1-schnell, SD 3.5 Large (Table 3)

Note: Image generation requires a GPU and ~10–20 GB VRAM per model.
This stage wraps pipeline/phase4_filter.py's image-generation path.
The VLM verification (monotonicity scoring) runs in Stage 4.

Usage
-----
    uv run python -m safegrad.pipeline.stage3_synthesis [OPTIONS]

Options
-------
    --input          Source JSONL (ladder format)     [default: metadata_stage2b.jsonl]
    --output         Output JSONL                     [default: metadata_stage3.jsonl]
    --image-root     Root dir to resolve image paths  [default: .]
    --project-root   Root dir for saving new images   [default: .]
    --rules          Path to rules JSONL              [default: data/rules.jsonl]
    --vlm-model      Vision LLM for scoring           [default: Qwen/Qwen3-VL-8B-Thinking]
    --min-score-gap  Monotonicity gap threshold       [default: 0.4]
    --batch-size     VLM batch size                   [default: 4]
    --no-generate    Skip image generation (audit only)
"""

import sys

# Delegate to phase4_filter which handles both generation and VLM audit.
# Stage 4 (verification-only path) re-uses the same code with --no-generate.
import pipeline.phase4_filter as _phase4


def main() -> None:
    _phase4.main()


if __name__ == "__main__":
    _phase4.main()
