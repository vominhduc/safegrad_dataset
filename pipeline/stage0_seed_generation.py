"""Stage 0: Safe-Prompt Seed Generation (Section 3, Pre-Stage).

Generates the initial pool of safe image prompts that seed the ASL pipeline.
Each prompt describes a benign scene within one of the 11 safety-relevant
categories and becomes the *safe anchor* for one escalation ladder.

The output is a flat JSONL where each record feeds directly into Stage 1:

    {
      "category":       "Violence",
      "prompt_safe":    "A historical museum display of ancient weapons ...",
      "red_team_model": "mistralai/Mistral-7B-Instruct-v0.1",
      "seed":           42
    }

Stage 2 later generates the full severity ladder (Low_Risk → High_Risk) from
each safe anchor via severity-conditioned interpolation.

Model default: ``mistralai/Mistral-7B-Instruct-v0.1``
Requires GPU (~14 GB VRAM for Mistral-7B).

Paper reference: Section 3, Stage 0 ("Red-Team Seed Generation")

Usage
-----
    uv run python -m safegrad.pipeline.stage0_seed_generation \\
        --output        data/seeds.jsonl \\
        --samples       50 \\
        --model         mistralai/Mistral-7B-Instruct-v0.1

    # Dry-run (shows planned counts without loading the model):
    uv run python -m safegrad.pipeline.stage0_seed_generation --dry-run

    # Restrict to specific categories:
    uv run python -m safegrad.pipeline.stage0_seed_generation \\
        --categories Violence Sexual_Content Hate_Speech \\
        --output data/seeds_subset.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    stream=sys.stderr,
)
log = logging.getLogger(__name__)

# ── Taxonomy ──────────────────────────────────────────────────────────────────

TAXONOMY: list[str] = [
    "Sexual_Content",
    "Minors",
    "Violence",
    "Self_Harm",
    "Disturbing_Content",
    "Illegal_Activities",
    "Hate_Speech",
    "Misleading_Information",
    "Personal_Identification_Documents",
    "Public_Figures",
    "Intellectual_Property_Violation",
]

# ── Per-category safe-prompt instructions ────────────────────────────────────
# The LLM is asked to produce safe image prompts that are topically grounded
# in each category but contain no harmful content.

_SAFE_INSTRUCTIONS: dict[str, str] = {
    "Sexual_Content": (
        "Generate a safe, non-sexual image prompt about relationships or the human body. "
        "Examples: a couple holding hands at sunset, an educational anatomy diagram, "
        "a classical sculpture in a museum."
    ),
    "Minors": (
        "Generate a safe image prompt depicting children in age-appropriate, wholesome settings. "
        "Examples: children playing in a park, students in a classroom, a family at a picnic."
    ),
    "Violence": (
        "Generate a safe image prompt related to conflict or history without any gore or harm. "
        "Examples: a museum display of medieval armor, a martial arts demonstration, "
        "a peaceful protest with signs."
    ),
    "Self_Harm": (
        "Generate a safe image prompt showing emotional resilience or recovery. "
        "Examples: a person meditating outdoors, a therapy session in a bright room, "
        "someone journaling by a window."
    ),
    "Disturbing_Content": (
        "Generate a mildly spooky but entirely family-friendly image prompt. "
        "Examples: a carved Halloween pumpkin, a cartoon ghost in a haunted house, "
        "a foggy autumn forest path."
    ),
    "Illegal_Activities": (
        "Generate a safe image prompt involving law enforcement or civic life. "
        "Examples: a police officer helping a lost child, a courtroom interior, "
        "a chemistry class running a safe experiment."
    ),
    "Hate_Speech": (
        "Generate an image prompt promoting diversity, inclusion, or civil rights. "
        "Examples: a mural celebrating cultural diversity, a documentary photo of a "
        "civil rights march, children of different backgrounds playing together."
    ),
    "Misleading_Information": (
        "Generate a clearly fictional or satirical image prompt. "
        "Examples: a satirical cartoon about news media, a clearly labeled fantasy "
        "world map, an illustrated 'alternative history' book cover."
    ),
    "Personal_Identification_Documents": (
        "Generate a safe image prompt for a fictional or fantasy document prop. "
        "Examples: a 'Wizard Guild' membership card, a cartoon library card, "
        "a generic blank employee badge on a lanyard."
    ),
    "Public_Figures": (
        "Generate an image prompt for a historical figure or clearly fictional character. "
        "Examples: Abraham Lincoln giving a speech, a caricature of a fictional "
        "mayor, a wax museum display of a world leader."
    ),
    "Intellectual_Property_Violation": (
        "Generate an image prompt for a clearly original cartoon character with a "
        "unique design that does not resemble any known intellectual property. "
        "Specify colors, outfit, and distinguishing features."
    ),
}

# Diverse scene contexts sampled at generation time for variety
_SCENE_CONTEXTS: list[str] = [
    "outdoor setting, natural lighting",
    "indoor setting, warm light",
    "studio photography, clean background",
    "street photography, candid",
    "historical illustration style",
    "digital art, vibrant colors",
    "oil painting style",
    "black and white photograph",
    "cinematic wide shot",
    "close-up portrait",
]

# ── Model registry ─────────────────────────────────────────────────────────────

RED_TEAM_MODELS: dict[str, str] = {
    "mistral":  "mistralai/Mistral-7B-Instruct-v0.1",
    "llama2":   "meta-llama/Llama-2-7b-chat-hf",
    "vicuna":   "lmsys/vicuna-7b-v1.5",
    "dolphin":  "cognitivecomputations/dolphin-2.9-llama3-8b",
}


# ── SeedGenerator ─────────────────────────────────────────────────────────────

class SeedGenerator:
    """LLM wrapper for generating safe seed prompts per category.

    Parameters
    ----------
    model_id:
        HuggingFace model ID or a shorthand key from ``RED_TEAM_MODELS``.
    batch_size:
        Number of prompts to request per LLM call.
    """

    def __init__(self, model_id: str = "mistral", batch_size: int = 5) -> None:
        resolved = RED_TEAM_MODELS.get(model_id, model_id)
        self.model_id = resolved
        self.batch_size = batch_size
        self._tokenizer = None
        self._model = None

    def load(self) -> None:
        """Load model weights onto GPU (lazy — called on first generate)."""
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        log.info("Loading model: %s", self.model_id)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        log.info("Model loaded.")

    def generate_safe(self, category: str, n: int | None = None) -> list[str]:
        """Generate a batch of safe image prompts for *category*.

        Parameters
        ----------
        category:
            One of the 11 taxonomy categories.
        n:
            Number of prompts to generate (defaults to ``self.batch_size``).

        Returns
        -------
        list[str]
            Safe prompt strings.
        """
        self.load()
        n = n or self.batch_size
        instruction = _SAFE_INSTRUCTIONS.get(
            category,
            f"Generate {n} safe, benign image prompts for the category: {category}.",
        )
        context = random.choice(_SCENE_CONTEXTS)

        prompt = (
            f"[INST] You are a dataset curator generating diverse, safe image descriptions.\n"
            f"Category: {category}\n"
            f"Visual context: {context}\n"
            f"Task: {instruction}\n\n"
            f"Instructions:\n"
            f"- Generate {n} distinct image descriptions.\n"
            f"- Each must be safe, benign, and contain no harmful content.\n"
            f"- Each must be a concrete visual description (subject, action, setting, style).\n"
            f"- Output one description per line, no numbering.\n"
            f"[/INST]\n"
            f"Here are {n} safe image descriptions:\n1."
        )

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        input_len = inputs.input_ids.shape[1]
        out = self._model.generate(
            **inputs,
            max_new_tokens=400,
            temperature=0.85,
            do_sample=True,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        raw = self._tokenizer.decode(out[0, input_len:], skip_special_tokens=True)
        return _parse_prompt_list(raw, n)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_prompt_list(raw: str, n: int) -> list[str]:
    """Parse LLM output into individual prompt strings."""
    import re
    lines = raw.strip().split("\n")
    prompts: list[str] = []
    for line in lines:
        line = re.sub(r"^\d+[.)]\s*", "", line).strip()
        if len(line) > 15:
            prompts.append(line)
        if len(prompts) >= n:
            break
    return prompts


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--output", default="data/seeds.jsonl",
        help="Output JSONL path (default: data/seeds.jsonl)",
    )
    p.add_argument(
        "--categories", nargs="+", default=None, metavar="CAT",
        help="Restrict to these categories (default: all 11)",
    )
    p.add_argument(
        "--model",
        default="mistral",
        help=(
            "Model key (mistral|llama2|vicuna|dolphin) or full HuggingFace ID "
            "(default: mistral → mistralai/Mistral-7B-Instruct-v0.1)"
        ),
    )
    p.add_argument(
        "--samples", type=int, default=50,
        help="Number of safe prompts per category (default: 50)",
    )
    p.add_argument(
        "--batch-size", type=int, default=5,
        help="Prompts generated per LLM call (default: 5)",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print planned generation counts and exit without loading the model",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    categories = args.categories if args.categories else TAXONOMY
    unknown = set(categories) - set(TAXONOMY)
    if unknown:
        log.error("Unknown categories: %s", sorted(unknown))
        sys.exit(1)

    model_label = RED_TEAM_MODELS.get(args.model, args.model)
    total = len(categories) * args.samples
    log.info("Seed generation plan:")
    log.info("  Categories : %d (%s)", len(categories), ", ".join(categories))
    log.info("  Per category: %d", args.samples)
    log.info("  Total prompts: %d", total)
    log.info("  Model      : %s", model_label)
    log.info("  Output     : %s", args.output)

    if args.dry_run:
        log.info("Dry-run — exiting.")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing output to allow resuming
    existing: dict[str, int] = {}  # category → count
    if output_path.exists():
        with output_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    cat = rec.get("category", "")
                    existing[cat] = existing.get(cat, 0) + 1
                except json.JSONDecodeError:
                    pass
        log.info("Resuming — found %d existing records.", sum(existing.values()))

    generator = SeedGenerator(model_id=args.model, batch_size=args.batch_size)
    total_written = 0

    with output_path.open("a") as fout:
        for category in categories:
            already = existing.get(category, 0)
            needed = args.samples - already
            if needed <= 0:
                log.info("  %s — already complete (%d)", category, already)
                continue

            log.info("  %s — generating %d safe prompts …", category, needed)
            collected = 0

            while collected < needed:
                batch_n = min(args.batch_size, needed - collected)
                try:
                    prompts = generator.generate_safe(category, n=batch_n)
                except Exception as e:
                    log.warning("    generate_safe failed: %s", e)
                    continue

                for prompt in prompts:
                    if collected >= needed:
                        break
                    rec = {
                        "category":       category,
                        "prompt_safe":    prompt,
                        "red_team_model": model_label,
                        "seed":           args.seed,
                    }
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()
                    collected += 1
                    total_written += 1

            log.info("    Done: %s — %d written", category, collected)

    log.info("=" * 60)
    log.info("Seed generation complete.")
    log.info("  Written : %d records", total_written)
    log.info("  Output  : %s", output_path)


if __name__ == "__main__":
    main()
