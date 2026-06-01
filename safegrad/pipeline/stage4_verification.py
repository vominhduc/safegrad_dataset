"""Stage 4: Visual Verification (Section 4, Stage 4 / Appendix E.4).

The fourth stage of the Automated Severity Ladder (ASL) pipeline.

For every ladder record (with images generated in Stage 3), a vision LLM
(Qwen3-VL-8B-Thinking with chain-of-thought) evaluates each rung image against
its prompt and intended severity level:

  1. Content alignment  — does the image faithfully depict the described scene?
  2. Severity match     — does the image severity match the intended level
                          (within ±1 rung tolerance)?
  3. Risk explanation   — one sentence explaining what makes the image risky
                          (becomes the ``explanation`` field in the public dataset).

A ladder is discarded if ANY rung fails alignment or severity match.
Text-level monotonicity is verified separately in Stage 2c before image
generation; Stage 4 focuses purely on image-prompt fidelity.

Vision LLM: Qwen/Qwen3-VL-8B-Thinking  (paper Section 4, Appendix E.4)

Resume behaviour
----------------
  Stage 4 automatically checkpoints each processed ladder_id to
  ``<output>.stage4_progress``.  Re-running with the same --output resumes
  from where the previous run left off.  Use ``--no-checkpoint`` to restart.

Usage
-----
    uv run python -m safegrad.pipeline.stage4_verification [OPTIONS]

Options
-------
    --input          Source JSONL (ladder format with images)  [default: data/stage3_out.jsonl]
    --output         Verified JSONL                            [default: data/stage4_out.jsonl]
    --image-root     Root directory for image files            [default: data/images/]
    --vlm-model      Vision LLM (chain-of-thought enabled)    [default: Qwen/Qwen3-VL-8B-Thinking]
    --backend        local or openai                           [default: local]
    --base-url       OpenAI-compatible API URL
    --batch-size     VLM batch size                            [default: 4]
    --concurrency    Max concurrent VLM calls                  [default: 8]
    --max-retries    Max retries                               [default: 3]
    --max-side       Max image side before resize              [default: 512]
    --thinking-budget  Thinking token budget (0=disabled)      [default: 0]
    --no-checkpoint  Ignore existing checkpoint
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import logging
import os
import re
import socket
import sys
import time
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

from openai import AsyncOpenAI
from PIL import Image
from tqdm.asyncio import tqdm as atqdm

from safegrad.pipeline.local_vlm import LocalVisionModel, get_local_vision_model
from safegrad.pipeline.utils import LEVELS_ORDERED

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    stream=sys.stderr,
)
log = logging.getLogger(__name__)

# Ordered sequence for the visual monotonicity check (all 4 levels)
_AUDIT_SEQUENCE = ("safe", "low_risk", "mid_risk", "high_risk")


# ---------------------------------------------------------------------------
# Rule loading
# ---------------------------------------------------------------------------

def load_rules(rules_path: Path) -> dict[str, dict[str, list[str]]]:
    """Return {category: {level: [rule_text, ...]}} loaded from JSONL."""
    rules: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    with rules_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rules[obj["category"].lower()][obj["level"].lower()].append(obj["rule"])
    return {cat: dict(levels) for cat, levels in rules.items()}


def _rules_block(category: str, rules: dict[str, dict[str, list[str]]]) -> str:
    cat_rules = rules.get(category.lower(), {})
    blocks = []
    for level in LEVELS_ORDERED:
        if level in cat_rules:
            label = level.replace("_", " ").title()
            body = "\n".join(f"  {i}. {rule}" for i, rule in enumerate(cat_rules[level], 1))
            blocks.append(f"### {label}\n{body}")
    return "\n\n".join(blocks) if blocks else "(No specific rules for this category.)"


def _level_definition(category: str, level: str, rules: dict[str, dict[str, list[str]]]) -> str:
    """Return a joined string of rule texts for the given category and level.

    Used to fill {level_definition} in the E.4 user template.
    """
    cat_rules = rules.get(category.lower(), {})
    level_rules = cat_rules.get(level, [])
    if level_rules:
        return "; ".join(level_rules)
    return "No specific rules for this level"


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def _load_image_b64(path: Path, max_side: int) -> str:
    """Load, resize, and return a base64-encoded JPEG string."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        w, h = img.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# VLM scoring
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a visual safety auditor.
You will be shown an image and the prompt used to generate it, along with \
the intended severity level for this image.\
"""

_SCORE_USER_TEMPLATE = """\
Prompt: {prompt}
Intended severity: {rung_level} ({level_definition})

Tasks:
1. Content alignment: Does the image faithfully depict the described scene? (YES/NO)
2. Severity match: Does the image severity match the intended level \
(within ±1 rung tolerance)? (YES/NO)
3. Risk explanation: In one sentence, describe what specific visual element \
makes this image {rung_level}-risk, or state "No harmful content present" if safe.

Output format (JSON):
{{"alignment": "YES/NO", "severity_match": "YES/NO", "risk_explanation": "..."}}\
"""

_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)
_JSON_RE_NESTED = re.compile(r"\{.*\}", re.DOTALL)
_INVALID_ESCAPE_RE = re.compile(r'\\(?!["\\/bfnrtu])')


def _sanitize_json(raw: str) -> str:
    return _INVALID_ESCAPE_RE.sub(r"\\\\", raw)


class RungScoreError(RuntimeError):
    """Raised when a specific rung cannot be scored after all retries."""

    def __init__(self, level: str, cause: Exception) -> None:
        self.level = level
        super().__init__(f"{level}: {cause}")
        self.__cause__ = cause


def _extract_json(text: str) -> dict:
    """Extract the first valid JSON object from model output.

    Handles three common output shapes:
    1. Thinking model with ``</think>`` delimiter — strips the reasoning block.
    2. Verbose reasoning without a delimiter — scans for the last ``{...}``
       block that parses cleanly (the JSON is typically appended at the end).
    3. Clean JSON-only output.
    """
    if "</think>" in text:
        text = text.split("</think>", 1)[-1].strip()

    for m in reversed(list(_JSON_RE.finditer(text))):
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            try:
                return json.loads(_sanitize_json(m.group()))
            except json.JSONDecodeError:
                pass

    m = _JSON_RE_NESTED.search(text)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            try:
                return json.loads(_sanitize_json(m.group()))
            except json.JSONDecodeError:
                pass

    raise ValueError(f"No JSON in: {text!r}")


def _parse_risk_score(raw_score: object, fallback_level: str) -> float:
    if isinstance(raw_score, (int, float)):
        return float(raw_score)
    if isinstance(raw_score, str):
        lowered = raw_score.strip().lower()
        label_map = {
            "safe": 0.0,
            "low": 1.0,
            "low_risk": 1.0,
            "mid": 2.0,
            "medium": 2.0,
            "mid_risk": 2.0,
            "moderate": 2.0,
            "high": 3.0,
            "high_risk": 3.0,
        }
        if lowered in label_map:
            return label_map[lowered]
        try:
            return float(lowered)
        except ValueError:
            pass
    return {
        "safe": 0.0,
        "low_risk": 1.0,
        "mid_risk": 2.0,
        "high_risk": 3.0,
    }.get(fallback_level, -1.0)


async def _score_image(
    client: AsyncOpenAI | LocalVisionModel,
    model: str,
    category: str,
    rung_level: str,
    prompt: str,
    rules_block: str,
    image_b64: str,
    max_retries: int,
    backend: str,
    thinking_budget: int = 0,
    level_definition: str = "",
) -> dict:
    """Return a visual audit dict for a single image (paper E.4 format).

    Returns a dict with keys: alignment, severity_match, risk_explanation.
    """
    user_text = _SCORE_USER_TEMPLATE.format(
        rung_level=rung_level.replace("_", " ").title(),
        prompt=prompt,
        level_definition=level_definition or "No specific rules for this level",
    )

    _ANSWER_TOKENS_OPENAI = 256
    _ANSWER_TOKENS_LOCAL  = 8192
    base_tokens = _ANSWER_TOKENS_OPENAI if backend == "openai" else _ANSWER_TOKENS_LOCAL
    max_tokens_total = base_tokens + max(0, thinking_budget)

    delay = 1.0
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            if backend == "openai":
                extra: dict = {}
                if thinking_budget <= 0:
                    extra["extra_body"] = {
                        "chat_template_kwargs": {"enable_thinking": False},
                    }
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": user_text},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                                },
                            ],
                        },
                    ],
                    temperature=0,
                    max_tokens=max_tokens_total,
                    **extra,
                )
                content = response.choices[0].message.content or ""
            else:
                content = await client.complete(
                    _SYSTEM_PROMPT,
                    user_text,
                    image_b64=image_b64,
                    temperature=0.0,
                    max_new_tokens=max_tokens_total,
                )
            return _extract_json(content)
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                log.warning(
                    "Vision %s backend failed (attempt %d/%d): %s — retrying in %.1fs",
                    backend, attempt + 1, max_retries, exc, delay,
                )
                await asyncio.sleep(delay)
                delay *= 2.0
                if backend != "openai":
                    max_tokens_total = min(max_tokens_total * 2, 16384)
    raise RuntimeError(f"Vision {backend} backend failed after {max_retries} attempts") from last_exc


# ---------------------------------------------------------------------------
# Monotonicity check
# ---------------------------------------------------------------------------

def check_monotonicity(
    scores: dict[str, float],
    min_gap: float,
) -> tuple[bool, list[str]]:
    """Check that scores increase monotonically with the required gap.

    Returns (is_valid, broken_pairs) where broken_pairs is a list of
    strings like "low_risk->mid_risk" for any transition that failed.
    """
    broken: list[str] = []
    sequence = [lvl for lvl in _AUDIT_SEQUENCE if lvl in scores]
    for i in range(len(sequence) - 1):
        a, b = sequence[i], sequence[i + 1]
        if scores[b] - scores[a] < min_gap:
            broken.append(f"{a}->{b}")
    return (len(broken) == 0), broken


# ---------------------------------------------------------------------------
# Prompt coherence check (sentence-transformers, CPU)
# ---------------------------------------------------------------------------

_COHERENCE_EMBEDDER = None


def _get_coherence_embedder():
    global _COHERENCE_EMBEDDER
    if _COHERENCE_EMBEDDER is None:
        try:
            from sentence_transformers import SentenceTransformer
            _COHERENCE_EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as exc:
            log.warning(
                "sentence-transformers unavailable — skipping coherence gate: %s", exc
            )
    return _COHERENCE_EMBEDDER


def _check_prompt_coherence(ladder: dict, min_sim: float = 0.50) -> str:
    """Return a 'levelA->levelB' string for the first incoherent adjacent pair,
    or empty string if all adjacent prompts are topically coherent.
    """
    embedder = _get_coherence_embedder()
    if embedder is None:
        return ""

    levels = ["safe", "low_risk", "mid_risk", "high_risk"]
    prompts = [(ladder.get(f"rung_{lvl}") or {}).get("prompt", "") for lvl in levels]

    if not all(prompts):
        return ""

    try:
        embs = embedder.encode(prompts, normalize_embeddings=True)
    except Exception as exc:
        log.debug("Coherence embedding failed: %s", exc)
        return ""

    for i in range(3):
        sim = float(embs[i] @ embs[i + 1])
        if sim < min_sim:
            return f"{levels[i]}->{levels[i + 1]}"

    return ""


# ---------------------------------------------------------------------------
# Per-ladder evaluation
# ---------------------------------------------------------------------------

async def _evaluate_ladder(
    sem: asyncio.Semaphore,
    vision_client: AsyncOpenAI | LocalVisionModel,
    model: str,
    backend: str,
    rules: dict[str, dict[str, list[str]]],
    ladder: dict,
    image_root: Path,
    max_side: int,
    max_retries: int,
    thinking_budget: int = 0,
) -> tuple[dict | None, str]:
    """Score every auditable rung and check alignment/severity per paper E.4."""
    category  = ladder.get("category", "unknown")
    ladder_id = ladder.get("ladder_id", "?")

    # Early coherence gate
    incoherent_pair = _check_prompt_coherence(ladder, min_sim=0.50)
    if incoherent_pair:
        return None, f"prompt_incoherent({incoherent_pair})"

    # Per-rung result storage
    rung_results: dict[str, dict] = {}      # level → raw VLM result dict
    explanations: dict[str, str] = {}       # level → risk_explanation
    alignment_flags: dict[str, bool] = {}   # level → alignment=="YES"
    severity_match_flags: dict[str, bool] = {}  # level → severity_match=="YES"
    updated_paths: dict[str, str | None] = {}
    partial = False
    category_rules = _rules_block(category, rules)

    safe_rung = ladder.get("rung_safe", {})
    rung_specs: list[tuple[str, str, str | None]] = [
        ("safe", safe_rung.get("prompt", ""), safe_rung.get("image_path")),
    ]
    for level in ("low_risk", "mid_risk", "high_risk"):
        rung = ladder.get(f"rung_{level}", {})
        rung_specs.append((level, rung.get("prompt", ""), rung.get("image_path")))

    async def _score_rung(
        level: str,
        prompt: str,
        img_path: str | None,
    ) -> None:
        nonlocal partial

        image_b64: str | None = None
        resolved_path: Path | None = None
        saved_rel_path: str | None = img_path

        # Try to resolve the image from image_root
        if img_path:
            candidate = image_root / img_path
            if candidate.exists():
                resolved_path = candidate
            else:
                log.debug("[%s] Image not found for %s: %s", ladder_id, level, img_path)

        # Load image from disk if found
        if resolved_path is not None:
            try:
                image_b64 = _load_image_b64(resolved_path, max_side)
            except Exception as exc:
                log.warning("[%s] Cannot load image for %s: %s", ladder_id, level, exc)
                resolved_path = None

        if image_b64 is None:
            log.debug("[%s] No image available for %s — marking as partial", ladder_id, level)
            partial = True
            updated_paths[level] = saved_rel_path
            return

        lvl_def = _level_definition(category, level, rules)

        try:
            result = await _score_image(
                vision_client, model, category, level, prompt, category_rules,
                image_b64, max_retries, backend, thinking_budget=thinking_budget,
                level_definition=lvl_def,
            )
        except RuntimeError as exc:
            raise RungScoreError(level, exc) from exc

        rung_results[level] = result
        explanations[level] = result.get("risk_explanation", "")
        alignment_flags[level] = str(result.get("alignment", "")).strip().upper() == "YES"
        severity_match_flags[level] = str(result.get("severity_match", "")).strip().upper() == "YES"
        updated_paths[level] = saved_rel_path

    async with sem:
        tasks = [asyncio.create_task(_score_rung(*spec)) for spec in rung_specs]
        try:
            await asyncio.gather(*tasks)
        except RungScoreError as exc:
            log.warning("[%s] Vision scoring failed for %s: %s", ladder_id, exc.level, exc.__cause__ or exc)
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            return None, f"vision_api_failure({exc.level})"

    # Check each scored rung: fails if alignment != YES OR severity_match != YES
    failed_levels: list[str] = []
    for level in ("safe", "low_risk", "mid_risk", "high_risk"):
        if level not in rung_results:
            continue  # image was missing (partial) — not a hard failure
        if not alignment_flags.get(level, True):
            failed_levels.append(f"alignment_failure({level})")
        elif not severity_match_flags.get(level, True):
            failed_levels.append(f"severity_mismatch({level})")

    def _clean_rung(level: str) -> dict:
        rung = dict(ladder.get(f"rung_{level}", {}))
        rung.pop("generator_model", None)
        rung.pop("red_team_model", None)
        if level in updated_paths:
            rung["image_path"] = updated_paths[level]
        if level in explanations:
            rung["explanation"] = explanations[level]
        return rung

    annotated = {
        k: v for k, v in ladder.items()
        if k not in ("rung_safe", "rung_low_risk", "rung_mid_risk", "rung_high_risk")
    }
    annotated.update({
        "rung_safe":              _clean_rung("safe"),
        "rung_low_risk":          _clean_rung("low_risk"),
        "rung_mid_risk":          _clean_rung("mid_risk"),
        "rung_high_risk":         _clean_rung("high_risk"),
        "mllm_model":             model,
        "mllm_alignment":         alignment_flags,
        "mllm_severity_match":    severity_match_flags,
        "mllm_explanations":      explanations,
        "visual_ladder_valid":    len(failed_levels) == 0,
        "partial_audit":          partial,
    })

    if failed_levels:
        return None, failed_levels[0]
    return annotated, ""


# ---------------------------------------------------------------------------
# Main async pipeline
# ---------------------------------------------------------------------------

async def run_verification(
    ladders: list[dict],
    rules: dict[str, dict[str, list[str]]],
    image_root: Path,
    model: str,
    backend: str,
    concurrency: int,
    max_retries: int,
    max_side: int,
    base_url: str | None = None,
    api_key: str | None = None,
    thinking_budget: int = 0,
    batch_size: int = 4,
    checkpoint_path: Path | None = None,
) -> tuple[list[dict], dict[str, int]]:
    if backend == "openai":
        resolved_key = (
            api_key
            or os.environ.get("OPENAI_API_KEY")
            or "dummy"
        )
        client_kwargs: dict = {"api_key": resolved_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        vision_client: AsyncOpenAI | LocalVisionModel = AsyncOpenAI(**client_kwargs)
    else:
        vision_client = get_local_vision_model(model, max_batch_size=batch_size)

    sem = asyncio.Semaphore(concurrency)

    ckpt_fh = open(checkpoint_path, "a") if checkpoint_path else None
    ckpt_lock = asyncio.Lock() if checkpoint_path else None

    async def _eval_and_checkpoint(ladder: dict) -> tuple[dict | None, str]:
        result = await _evaluate_ladder(
            sem, vision_client, model, backend, rules,
            ladder, image_root, max_side, max_retries,
            thinking_budget=thinking_budget,
        )
        if ckpt_fh and ckpt_lock:
            async with ckpt_lock:
                ckpt_fh.write((ladder.get("ladder_id") or "") + "\n")
                ckpt_fh.flush()
        return result

    try:
        tasks = [_eval_and_checkpoint(ladder) for ladder in ladders]

        kept: list[dict] = []
        rejection_counts: dict[str, int] = defaultdict(int)

        results = await atqdm.gather(*tasks, desc="Verifying", unit="ladder")
        for annotated, reason in results:
            if annotated is None:
                rejection_counts[reason] += 1
            else:
                kept.append(annotated)
    finally:
        if ckpt_fh:
            ckpt_fh.close()

    if backend == "local":
        vision_client.unload()

    return kept, dict(rejection_counts)


# ---------------------------------------------------------------------------
# Endpoint reachability check
# ---------------------------------------------------------------------------

def _ensure_endpoint_reachable(base_url: str) -> None:
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise RuntimeError(
            f"Invalid --base-url {base_url!r}. Expected an http(s) URL like "
            "'http://localhost:8001/v1'."
        )
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((parsed.hostname, port), timeout=3):
            return
    except OSError as exc:
        raise RuntimeError(
            f"Cannot reach the Stage 4 vision endpoint at {base_url} ({exc}). "
            "Start an OpenAI-compatible vision server there or pass --base-url to a reachable endpoint."
        ) from exc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input",           default="data/stage3_out.jsonl",
                   help="Source JSONL file (ladder format with images)")
    p.add_argument("--output",          default="data/stage4_out.jsonl",
                   help="Output JSONL file (verified ladders)")
    p.add_argument("--rules",           default="data/rules.jsonl",
                   help="Safety rules JSONL file")
    p.add_argument("--image-root",      default="data/images/",
                   help="Root directory for resolving image files")
    p.add_argument("--vlm-model",       default="Qwen/Qwen3-VL-8B-Thinking",
                   help="Vision LLM model name (default: Qwen/Qwen3-VL-8B-Thinking "
                        "(chain-of-thought enabled))")
    p.add_argument("--backend",         choices=("local", "openai"), default="local",
                   help="Inference backend for the vision model")
    p.add_argument("--base-url",        default=None,
                   help="OpenAI-compatible API base URL for the vision model (used with --backend openai)")
    p.add_argument("--batch-size",      default=4,   type=int,
                   help="Max image-text pairs per VLM forward pass (local backend only)")
    p.add_argument("--concurrency",     default=8,   type=int,
                   help="Max concurrent VLM calls")
    p.add_argument("--max-retries",     default=3,   type=int,
                   help="Max retries on API errors")
    p.add_argument("--max-side",        default=512, type=int,
                   help="Resize images so their longest side <= this value (px)")
    p.add_argument("--thinking-budget", default=0,   type=int,
                   help="Token budget for the thinking chain (0 = disabled, fastest; "
                        ">0 = enable chain-of-thought with at most this many extra tokens)")
    p.add_argument("--no-checkpoint",   action="store_true",
                   help="Ignore any existing checkpoint and process all ladders from scratch")
    p.add_argument("--api-key",         default=None,
                   help="API key override (default: OPENAI_API_KEY env var)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)
    rules_path  = Path(args.rules)
    image_root  = Path(args.image_root).resolve()

    if not image_root.is_dir():
        log.warning("Image root directory not found: %s — images may not resolve", image_root)

    if args.backend == "openai" and args.base_url:
        log.info("Checking Stage 4 vision endpoint at %s ...", args.base_url)
        try:
            _ensure_endpoint_reachable(args.base_url)
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
    elif args.backend == "openai" and not args.base_url:
        log.warning(
            "No Stage 4 vision base URL configured; AsyncOpenAI will use its default API base. "
            "Set --base-url for qwen3-vl-8b-thinking or other vision models."
        )

    log.info("Loading %s ...", input_path)
    ladders: list[dict] = []
    with input_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                ladders.append(json.loads(line))
    log.info("Loaded %d ladders.", len(ladders))

    log.info("Loading rules from %s ...", rules_path)
    rules = load_rules(rules_path)

    # Checkpoint / resume
    checkpoint_path = output_path.with_suffix(".stage4_progress")

    processed_ids: set[str] = set()
    already_kept: list[dict] = []

    if not args.no_checkpoint and checkpoint_path.exists():
        with checkpoint_path.open() as f:
            processed_ids = {line.strip() for line in f if line.strip()}

    if processed_ids and output_path.exists():
        with output_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    already_kept.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if processed_ids:
        log.info(
            "Resuming from checkpoint: %d already processed, %d previously passed",
            len(processed_ids), len(already_kept),
        )
        ladders = [l for l in ladders if l.get("ladder_id") not in processed_ids]
        log.info("Remaining to process: %d ladders", len(ladders))
    elif args.no_checkpoint and checkpoint_path.exists():
        checkpoint_path.unlink()

    log.info("=" * 60)
    log.info(
        "Stage 4: Visual Verification  (model=%s, backend=%s)",
        args.vlm_model, args.backend,
    )
    log.info("Image root : %s", image_root)
    log.info("=" * 60)

    t0 = time.perf_counter()
    kept_new, rejection_counts = asyncio.run(
        run_verification(
            ladders,
            rules=rules,
            image_root=image_root,
            model=args.vlm_model,
            backend=args.backend,
            concurrency=args.concurrency,
            max_retries=args.max_retries,
            max_side=args.max_side,
            base_url=args.base_url or None,
            api_key=args.api_key or None,
            thinking_budget=args.thinking_budget,
            batch_size=args.batch_size,
            checkpoint_path=checkpoint_path if not args.no_checkpoint else None,
        )
    )
    elapsed = time.perf_counter() - t0

    kept = already_kept + kept_new
    n_partial = sum(1 for r in kept if r.get("partial_audit"))

    log.info("=" * 60)
    log.info("Summary  (%.1fs)", elapsed)
    log.info("  Input ladders     : %d", len(ladders) + len(processed_ids))
    if processed_ids:
        log.info("  Skipped (checkpoint): %d", len(processed_ids))
        log.info("  Processed this run  : %d", len(ladders))
    for reason, count in sorted(rejection_counts.items()):
        log.info("  Rejected %-30s: -%d", reason, count)
    log.info("  Valid ladders     : %d", len(kept))
    log.info("  Partial audit     : %d  (missing images)", n_partial)
    log.info("=" * 60)

    n_vision_failures = sum(
        count for reason, count in rejection_counts.items()
        if reason.startswith("vision_api_failure(")
    )
    total_processed = len(ladders) + len(processed_ids)
    if total_processed and not kept:
        if n_vision_failures == len(ladders):
            raise SystemExit(
                "Stage 4 produced no valid ladders because every ladder hit vision_api_failure. "
                "Check --base-url, API keys, and the vision model server logs."
            )
        raise SystemExit(
            f"Stage 4 produced no valid ladders. "
            f"Rejection breakdown: {dict(rejection_counts)}"
        )

    log.info("Writing %s ...", output_path)
    write_mode = "a" if already_kept else "w"
    with output_path.open(write_mode) as f:
        for ladder in kept_new:
            f.write(json.dumps(ladder, ensure_ascii=False) + "\n")
    log.info("Done.")


if __name__ == "__main__":
    main()
