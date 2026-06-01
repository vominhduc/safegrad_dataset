"""Shared utilities for the SafeGrad pipeline."""

import hashlib
import re

# ---------------------------------------------------------------------------
# Severity-level constants
# ---------------------------------------------------------------------------

LEVELS_ORDERED: tuple[str, ...] = ("safe", "low_risk", "mid_risk", "high_risk")
UNSAFE_LEVELS: tuple[str, ...] = ("low_risk", "mid_risk", "high_risk")
LEVEL_RANK: dict[str, int] = {lvl: i for i, lvl in enumerate(LEVELS_ORDERED)}


def norm_level(s: str) -> str:
    """Normalise a severity label string to one of the four canonical forms.

    Returns one of: 'safe', 'low_risk', 'mid_risk', 'high_risk'.
    Falls back to 'safe' for unrecognised strings.
    """
    s = s.strip().lower().replace(" ", "_").replace("-", "_")
    if s in LEVELS_ORDERED:
        return s
    if s in ("low", "lowrisk"):
        return "low_risk"
    if s in ("mid", "medium", "medium_risk", "midrisk"):
        return "mid_risk"
    if s in ("high", "highrisk"):
        return "high_risk"
    return "safe"


# ---------------------------------------------------------------------------
# Strategy detection
# Each entry is a (strategy_name, list_of_keyword_patterns).
# Patterns are applied case-insensitively; first match wins.
# Prompts that match none are labelled "explicit" (direct/literal framing).
# ---------------------------------------------------------------------------
_STRATEGY_PATTERNS: list[tuple[str, list[str]]] = [
    (
        "metaphor",
        [
            r"\bmetaphor\b", r"\bsymbolically\b", r"\bfiguratively\b",
            r"\ballegor", r"\banalog", r"\bas if\b", r"\brepresents\b",
            r"\bembodies\b", r"\bstand[s]? for\b",
        ],
    ),
    (
        "roleplay",
        [
            r"\brole[\s-]?play\b", r"\bpretend\b", r"\bimagine you are\b",
            r"\bact as\b", r"\bin character\b", r"\bimpersonat",
            r"\bplay the role\b",
        ],
    ),
    (
        "fictional",
        [
            r"\bfictional\b", r"\bin a (story|novel|movie|film|game|fantasy)\b",
            r"\bnarrative\b", r"\bscript\b", r"\bscreenplay\b",
            r"\bfairy[\s-]?tale\b", r"\bfable\b", r"\bsci[\s-]?fi\b",
        ],
    ),
    (
        "technical",
        [
            r"\bclinical\b", r"\bmedical\b", r"\bscientific\b",
            r"\btherapeutic\b", r"\bacademic\b", r"\bresearch\b",
            r"\bstatistical\b", r"\bdiagnost",
        ],
    ),
    (
        "indirect",
        [
            r"\bimplicit\b", r"\bsubt\b", r"\beuphemism\b",
            r"\binnuendo\b", r"\bveiled\b", r"\bambiguous\b",
            r"\bindirect\b", r"\ballud",
        ],
    ),
]

_COMPILED: list[tuple[str, list[re.Pattern]]] = [
    (name, [re.compile(p, re.IGNORECASE) for p in pats])
    for name, pats in _STRATEGY_PATTERNS
]


def classify_strategy(prompt: str) -> str:
    """Return the rhetorical strategy used in *prompt*.

    Returns one of: 'metaphor', 'roleplay', 'fictional', 'technical',
    'indirect', or 'explicit' (default when nothing else matches).
    """
    for name, patterns in _COMPILED:
        if any(pat.search(prompt) for pat in patterns):
            return name
    return "explicit"


def descriptiveness_score(record: dict) -> float:
    """Proxy for how descriptive / information-rich a record is.

    Works with both the old flat schema (``prompt_unsafe`` / ``prompt_safe``
    top-level keys produced by Phase 1) and the new rung-based schema
    (``rung_safe`` / ``rung_{level}`` objects produced by Phase 2+).

    Combines character length of both prompts with a small bonus for
    records whose unsafe verdict is confirmed (UNSAFE) and safe verdict
    is confirmed (SAFE), signalling a high-quality, verified sample.
    """
    # --- unsafe prompt ---
    unsafe_prompt: str = record.get("prompt_unsafe", "")
    if not unsafe_prompt:
        for level in UNSAFE_LEVELS:
            rung = record.get(f"rung_{level}", {})
            if rung.get("prompt"):
                unsafe_prompt = rung["prompt"]
                break

    # --- safe prompt ---
    safe_rung = record.get("rung_safe", {})
    safe_prompt: str = safe_rung.get("prompt") or record.get("prompt_safe", "")

    length = len(unsafe_prompt) + len(safe_prompt)

    # --- verdict bonus ---
    unsafe_verdict: str = record.get("verdict_unsafe", "")
    if not unsafe_verdict:
        for level in UNSAFE_LEVELS:
            rung = record.get(f"rung_{level}", {})
            if rung.get("verdict"):
                unsafe_verdict = rung["verdict"]
                break

    safe_verdict: str = safe_rung.get("verdict") or record.get("verdict_safe", "")

    verdict_bonus = (
        50 if unsafe_verdict == "UNSAFE" else 0
    ) + (
        50 if safe_verdict == "SAFE" else 0
    )
    return float(length + verdict_bonus)


def md5_pair(prompt_unsafe: str, prompt_safe: str) -> str:
    """MD5 hash of the canonical (unsafe, safe) prompt pair."""
    key = f"{prompt_unsafe.strip()}|||{prompt_safe.strip()}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()
