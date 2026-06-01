from __future__ import annotations

import os
from pathlib import Path


def _read_token_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    token = path.read_text(encoding="utf-8", errors="ignore").strip()
    return token or None


def resolve_hf_token() -> str | None:
    env_token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    if env_token:
        return env_token

    candidates: list[Path] = []
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        candidates.append(Path(hf_home) / "token")

    home = Path.home()
    candidates.extend([
        home / ".cache" / "huggingface" / "token",
        home / ".huggingface" / "token",
    ])

    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        token = _read_token_file(path)
        if token:
            return token
    return None
