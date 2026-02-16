from __future__ import annotations
from pathlib import Path
from typing import Dict

_PROMPT_CACHE: Dict[str, str] = {}


def load_prompt(path: str) -> str:
    """
    Loads a text prompt from disk and caches it.
    Path is relative or absolute.
    """
    if path in _PROMPT_CACHE:
        return _PROMPT_CACHE[path]
    p = Path(path)
    txt = p.read_text(encoding="utf-8")
    _PROMPT_CACHE[path] = txt
    return txt
