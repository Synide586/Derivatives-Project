from __future__ import annotations

import re
from typing import Dict, List, Set, Tuple

from .schema import ConNoteMeta, ConCaseType
from .parser import parse_single_note_block


FINAL_ANSWER_SENT_RE = re.compile(r"(?<=[.!?])\s+")
CITE_AT_END_RE = re.compile(r"(?:\[[dD]\d+\])+$")          # citations must be at end
CITE_TOKEN_RE = re.compile(r"\[([dD]\d+)\]")              # capture dK tokens


def build_note_meta(note_text: str, prev_current_best: str) -> Tuple[str, ConNoteMeta]:
    parsed = parse_single_note_block(note_text)

    con_case_raw = parsed.fields["CoNCase"].strip().lower()
    con_case: ConCaseType = con_case_raw  # type: ignore[assignment]

    current_best = parsed.fields["CurrentBestAnswer"].strip()
    changed = current_best != prev_current_best

    meta: ConNoteMeta = {
        "con_case": con_case,
        "current_best": current_best,
        "changed_answer": bool(changed),
    }
    return current_best, meta


def enforce_unknown_rule(con_case: str, current_best: str) -> List[str]:
    errs: List[str] = []
    if con_case == "irrelevant_answer_unknown":
        if current_best.strip().lower() != "unknown":
            errs.append("CoNCase=irrelevant_answer_unknown but CurrentBestAnswer is not exactly 'unknown'.")
    return errs


def split_sentences(text: str) -> List[str]:
    t = " ".join(text.strip().split())
    if not t:
        return []
    # split on sentence end punctuation + whitespace
    parts = FINAL_ANSWER_SENT_RE.split(t)
    return [p.strip() for p in parts if p.strip()]


def validate_final_answer_citations(final_answer: str, valid_doc_ids: Set[str]) -> List[str]:
    """
    Enforces:
    - If abstaining: exact phrase "Unknown based on provided documents." (case-sensitive),
      no citations.
    - Else: 2-4 sentences, each sentence ends with one or more citations like [d1][d2]
      with NO spaces between citations (this regex enforces adjacency).
    - Every cited doc id must be in valid_doc_ids.
    """
    errs: List[str] = []
    fa = final_answer.strip()

    if fa == "Unknown based on provided documents.":
        # No citations allowed
        if CITE_TOKEN_RE.search(fa):
            errs.append("Abstention must not contain citations.")
        return errs

    sents = split_sentences(fa)
    if not (2 <= len(sents) <= 4):
        errs.append(f"FinalAnswer must have 2–4 sentences, got {len(sents)}.")

    for idx, s in enumerate(sents, start=1):
        # Require citations exactly at end
        if not CITE_AT_END_RE.search(s):
            errs.append(f"Sentence {idx} must end with citations like [d1][d2] (no spaces).")
            continue

        cited = [m.group(1).lower() for m in CITE_TOKEN_RE.finditer(s)]
        if not cited:
            errs.append(f"Sentence {idx} has no citations.")
            continue

        bad = [c for c in cited if c not in valid_doc_ids]
        if bad:
            errs.append(f"Sentence {idx} cites invalid doc_ids: {bad}. Allowed: {sorted(valid_doc_ids)}")

        # Enforce no spaces between adjacent citations by checking the tail substring
        tail = CITE_TOKEN_RE.sub(lambda m: f"[{m.group(1).lower()}]", s)
        # if there is a pattern "] [" near the end, it's spaced
        if re.search(r"\]\s+\[d\d+\]\s*$", tail):
            errs.append(f"Sentence {idx} has spaces between citations; must be like [d1][d2].")

    return errs
