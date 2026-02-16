from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from code_rag.ChainON.parser import parse_single_note_block


REQUIRED_FIELDS = ["DocSummary", "CoNCase", "CurrentBestAnswer", "Rationale"]

ALLOWED_CON_CASE = {
    "relevant_find_answer",
    "irrelevant_infer_answer",
    "irrelevant_answer_unknown",
}

# inference markers required for "irrelevant_infer_answer"
INFER_MARKERS = {
    "infer",
    "inferred",
    "inference",
    "inherent knowledge",
    "intrinsic knowledge",
    "deduce",
    "deduced",
    "therefore",
    "so i",
    "so we",
}


def _simple_tokens(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


STOP = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "as", "is", "are", "was", "were",
    "be", "by", "at", "it", "this", "that", "from", "but", "not", "their", "they", "he", "she", "we", "you",
}


def _nontrivial_tokens(s: str) -> List[str]:
    return [t for t in _simple_tokens(s) if t not in STOP and len(t) >= 3]


def _bigrams(tokens: List[str]) -> List[Tuple[str, str]]:
    return list(zip(tokens, tokens[1:]))


@dataclass
class NoteQualityConfig:
    max_docsummary_tokens: int = 60
    max_rationale_tokens: int = 80
    max_total_tokens: int = 180
    min_overlap_frac: float = 0.02
    min_anchor_bigrams: int = 1


@dataclass
class NoteQualityResult:
    ok: bool
    errors: List[str]
    warnings: List[str]
    parsed_con_case: Optional[str] = None


def check_note_quality(
    note_text: str,
    doc_text: str,
    cfg: NoteQualityConfig = NoteQualityConfig(),
) -> NoteQualityResult:
    errors: List[str] = []
    warnings: List[str] = []

    # ---- Format checks (hard fail) ----
    for f in REQUIRED_FIELDS:
        count = len(re.findall(rf"\b{re.escape(f)}\s*:", note_text))
        if count != 1:
            errors.append(f"Field '{f}:' must appear exactly once (found {count}).")

    try:
        parsed = parse_single_note_block(note_text)
    except Exception as e:
        errors.append(f"Parse error: {e}")
        return NoteQualityResult(ok=False, errors=errors, warnings=warnings)

    con_case = parsed.fields["CoNCase"].strip().lower()
    if con_case not in ALLOWED_CON_CASE:
        errors.append(f"CoNCase must be one of {sorted(ALLOWED_CON_CASE)} (got '{con_case}').")

    docsum_toks = _simple_tokens(parsed.fields["DocSummary"])
    rat_toks = _simple_tokens(parsed.fields["Rationale"])
    tot_toks = _simple_tokens(note_text)

    if len(docsum_toks) > cfg.max_docsummary_tokens:
        errors.append(f"DocSummary too long: {len(docsum_toks)} tokens > {cfg.max_docsummary_tokens}.")
    if len(rat_toks) > cfg.max_rationale_tokens:
        errors.append(f"Rationale too long: {len(rat_toks)} tokens > {cfg.max_rationale_tokens}.")
    if len(tot_toks) > cfg.max_total_tokens:
        errors.append(f"Total note too long: {len(tot_toks)} tokens > {cfg.max_total_tokens}.")

    # CoN case specific hard constraints:
    cba = parsed.fields["CurrentBestAnswer"].strip()
    if con_case == "irrelevant_answer_unknown":
        if cba.lower() != "unknown":
            errors.append("If CoNCase=irrelevant_answer_unknown, CurrentBestAnswer must be exactly 'unknown'.")

    if con_case == "irrelevant_infer_answer":
        rat_lower = parsed.fields["Rationale"].lower()
        if not any(m in rat_lower for m in INFER_MARKERS):
            errors.append(
                "If CoNCase=irrelevant_infer_answer, Rationale must explicitly indicate inference "
                f"(must contain one of: {sorted(INFER_MARKERS)})."
            )

    if errors:
        return NoteQualityResult(ok=False, errors=errors, warnings=warnings, parsed_con_case=con_case)

    # ---- Grounding checks (light but useful) ----
    doc_nt = _nontrivial_tokens(doc_text)
    if len(doc_nt) == 0:
        warnings.append("Document text has no nontrivial tokens; grounding checks skipped.")
        return NoteQualityResult(ok=True, errors=[], warnings=warnings, parsed_con_case=con_case)

    doc_set = set(doc_nt)

    docsum_nt = _nontrivial_tokens(parsed.fields["DocSummary"])
    if docsum_nt:
        overlap = sum(1 for t in docsum_nt if t in doc_set) / max(1, len(docsum_nt))
        if overlap < cfg.min_overlap_frac:
            errors.append(f"DocSummary overlap too low: {overlap:.3f} < {cfg.min_overlap_frac}.")

    rat_nt = _nontrivial_tokens(parsed.fields["Rationale"])
    rat_bi = set(_bigrams(rat_nt))
    doc_bi = set(_bigrams(doc_nt))
    anchor_hits = len(rat_bi.intersection(doc_bi))
    if anchor_hits < cfg.min_anchor_bigrams:
        errors.append(f"Rationale missing anchor phrase: bigram_hits={anchor_hits} < {cfg.min_anchor_bigrams}.")

    return NoteQualityResult(ok=(len(errors) == 0), errors=errors, warnings=warnings, parsed_con_case=con_case)


def repair_instruction_format_fix() -> str:
    return (
        "Rewrite the note strictly following the template with EXACTLY these fields once each:\n"
        "DocSummary:\nCoNCase:\nCurrentBestAnswer:\nRationale:\n"
        "CoNCase must be exactly one of: relevant_find_answer | irrelevant_infer_answer | irrelevant_answer_unknown.\n"
        "If CoNCase=irrelevant_answer_unknown, CurrentBestAnswer must be exactly 'unknown'.\n"
        "Keep it short and do not add any other headings."
    )


def repair_instruction_grounding_fix() -> str:
    return (
        "Rewrite the note to be grounded in the document. In Rationale, include one short quoted phrase "
        "(<= 8 words) copied verbatim from the document as an anchor. Keep the 4-field schema with CoNCase."
    )
