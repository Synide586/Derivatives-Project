from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

REQUIRED_FIELDS = ["DocSummary", "CoNCase", "CurrentBestAnswer", "Rationale"]
ALLOWED_CON_CASE = {
    "relevant_find_answer",
    "irrelevant_infer_answer",
    "irrelevant_answer_unknown",
}


@dataclass
class ParsedNote:
    fields: Dict[str, str]
    verdict: Optional[str] = None  # legacy


def _extract_field_block(text: str, field: str) -> Optional[str]:
    pattern = rf"(?s)\b{re.escape(field)}\s*:\s*(.*?)(?=\n(?:{'|'.join(REQUIRED_FIELDS)}|Verdict)\s*:|\Z)"
    m = re.search(pattern, text)
    if not m:
        return None
    return m.group(1).strip()


def parse_single_note_block(note_text: str) -> ParsedNote:
    fields: Dict[str, str] = {}
    for f in REQUIRED_FIELDS:
        v = _extract_field_block(note_text, f)
        if v is None:
            raise ValueError(f"Missing required field: {f}")
        fields[f] = v

    verdict = _extract_field_block(note_text, "Verdict")
    return ParsedNote(fields=fields, verdict=verdict)


def parse_student_output(full_text: str, k: int) -> Tuple[List[str], str]:
    """
    Returns:
      - notes: list of note blocks (raw text inside ReadingNote i: ...), length k
      - final_answer_block: raw text after FinalAnswer:
    Supports headers like:
      ReadingNote 1:
      ReadingNote 1 (DocID: d1):
    """
    notes: List[str] = []

    for i in range(1, k + 1):
        # Match "ReadingNote i" optionally followed by "(DocID: di)"
        start_pat = rf"ReadingNote\s+{i}(?:\s*\(DocID:\s*d{i}\s*\))?\s*:\s*"
        # Next header can be ReadingNote i+1 with optional DocID, or FinalAnswer
        if i < k:
            next_pat = rf"\nReadingNote\s+{i+1}(?:\s*\(DocID:\s*d{i+1}\s*\))?\s*:"
        else:
            next_pat = r"\nFinalAnswer\s*:"

        pat = rf"(?s){start_pat}(.*?)(?={next_pat}|\Z)"
        m = re.search(pat, full_text)
        if not m:
            raise ValueError(f"Could not find ReadingNote {i} block")
        notes.append(m.group(1).strip())

    m_final = re.search(r"(?s)FinalAnswer\s*:\s*(.*)\Z", full_text)
    if not m_final:
        raise ValueError("Could not find FinalAnswer block")
    final_block = m_final.group(1).strip()
    return notes, final_block


def note_block_to_canonical_text(note_text: str) -> str:
    parsed = parse_single_note_block(note_text)
    con_case = parsed.fields["CoNCase"].strip().lower()
    if con_case not in ALLOWED_CON_CASE:
        raise ValueError(f"CoNCase not in allowed enum: {con_case}")

    return "\n".join(
        [
            f"DocSummary: {parsed.fields['DocSummary'].strip()}",
            f"CoNCase: {parsed.fields['CoNCase'].strip()}",
            f"CurrentBestAnswer: {parsed.fields['CurrentBestAnswer'].strip()}",
            f"Rationale: {parsed.fields['Rationale'].strip()}",
        ]
    )
