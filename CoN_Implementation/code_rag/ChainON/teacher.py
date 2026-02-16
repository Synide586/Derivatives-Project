from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from code_rag.ChainON.prompt_loader import load_prompt
from code_rag.teacher_notes.note_quality import (
    NoteQualityConfig,
    check_note_quality,
)
from .parser import note_block_to_canonical_text
from .validator import build_note_meta, enforce_unknown_rule


@dataclass
class TeacherGenConfig:
    max_retries: int = 3
    note_quality: NoteQualityConfig = field(default_factory=NoteQualityConfig)
    prompt_dir: str = "prompts/ChainON"


class LLMClient:
    """
    Minimal interface you can adapt to your existing llm wrapper.
    Must implement: generate(system: str, user: str) -> str
    """
    def generate(self, system: str, user: str) -> str:  # pragma: no cover
        raise NotImplementedError


def _fmt_teacher_user(
    template: str,
    *,
    question: str,
    i: int,
    doc_id: str,
    doc_text: str,
    previous_notes: List[str],
) -> str:
    prev = "(empty)" if not previous_notes else "\n\n".join(previous_notes)
    return template.format(
        question=question,
        i=i,
        i_minus_1=max(0, i - 1),
        doc_id=doc_id,
        doc_text=doc_text,
        previous_notes=prev,
    )


def generate_gold_notes_for_example(
    *,
    llm: LLMClient,
    question: str,
    docs: List[Dict[str, Any]],  # each has doc_id + snippet as doc_text
    gold_answer: Optional[str] = None,  # intentionally unused (no gold leakage)
    cfg: TeacherGenConfig = TeacherGenConfig(),
) -> Tuple[List[str], List[Dict[str, Any]]]:
    system_txt = load_prompt(f"{cfg.prompt_dir}/teacher_system.txt")
    user_template = load_prompt(f"{cfg.prompt_dir}/teacher_user_template.txt")
    repair_format = load_prompt(f"{cfg.prompt_dir}/repair_format.txt")
    repair_grounding = load_prompt(f"{cfg.prompt_dir}/repair_grounding.txt")

    notes: List[str] = []
    note_meta: List[Dict[str, Any]] = []
    prev_current_best = "uncertain"

    for i, d in enumerate(docs, start=1):
        doc_id = d["doc_id"]
        doc_text = d["snippet"]

        user = _fmt_teacher_user(
            user_template,
            question=question,
            i=i,
            doc_id=doc_id,
            doc_text=doc_text,
            previous_notes=notes,
        )

        raw = None
        last_errors: List[str] = []

        for _attempt in range(cfg.max_retries):
            raw = llm.generate(system=system_txt.format(i=i), user=user).strip()

            q = check_note_quality(raw, doc_text, cfg=cfg.note_quality)
            last_errors = q.errors[:]

            if not q.ok:
                # Repair selection: format-ish vs grounding-ish
                if any(
                    ("Field" in e) or ("Parse error" in e) or ("CoNCase" in e)
                    for e in q.errors
                ):
                    user = user + "\n\n" + repair_format
                else:
                    user = user + "\n\n" + repair_grounding
                continue

            canonical = note_block_to_canonical_text(raw)
            new_current_best, meta = build_note_meta(canonical, prev_current_best)

            rule_errs = enforce_unknown_rule(meta["con_case"], new_current_best)
            if rule_errs:
                last_errors = rule_errs
                user = user + "\n\n" + repair_format
                continue

            notes.append(canonical)
            note_meta.append(meta)
            prev_current_best = new_current_best
            break

        if raw is None or len(notes) != i:
            raise RuntimeError(
                f"Failed to generate valid ReadingNote {i}/{len(docs)} for doc_id={doc_id}. "
                f"Last errors: {json.dumps(last_errors, ensure_ascii=False)}"
            )

    return notes, note_meta
