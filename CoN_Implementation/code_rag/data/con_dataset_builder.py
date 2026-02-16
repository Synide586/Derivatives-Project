from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from code_rag.ChainON.schema import ConBlock, Example, RetrievedDoc
from code_rag.ChainON.teacher import LLMClient, TeacherGenConfig, generate_gold_notes_for_example


STRIP_KEYS = {
    "per_doc_notes",
    "conflict_type",
    "conflict_reason",
    "answerable_under_evidence",
    "expected_response",
    "think",
    # plus any other keys you used for your old oracle/e2e pipeline
}


def strip_to_base_con_inputs(ex: Dict[str, Any]) -> Example:
    out: Example = {
        "id": ex["id"],
        "query": ex["query"],
        "retrieved_docs": ex["retrieved_docs"],
    }
    if "gold_answer" in ex:
        out["gold_answer"] = ex["gold_answer"]
    # Remove everything else explicitly
    return out


@dataclass
class SplitConfig:
    seed: int = 42
    dev_frac: float = 0.1
    test_frac: float = 0.1


def split_data(items: List[Example], cfg: SplitConfig) -> Tuple[List[Example], List[Example], List[Example]]:
    rng = random.Random(cfg.seed)
    rng.shuffle(items)
    n = len(items)
    n_test = int(n * cfg.test_frac)
    n_dev = int(n * cfg.dev_frac)
    test = items[:n_test]
    dev = items[n_test : n_test + n_dev]
    train = items[n_test + n_dev :]
    return train, dev, test


def build_con_augmented_example(
    ex: Example,
    llm: LLMClient,
    teacher_cfg: TeacherGenConfig,
    prompt_version: str = "con_v1",
) -> Example:
    docs: List[RetrievedDoc] = ex["retrieved_docs"]
    k = len(docs)
    doc_order = [d["doc_id"] for d in docs]

    gold_answer: Optional[str] = ex.get("gold_answer")
    notes, note_meta = generate_gold_notes_for_example(
        llm=llm,
        question=ex["query"],
        docs=docs,
        gold_answer=None,
        cfg=teacher_cfg,
    )

    final_target = note_meta[-1]["current_best"] if note_meta else ""

    con = ConBlock(
        k=k,
        doc_order=doc_order,
        notes=notes,
        note_meta=note_meta,
        final_target_answer=final_target,
        prompt_version=prompt_version,
    )
    con.assert_invariants(docs)

    out: Example = dict(ex)
    out["con"] = {
        "k": con.k,
        "doc_order": con.doc_order,
        "notes": con.notes,
        "note_meta": con.note_meta,
        "final_target_answer": con.final_target_answer,
        "prompt_version": con.prompt_version,
    }
    return out


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def write_jsonl(path: str, items: Iterable[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
