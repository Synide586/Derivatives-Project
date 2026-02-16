from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, TypedDict


# Paper-aligned CoN cases:
# (a) relevant -> find answer
# (b) irrelevant but useful context -> infer answer
# (c) irrelevant and cannot infer -> unknown
ConCaseType = Literal[
    "relevant_find_answer",
    "irrelevant_infer_answer",
    "irrelevant_answer_unknown",
]


class RetrievedDoc(TypedDict):
    doc_id: str
    source_url: str
    snippet: str
    timestamp: str


class ConNoteMeta(TypedDict):
    con_case: ConCaseType
    current_best: str  # answer string OR "unknown" OR "uncertain"
    changed_answer: bool


@dataclass
class ConBlock:
    k: int
    doc_order: List[str]
    notes: List[str]
    note_meta: List[ConNoteMeta]
    final_target_answer: str
    prompt_version: str = "con_v1"

    def assert_invariants(self, retrieved_docs: List[RetrievedDoc]) -> None:
        assert self.k == len(retrieved_docs), f"k={self.k} != len(retrieved_docs)={len(retrieved_docs)}"
        assert len(self.notes) == self.k, f"len(notes)={len(self.notes)} != k={self.k}"
        assert len(self.note_meta) == self.k, f"len(note_meta)={len(self.note_meta)} != k={self.k}"
        assert len(self.doc_order) == self.k, f"len(doc_order)={len(self.doc_order)} != k={self.k}"
        for i, d in enumerate(retrieved_docs):
            assert self.doc_order[i] == d["doc_id"], (
                f"doc_order[{i}]={self.doc_order[i]} != retrieved_docs[{i}].doc_id={d['doc_id']}"
            )

        # CoN case invariant: if case is "irrelevant_answer_unknown", current_best must be "unknown"
        for m in self.note_meta:
            if m["con_case"] == "irrelevant_answer_unknown":
                assert m["current_best"].strip().lower() == "unknown", (
                    "If con_case=irrelevant_answer_unknown, current_best must be exactly 'unknown'."
                )


class Example(TypedDict, total=False):
    id: str
    query: str
    retrieved_docs: List[RetrievedDoc]
    gold_answer: str
    con: Dict[str, Any]