from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Set

from code_rag.ChainON.prompt_loader import load_prompt
from code_rag.ChainON.anthropic_client import AnthropicLLM
from code_rag.ChainON.validator import validate_final_answer_citations
from code_rag.ChainON.parser import parse_student_output


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def format_docs_block(retrieved_docs: List[Dict[str, Any]]) -> str:
    blocks = []
    for d in retrieved_docs:
        doc_id = d.get("doc_id", "")
        snippet = d.get("snippet", "")
        blocks.append(f"doc_id={doc_id}\n{snippet}")
    return "\n\n".join(blocks)


def extract_valid_doc_ids(retrieved_docs: List[Dict[str, Any]]) -> Set[str]:
    out: Set[str] = set()
    for d in retrieved_docs:
        doc_id = str(d.get("doc_id", "")).strip().lower()
        if doc_id:
            out.add(doc_id)
    return out


def build_citation_repair_user(question: str, docs_block: str, draft: str) -> str:
    return (
        f"Question:\n{question}\n\n"
        f"Documents (in order). Each document begins with its doc_id:\n{docs_block}\n\n"
        f"Draft Output (keep ReadingNotes unchanged; repair only FinalAnswer):\n{draft}\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", required=True, help="JSONL with {id, query, retrieved_docs}")
    ap.add_argument("--out_jsonl", required=True, help="Write outputs here")
    ap.add_argument("--prompt_dir", default="prompts/ChainON")
    ap.add_argument("--model", default=None, help="Optional Anthropic model override")
    ap.add_argument("--max_output_tokens", type=int, default=16000)
    ap.add_argument("--enable_citation_repair", action="store_true", default=False)
    ap.add_argument("--repair_max_output_tokens", type=int, default=8000)
    args = ap.parse_args()

    system = load_prompt(f"{args.prompt_dir}/student_system.txt")
    user_template = load_prompt(f"{args.prompt_dir}/student_user_template.txt")

    repair_system = load_prompt(f"{args.prompt_dir}/repair_citations.txt")

    llm = AnthropicLLM(model=args.model) if args.model else AnthropicLLM()

    rows = read_jsonl(args.input_jsonl)

    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_jsonl, "w", encoding="utf-8") as f_out:
        for ex in rows:
            q = ex["query"]
            docs = ex["retrieved_docs"]
            k = len(docs)

            docs_block = format_docs_block(docs)
            valid_doc_ids = extract_valid_doc_ids(docs)

            user = user_template.format(question=q, docs_block=docs_block, k=k)

            raw = llm.generate(
                system=system,
                user=user,
                max_output_tokens=args.max_output_tokens,
            ).strip()

            # Validate FinalAnswer citations; if invalid, repair (optional but recommended)
            try:
                _, final_block = parse_student_output(raw, k=k)
                errs = validate_final_answer_citations(final_block, valid_doc_ids)
            except Exception as e:
                errs = [f"Parse failure: {repr(e)}"]

            if errs and args.enable_citation_repair:
                repair_user = build_citation_repair_user(q, docs_block, raw)
                repaired = llm.generate(
                    system=repair_system,
                    user=repair_user,
                    max_output_tokens=args.repair_max_output_tokens,
                    temperature=0.0,
                ).strip()

                # Re-validate
                try:
                    _, final_block2 = parse_student_output(repaired, k=k)
                    errs2 = validate_final_answer_citations(final_block2, valid_doc_ids)
                except Exception as e:
                    errs2 = [f"Parse failure after repair: {repr(e)}"]

                if not errs2:
                    raw = repaired
                else:
                    # Keep original raw, but also surface errors for debugging
                    raw = raw + "\n\n" + "VALIDATION_ERRORS:\n" + "\n".join(errs2)

            out = {"id": ex.get("id"), "answer_raw": raw}
            f_out.write(json.dumps(out, ensure_ascii=False) + "\n")

    print("Wrote:", args.out_jsonl)


if __name__ == "__main__":
    main()
