from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from code_rag.ChainON.prompt_loader import load_prompt
from code_rag.ChainON.anthropic_client import AnthropicLLM


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def format_docs_block(retrieved_docs: List[Dict[str, Any]]) -> str:
    # Keep it simple + stable: doc_id + snippet
    blocks = []
    for d in retrieved_docs:
        doc_id = d.get("doc_id", "")
        snippet = d.get("snippet", "")
        blocks.append(f"doc_id={doc_id}\n{snippet}")
    return "\n\n".join(blocks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", required=True, help="JSONL with {id, query, retrieved_docs}")
    ap.add_argument("--out_jsonl", required=True, help="Write outputs here")
    ap.add_argument("--prompt_dir", default="prompts/ChainON")
    ap.add_argument("--max_new_tokens", type=int, default=900)
    ap.add_argument("--model", default=None, help="Optional override if your Anthropic client supports it")
    args = ap.parse_args()

    system = load_prompt(f"{args.prompt_dir}/student_system.txt")
    user_template = load_prompt(f"{args.prompt_dir}/student_user_template.txt")

    llm = AnthropicLLM(model=args.model) if args.model else AnthropicLLM()

    rows = read_jsonl(args.input_jsonl)

    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_jsonl, "w", encoding="utf-8") as f_out:
        for ex in rows:
            q = ex["query"]
            docs = ex["retrieved_docs"]
            k = len(docs)

            docs_block = format_docs_block(docs)
            user = user_template.format(question=q, docs_block=docs_block, k=k)

            raw = llm.generate(system=system, user=user, max_output_tokens=16000).strip()

            out = {
                "id": ex.get("id"),
                "answer_raw": raw,
            }
            f_out.write(json.dumps(out, ensure_ascii=False) + "\n")

    print("Wrote:", args.out_jsonl)


if __name__ == "__main__":
    main()
