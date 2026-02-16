from __future__ import annotations

from typing import List, Tuple

from code_rag.ChainON.prompt_loader import load_prompt


def build_student_prompt(
    question: str,
    docs: List[str],
    prompt_dir: str = "prompts/ChainON",
) -> Tuple[str, str]:
    system = load_prompt(f"{prompt_dir}/student_system.txt")
    user_template = load_prompt(f"{prompt_dir}/student_user_template.txt")

    # Ensure each doc begins with its doc_id (d1..dK) to match the prompt contract
    blocks = []
    for i, t in enumerate(docs, start=1):
        blocks.append(f"doc_id=d{i}\n{t}")
    docs_block = "\n\n".join(blocks)

    user = user_template.format(question=question, docs_block=docs_block, k=len(docs))
    return system, user
