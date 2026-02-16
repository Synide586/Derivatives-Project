from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from code_rag.ChainON.prompt_loader import load_prompt
from code_rag.ChainON.local_llm_client import LocalLLM, VLLMClient


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read JSONL file into list of dictionaries."""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def format_docs_block(retrieved_docs: List[Dict[str, Any]]) -> str:
    """Format retrieved documents into a text block."""
    blocks = []
    for d in retrieved_docs:
        doc_id = d.get("doc_id", "")
        snippet = d.get("snippet", "")
        blocks.append(f"doc_id={doc_id}\n{snippet}")
    return "\n\n".join(blocks)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Simple CoN inference without citation repair (for research baselines)"
    )
    
    # Input/Output
    ap.add_argument("--input_jsonl", required=True,
                    help="Input JSONL with {id, query, retrieved_docs}")
    ap.add_argument("--out_jsonl", required=True,
                    help="Output JSONL path for results")
    ap.add_argument("--prompt_dir", default="prompts/ChainON",
                    help="Directory containing prompt templates")
    
    # Model selection
    ap.add_argument("--model_path", required=True,
                    help="Path to local model or HuggingFace model ID")
    ap.add_argument("--use_vllm", action="store_true",
                    help="Use vLLM for faster inference")
    
    # Generation parameters
    ap.add_argument("--max_output_tokens", type=int, default=16000,
                    help="Maximum tokens to generate")
    ap.add_argument("--temperature", type=float, default=0.2,
                    help="Sampling temperature")
    
    # Memory optimization
    ap.add_argument("--load_in_8bit", action="store_true",
                    help="Use 8-bit quantization")
    ap.add_argument("--load_in_4bit", action="store_true",
                    help="Use 4-bit quantization")
    
    # Progress tracking
    ap.add_argument("--report_every", type=int, default=10,
                    help="Print progress every N examples")
    
    args = ap.parse_args()
    
    print("=" * 80)
    print("Chain-of-Note Inference (No Repair) - Research Baseline")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Input: {args.input_jsonl}")
    print(f"Output: {args.out_jsonl}")
    print(f"vLLM: {'Enabled' if args.use_vllm else 'Disabled'}")
    if args.load_in_8bit:
        print("Quantization: 8-bit")
    elif args.load_in_4bit:
        print("Quantization: 4-bit")
    else:
        print("Quantization: None")
    print("=" * 80)
    
    # Load prompts
    print("\nLoading prompts...")
    system = load_prompt(f"{args.prompt_dir}/student_system.txt")
    user_template = load_prompt(f"{args.prompt_dir}/student_user_template.txt")
    
    # Initialize model
    print("\nInitializing model...")
    if args.use_vllm:
        llm = VLLMClient(
            model_path=args.model_path,
            max_output_tokens=args.max_output_tokens,
            temperature=args.temperature,
        )
    else:
        llm = LocalLLM(
            model_path=args.model_path,
            max_output_tokens=args.max_output_tokens,
            temperature=args.temperature,
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit,
        )
    
    # Load input data
    print("\nLoading input data...")
    rows = read_jsonl(args.input_jsonl)
    print(f"Loaded {len(rows)} examples")
    
    # Create output directory
    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    
    # Process examples
    print("\nStarting inference...\n")
    with open(args.out_jsonl, "w", encoding="utf-8") as f_out:
        for idx, ex in enumerate(rows, start=1):
            example_id = ex.get("id", f"example_{idx}")
            
            if idx % args.report_every == 0 or idx == 1:
                print(f"[{idx}/{len(rows)}] Processing: {example_id}")
            
            q = ex["query"]
            docs = ex["retrieved_docs"]
            k = len(docs)
            
            docs_block = format_docs_block(docs)
            user = user_template.format(question=q, docs_block=docs_block, k=k)
            
            # Generate output (no repair)
            raw = llm.generate(
                system=system,
                user=user,
                max_output_tokens=args.max_output_tokens,
            ).strip()
            
            # Write output
            out = {
                "id": example_id,
                "answer_raw": raw,
            }
            f_out.write(json.dumps(out, ensure_ascii=False) + "\n")
    
    print("\n" + "=" * 80)
    print("INFERENCE COMPLETE")
    print("=" * 80)
    print(f"Total examples processed: {len(rows)}")
    print(f"Output written to: {args.out_jsonl}")
    print("=" * 80)


if __name__ == "__main__":
    main()
