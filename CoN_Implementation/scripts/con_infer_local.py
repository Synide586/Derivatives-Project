from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Set

from code_rag.ChainON.prompt_loader import load_prompt
from code_rag.ChainON.local_llm_client import LocalLLM, VLLMClient
from code_rag.ChainON.validator import validate_final_answer_citations
from code_rag.ChainON.parser import parse_student_output


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


def extract_valid_doc_ids(retrieved_docs: List[Dict[str, Any]]) -> Set[str]:
    """Extract valid document IDs from retrieved docs."""
    out: Set[str] = set()
    for d in retrieved_docs:
        doc_id = str(d.get("doc_id", "")).strip().lower()
        if doc_id:
            out.add(doc_id)
    return out


def build_citation_repair_user(question: str, docs_block: str, draft: str) -> str:
    """Build repair prompt for citation fixing."""
    return (
        f"Question:\n{question}\n\n"
        f"Documents (in order). Each document begins with its doc_id:\n{docs_block}\n\n"
        f"Draft Output (keep ReadingNotes unchanged; repair only FinalAnswer):\n{draft}\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run Chain-of-Note inference with local models (Qwen, Llama, Mistral)"
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
                    help="Use vLLM for faster inference (requires vllm installed)")
    
    # Generation parameters
    ap.add_argument("--max_output_tokens", type=int, default=16000,
                    help="Maximum tokens to generate")
    ap.add_argument("--temperature", type=float, default=0.2,
                    help="Sampling temperature (0.0 = deterministic)")
    
    # Memory optimization
    ap.add_argument("--load_in_8bit", action="store_true", 
                    help="Use 8-bit quantization (saves ~50%% memory)")
    ap.add_argument("--load_in_4bit", action="store_true", 
                    help="Use 4-bit quantization (saves ~75%% memory)")
    
    # Citation repair (optional)
    ap.add_argument("--enable_citation_repair", action="store_true", default=False,
                    help="Enable citation repair (second LLM call to fix formatting)")
    ap.add_argument("--repair_max_output_tokens", type=int, default=8000,
                    help="Max tokens for repair generation")
    
    # Progress tracking
    ap.add_argument("--report_every", type=int, default=10,
                    help="Print progress every N examples")
    
    args = ap.parse_args()
    
    print("=" * 80)
    print("Chain-of-Note Inference with Local Models")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Input: {args.input_jsonl}")
    print(f"Output: {args.out_jsonl}")
    print(f"Citation Repair: {'Enabled' if args.enable_citation_repair else 'Disabled'}")
    print(f"vLLM: {'Enabled' if args.use_vllm else 'Disabled'}")
    if args.load_in_8bit:
        print("Quantization: 8-bit")
    elif args.load_in_4bit:
        print("Quantization: 4-bit")
    else:
        print("Quantization: None (full precision)")
    print("=" * 80)
    
    # Load prompts
    print("\nLoading prompts...")
    system = load_prompt(f"{args.prompt_dir}/student_system.txt")
    user_template = load_prompt(f"{args.prompt_dir}/student_user_template.txt")
    
    if args.enable_citation_repair:
        repair_system = load_prompt(f"{args.prompt_dir}/repair_citations.txt")
    
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
    
    # Statistics
    total_examples = len(rows)
    citation_errors_detected = 0
    repairs_attempted = 0
    repairs_successful = 0
    
    # Process examples
    print("\nStarting inference...\n")
    with open(args.out_jsonl, "w", encoding="utf-8") as f_out:
        for idx, ex in enumerate(rows, start=1):
            example_id = ex.get("id", f"example_{idx}")
            
            if idx % args.report_every == 0 or idx == 1:
                print(f"[{idx}/{total_examples}] Processing: {example_id}")
            
            # Extract question and documents
            q = ex["query"]
            docs = ex["retrieved_docs"]
            k = len(docs)
            
            # Format documents
            docs_block = format_docs_block(docs)
            valid_doc_ids = extract_valid_doc_ids(docs)
            
            # Build user prompt
            user = user_template.format(question=q, docs_block=docs_block, k=k)
            
            # Generate initial output
            raw = llm.generate(
                system=system,
                user=user,
                max_output_tokens=args.max_output_tokens,
            ).strip()
            
            # Validate citations
            citation_errors = []
            try:
                _, final_block = parse_student_output(raw, k=k)
                citation_errors = validate_final_answer_citations(final_block, valid_doc_ids)
            except Exception as e:
                citation_errors = [f"Parse failure: {repr(e)}"]
            
            # Track citation error statistics
            if citation_errors:
                citation_errors_detected += 1
                if idx % args.report_every == 0:
                    print(f"  ⚠ Citation errors: {len(citation_errors)}")
                    for err in citation_errors[:3]:  # Show first 3 errors
                        print(f"    - {err}")
            
            # Attempt repair if enabled and errors found
            if citation_errors and args.enable_citation_repair:
                repairs_attempted += 1
                
                if idx % args.report_every == 0:
                    print(f"  🔧 Attempting citation repair...")
                
                # Build repair prompt
                repair_user = build_citation_repair_user(q, docs_block, raw)
                
                # Generate repaired output
                repaired = llm.generate(
                    system=repair_system,
                    user=repair_user,
                    max_output_tokens=args.repair_max_output_tokens,
                    temperature=0.0,  # Use temperature=0 for deterministic repair
                ).strip()
                
                # Re-validate repaired output
                repaired_errors = []
                try:
                    _, final_block2 = parse_student_output(repaired, k=k)
                    repaired_errors = validate_final_answer_citations(final_block2, valid_doc_ids)
                except Exception as e:
                    repaired_errors = [f"Parse failure after repair: {repr(e)}"]
                
                # Use repaired version if it's better
                if not repaired_errors:
                    raw = repaired
                    repairs_successful += 1
                    if idx % args.report_every == 0:
                        print(f"  ✓ Repair successful!")
                else:
                    # Keep original and append validation errors for debugging
                    if idx % args.report_every == 0:
                        print(f"  ✗ Repair failed, keeping original")
                    raw = raw + "\n\n" + "VALIDATION_ERRORS:\n" + "\n".join(repaired_errors)
            
            # Write output
            out = {
                "id": example_id,
                "answer_raw": raw,
            }
            f_out.write(json.dumps(out, ensure_ascii=False) + "\n")
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("INFERENCE COMPLETE")
    print("=" * 80)
    print(f"Total examples processed: {total_examples}")
    print(f"Citation errors detected: {citation_errors_detected} ({citation_errors_detected/total_examples*100:.1f}%)")
    
    if args.enable_citation_repair:
        print(f"Repairs attempted: {repairs_attempted}")
        print(f"Repairs successful: {repairs_successful}")
        if repairs_attempted > 0:
            print(f"Repair success rate: {repairs_successful/repairs_attempted*100:.1f}%")
    
    print(f"\nOutput written to: {args.out_jsonl}")
    print("=" * 80)


if __name__ == "__main__":
    main()
