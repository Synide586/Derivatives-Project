from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any, Dict, List

from code_rag.data.con_dataset_builder import (
    SplitConfig,
    build_con_augmented_example,
    read_jsonl,
    split_data,
    strip_to_base_con_inputs,
    write_jsonl,
)
from code_rag.ChainON.teacher import TeacherGenConfig
from code_rag.ChainON.anthropic_client import AnthropicLLM


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--report_path", default="outputs/con_notegen_report.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dev_frac", type=float, default=0.1)
    ap.add_argument("--test_frac", type=float, default=0.1)
    ap.add_argument("--max_retries", type=int, default=3)
    args = ap.parse_args()

    raw = read_jsonl(args.input_jsonl)

    base = [strip_to_base_con_inputs(x) for x in raw]
    base_path = str(Path(args.out_dir) / "base_con_inputs.jsonl")
    write_jsonl(base_path, base)

    train, dev, test = split_data(base, SplitConfig(seed=args.seed, dev_frac=args.dev_frac, test_frac=args.test_frac))

    llm = AnthropicLLM()
    teacher_cfg = TeacherGenConfig(
        max_retries=args.max_retries,
        prompt_dir="prompts/ChainON",
    )

    stats: Dict[str, Any] = {
        "format_error_rate": 0.0,
        "regen_rate": 0.0,
        "overlap_fail_rate": 0.0,
        "distribution_con_case": collections.Counter(),
        "avg_note_length_tokens": 0.0,
        "n_total": 0,
        "n_failed": 0,
    }

    def build_split(name: str, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out_items: List[Dict[str, Any]] = []
        for ex in items:
            stats["n_total"] += 1
            try:
                aug = build_con_augmented_example(ex, llm=llm, teacher_cfg=teacher_cfg, prompt_version="con_v1")

                con_cases = [m["con_case"] for m in aug["con"]["note_meta"]]
                stats["distribution_con_case"].update(con_cases)

                lens = [len(n.split()) for n in aug["con"]["notes"]]
                stats["avg_note_length_tokens"] += sum(lens) / max(1, len(lens))

                out_items.append(aug)
            except Exception as e:
                import traceback
                stats["n_failed"] += 1
                print(f"\n[FAIL] id={ex.get('id')} error={repr(e)}")
                traceback.print_exc()
        return out_items

    con_train = build_split("train", train)
    con_dev = build_split("dev", dev)
    con_test = build_split("test", test)

    if stats["n_total"] > 0:
        denom = max(1, stats["n_total"] - stats["n_failed"])
        stats["avg_note_length_tokens"] = stats["avg_note_length_tokens"] / denom

    stats["distribution_con_case"] = dict(stats["distribution_con_case"])

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    write_jsonl(str(Path(args.out_dir) / "con_train.jsonl"), con_train)
    write_jsonl(str(Path(args.out_dir) / "con_dev.jsonl"), con_dev)
    write_jsonl(str(Path(args.out_dir) / "con_test.jsonl"), con_test)

    Path(args.report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("Wrote:", args.out_dir)
    print("Report:", args.report_path)


if __name__ == "__main__":
    main()
