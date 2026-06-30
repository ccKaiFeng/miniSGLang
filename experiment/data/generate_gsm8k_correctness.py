#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "ZipCache/ZipCache/asset/gsm8k_sample.txt"
OUTPUT = ROOT / "experiment/data/gsm8k_correctness.jsonl"

ANSWER_RE = re.compile(r"The answer is\s+(-?\d+(?:\.\d+)?)", re.IGNORECASE)


def normalize_number(text: str) -> str:
    value = text.strip().replace(",", "")
    if value.endswith(".0"):
        value = value[:-2]
    return value


def parse_examples(text: str) -> list[dict]:
    rows = []
    blocks = [b.strip() for b in text.split("\n\nQuestion: ") if b.strip()]
    for idx, block in enumerate(blocks, start=1):
        if not block.startswith("Question:"):
            block = "Question: " + block
        answer_match = ANSWER_RE.search(block)
        if answer_match is None:
            continue
        question_part = block.split("Let's think step by step", 1)[0]
        question = question_part[len("Question:") :].strip()
        answer = normalize_number(answer_match.group(1))
        prompt = (
            "Solve the following grade-school math problem. "
            "You may reason step by step, but the final line must be exactly: "
            "'The answer is <number>'.\n\n"
            f"Question: {question}"
        )
        rows.append(
            {
                "id": f"gsm8k_{idx:03d}",
                "prompt": prompt,
                "answer": answer,
                "max_tokens": 256,
            }
        )
    return rows


def main() -> None:
    rows = parse_examples(SOURCE.read_text(encoding="utf-8"))
    with OUTPUT.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
