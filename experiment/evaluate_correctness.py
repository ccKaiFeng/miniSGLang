#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List


ANSWER_PATTERNS = [
    re.compile(r"The answer is\s+(-?\d+(?:,\d{3})*(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"答案是\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)"),
    re.compile(r"最终答案[：:]\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)"),
]
NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def base_id(req_id: str) -> str:
    return re.sub(r"_r\d+$", "", str(req_id))


def normalize_number(value: Any) -> str:
    text = str(value).strip().replace(",", "")
    if text.endswith(".0"):
        text = text[:-2]
    return text


def extract_answer(text: str) -> str | None:
    for pattern in ANSWER_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            return normalize_number(matches[-1])
    numbers = NUMBER_RE.findall(text)
    if not numbers:
        return None
    return normalize_number(numbers[-1])


def evaluate(dataset: Path, results: Path) -> Dict[str, Any]:
    expected = {
        str(row["id"]): normalize_number(row["answer"])
        for row in load_jsonl(dataset)
        if "answer" in row
    }
    rows = load_jsonl(results)
    details = []
    correct = 0
    judged = 0
    for row in rows:
        req_id = str(row.get("id", ""))
        key = base_id(req_id)
        answer = expected.get(key)
        output_text = str(row.get("output_text", ""))
        prediction = extract_answer(output_text)
        is_correct = row.get("ok") is True and answer is not None and prediction == answer
        if answer is not None:
            judged += 1
            correct += int(is_correct)
        details.append(
            {
                "id": req_id,
                "base_id": key,
                "ok": bool(row.get("ok")),
                "expected": answer,
                "prediction": prediction,
                "correct": is_correct,
                "output_preview": output_text[:300],
            }
        )
    return {
        "dataset": str(dataset),
        "results": str(results),
        "num_results": len(rows),
        "num_judged": judged,
        "num_correct": correct,
        "accuracy": correct / judged if judged else 0.0,
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate numeric-answer correctness JSONL results.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    summary = evaluate(args.dataset, args.results)
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
