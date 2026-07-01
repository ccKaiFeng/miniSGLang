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
CHOICE_PATTERNS = [
    re.compile(r"(?:答案|最终答案|选项|answer)\s*[：:是is]*\s*([ABCD])", re.IGNORECASE),
    re.compile(r"\b([ABCD])\b", re.IGNORECASE),
]


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


def normalize_text(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\u4e00-\u9fff.%-]+", " ", text)
    return text.strip()


def extract_number_answer(text: str) -> str | None:
    for pattern in ANSWER_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            return normalize_number(matches[-1])
    numbers = NUMBER_RE.findall(text)
    if not numbers:
        return None
    return normalize_number(numbers[-1])


def extract_choice_answer(text: str) -> str | None:
    for pattern in CHOICE_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            return str(matches[-1]).upper()
    return None


def extract_answer(text: str, answer_type: str) -> str | None:
    if answer_type == "choice":
        return extract_choice_answer(text)
    if answer_type == "number":
        return extract_number_answer(text)
    return normalize_text(text)


def is_text_contains_correct(output_text: str, answers: List[str]) -> bool:
    output = normalize_text(output_text)
    if not output:
        return False
    for answer in answers:
        expected = normalize_text(answer)
        if expected and (expected in output or output in expected):
            return True
    return False


def evaluate(dataset: Path, results: Path) -> Dict[str, Any]:
    dataset_rows = load_jsonl(dataset)
    expected = {str(row["id"]): row for row in dataset_rows}
    rows = load_jsonl(results)
    details = []
    correct = 0
    judged = 0
    for row in rows:
        req_id = str(row.get("id", ""))
        key = base_id(req_id)
        source = expected.get(key)
        output_text = str(row.get("output_text", ""))
        answer_type = str((source or {}).get("answer_type") or "number")
        expected_answer: Any = None
        prediction: str | None = None
        is_correct = False
        if source is not None and row.get("ok") is True:
            if answer_type == "text_contains":
                answers = [str(x) for x in source.get("answers", [])]
                expected_answer = answers
                prediction = normalize_text(output_text[:1000])
                is_correct = is_text_contains_correct(output_text, answers)
            elif answer_type == "choice":
                expected_answer = str(source.get("answer", "")).strip().upper()
                prediction = extract_answer(output_text, answer_type)
                is_correct = prediction == expected_answer
            else:
                expected_answer = normalize_number(source.get("answer", ""))
                prediction = extract_answer(output_text, "number")
                is_correct = prediction == expected_answer
        if source is not None:
            judged += 1
            correct += int(is_correct)
        details.append(
            {
                "id": req_id,
                "base_id": key,
                "ok": bool(row.get("ok")),
                "answer_type": answer_type,
                "expected": expected_answer,
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
