#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_CMMLU_SUBJECTS = [
    "computer_science",
    "high_school_mathematics",
    "college_mathematics",
    "machine_learning",
    "chinese_history",
    "professional_medicine",
    "economics",
    "legal_and_moral_basis",
]

DEFAULT_LONGBENCH_TASKS = [
    "hotpotqa",
    "qasper",
    "multifieldqa_en",
    "multifieldqa_zh",
    "2wikimqa",
    "passage_retrieval_en",
    "passage_retrieval_zh",
]


def load_from_disk_dataset(path: Path) -> Any:
    try:
        from datasets import load_from_disk
    except Exception as exc:  # pragma: no cover - depends on local environment.
        raise SystemExit(
            "prepare_public_workloads.py needs the `datasets` package to read "
            "downloaded Arrow datasets. Install it with: "
            "python -m pip install datasets pyarrow"
        ) from exc
    return load_from_disk(str(path))


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def extract_gsm8k_final_answer(answer: str) -> str:
    if "####" in answer:
        return answer.rsplit("####", 1)[-1].strip().replace(",", "")
    numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", answer)
    return numbers[-1].replace(",", "") if numbers else answer.strip()


def truncate_text(text: str, max_chars: int) -> str:
    text = str(text)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = max_chars * 3 // 4
    tail = max_chars - head
    return text[:head] + "\n\n...[context truncated for benchmark]...\n\n" + text[-tail:]


def make_gsm8k(root: Path, out: Path, limit: int) -> int:
    ds = load_from_disk_dataset(root / "gsm8k" / "main")
    split = ds["test"] if "test" in ds else ds["train"]
    rows = []
    for i, row in enumerate(split.select(range(min(limit, len(split))))):
        rows.append(
            {
                "id": f"gsm8k_public_{i:04d}",
                "source": "openai/gsm8k",
                "task": "gsm8k",
                "answer_type": "number",
                "prompt": (
                    "Solve the following grade-school math problem. You may reason "
                    "step by step, but the final line must be exactly: "
                    "'The answer is <number>'.\n\n"
                    f"Question: {row['question']}"
                ),
                "answer": extract_gsm8k_final_answer(row["answer"]),
                "max_tokens": 1024,
            }
        )
    return write_jsonl(out, rows)


def make_cmmlu(root: Path, out: Path, limit: int, subjects: List[str]) -> int:
    rows: List[Dict[str, Any]] = []
    per_subject = max(1, limit // max(1, len(subjects)))
    for subject in subjects:
        subject_dir = root / "cmmlu" / subject
        if not subject_dir.exists():
            continue
        ds = load_from_disk_dataset(subject_dir)
        split = ds["test"] if "test" in ds else ds["dev"]
        for i, row in enumerate(split.select(range(min(per_subject, len(split))))):
            prompt = (
                "下面是一道中文多项选择题。请只在最后一行输出一个选项字母，格式为："
                "答案：A/B/C/D。\n\n"
                f"科目：{subject}\n"
                f"题目：{row['Question']}\n"
                f"A. {row['A']}\n"
                f"B. {row['B']}\n"
                f"C. {row['C']}\n"
                f"D. {row['D']}\n"
            )
            rows.append(
                {
                    "id": f"cmmlu_{subject}_{i:04d}",
                    "source": "haonan-li/cmmlu",
                    "task": "cmmlu",
                    "subject": subject,
                    "answer_type": "choice",
                    "choices": {"A": row["A"], "B": row["B"], "C": row["C"], "D": row["D"]},
                    "prompt": prompt,
                    "answer": str(row["Answer"]).strip().upper(),
                    "max_tokens": 128,
                }
            )
            if len(rows) >= limit:
                return write_jsonl(out, rows)
    return write_jsonl(out, rows)


def longbench_prompt(row: Dict[str, Any], *, max_context_chars: int) -> str:
    context = truncate_text(str(row.get("context", "")), max_context_chars)
    question = str(row.get("input", ""))
    lang = str(row.get("language", "en"))
    if lang == "zh":
        return (
            "请根据下面的长上下文回答问题。答案要简洁，并且不要编造上下文中没有的信息。\n\n"
            f"上下文：\n{context}\n\n问题：{question}\n"
        )
    return (
        "Answer the question using only the long context below. Keep the answer concise.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n"
    )


def collect_longbench_rows(
    root: Path,
    tasks: List[str],
    *,
    max_context_chars: int,
    per_task: int,
    sort_by_length: bool = False,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for task in tasks:
        task_dir = root / "longbench" / task
        if not task_dir.exists():
            continue
        ds = load_from_disk_dataset(task_dir)
        items = list(ds)
        if sort_by_length:
            items.sort(key=lambda x: int(x.get("length") or len(str(x.get("context", "")))), reverse=True)
        for i, row in enumerate(items[:per_task]):
            answers = row.get("answers") or []
            rows.append(
                {
                    "id": f"longbench_{task}_{i:04d}",
                    "source": "THUDM/LongBench",
                    "task": task,
                    "answer_type": "text_contains",
                    "prompt": longbench_prompt(row, max_context_chars=max_context_chars),
                    "answers": list(answers) if isinstance(answers, list) else [str(answers)],
                    "max_tokens": 512,
                    "input_chars": len(str(row.get("context", ""))) + len(str(row.get("input", ""))),
                    "reported_length": row.get("length"),
                }
            )
    return rows


def make_longbench_qa(root: Path, out: Path, limit: int, tasks: List[str]) -> int:
    per_task = max(1, limit // max(1, len(tasks)))
    rows = collect_longbench_rows(
        root, tasks, max_context_chars=14000, per_task=per_task, sort_by_length=False
    )
    return write_jsonl(out, rows[:limit])


def make_long_context_pressure(root: Path, out: Path, limit: int, tasks: List[str]) -> int:
    per_task = max(1, limit // max(1, len(tasks)))
    rows = collect_longbench_rows(
        root, tasks, max_context_chars=28000, per_task=per_task, sort_by_length=True
    )
    for row in rows:
        row["id"] = row["id"].replace("longbench_", "long_context_")
        row["max_tokens"] = 256
    rows.sort(key=lambda x: int(x.get("input_chars") or 0), reverse=True)
    return write_jsonl(out, rows[:limit])


def make_public_shared_prefix(
    root: Path,
    out: Path,
    *,
    groups: int,
    prompts_per_group: int,
    max_context_chars: int,
) -> int:
    source_tasks = ["hotpotqa", "qasper", "multifieldqa_zh", "multi_news", "gov_report"]
    candidates = collect_longbench_rows(
        root,
        [task for task in source_tasks if (root / "longbench" / task).exists()],
        max_context_chars=max_context_chars,
        per_task=max(groups * 2, 4),
        sort_by_length=True,
    )
    candidates.sort(key=lambda x: int(x.get("input_chars") or 0), reverse=True)
    templates = [
        "Answer the original question from this context: {question}",
        "List the key evidence needed to answer this question: {question}",
        "Give a concise answer and cite the relevant sentence for: {question}",
        "Summarize the context facts that matter for this query: {question}",
        "If the answer is not explicit, say so. Query: {question}",
        "Return only the final answer for: {question}",
        "Explain the reasoning path for: {question}",
        "Provide a short factual answer to: {question}",
    ]
    rows: List[Dict[str, Any]] = []
    for group_id, base in enumerate(candidates[:groups]):
        prompt = base["prompt"]
        if "Question:" in prompt:
            context_part, question = prompt.rsplit("Question:", 1)
        elif "问题：" in prompt:
            context_part, question = prompt.rsplit("问题：", 1)
        else:
            context_part, question = prompt, "What is the key answer?"
        for qid in range(prompts_per_group):
            question_text = templates[qid % len(templates)].format(question=question.strip())
            rows.append(
                {
                    "id": f"public_shared_prefix_g{group_id:03d}_q{qid:03d}",
                    "source": "THUDM/LongBench-derived-shared-prefix",
                    "task": "public_shared_prefix",
                    "group_id": group_id,
                    "prompt": context_part + "Question: " + question_text + "\n",
                    "max_tokens": 160,
                    "input_chars": len(context_part) + len(question_text),
                }
            )
    return write_jsonl(out, rows)


def make_ruler_squad(root: Path, out: Path, limit: int) -> int:
    squad_path = root / "ruler" / "squad.json"
    if not squad_path.exists():
        return write_jsonl(out, [])
    obj = json.loads(squad_path.read_text(encoding="utf-8"))
    rows: List[Dict[str, Any]] = []
    for article in obj.get("data", []):
        title = article.get("title", "")
        for paragraph in article.get("paragraphs", []):
            context = paragraph.get("context", "")
            for qa in paragraph.get("qas", []):
                if qa.get("is_impossible"):
                    continue
                answers = [a.get("text", "") for a in qa.get("answers", []) if a.get("text")]
                if not answers:
                    continue
                rows.append(
                    {
                        "id": f"ruler_squad_{len(rows):04d}",
                        "source": "NVIDIA/RULER SQuAD helper data",
                        "task": "ruler_squad",
                        "title": title,
                        "answer_type": "text_contains",
                        "prompt": (
                            "Answer the question using only the context below. "
                            "Keep the answer short.\n\n"
                            f"Context:\n{context}\n\nQuestion: {qa.get('question', '')}\n"
                        ),
                        "answers": answers,
                        "max_tokens": 256,
                    }
                )
                if len(rows) >= limit:
                    return write_jsonl(out, rows)
    return write_jsonl(out, rows)


def copy_synthetic_shared_prefix(root: Path, out: Path, limit: int) -> int:
    src = root / "synthetic" / "generated_shared_prefix.jsonl"
    rows: List[Dict[str, Any]] = []
    if src.exists():
        with src.open("r", encoding="utf-8") as f:
            for line in f:
                if len(rows) >= limit:
                    break
                row = json.loads(line)
                row["id"] = f"synthetic_shared_prefix_{int(row['id']):04d}"
                row["source"] = "generated-shared-prefix"
                row["task"] = "synthetic_shared_prefix"
                row["max_tokens"] = int(row.get("max_tokens") or row.get("output_len") or 256)
                rows.append(row)
    return write_jsonl(out, rows)


def write_manifest(out_dir: Path, counts: Dict[str, int], args: argparse.Namespace) -> None:
    manifest = {
        "workload_dir": str(out_dir),
        "source_root": str(args.root),
        "counts": counts,
        "notes": [
            "These JSONL files are derived from the downloaded public datasets under experiment/.",
            "run_all_experiments.py uses these files by default.",
            "experiment/data is kept for historical handcrafted tests but is not used by default.",
        ],
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build benchmark JSONL workloads from local public datasets.")
    parser.add_argument("--root", type=Path, default=Path("experiment"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiment/workloads"))
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--gsm8k-limit", type=int, default=128)
    parser.add_argument("--cmmlu-limit", type=int, default=256)
    parser.add_argument("--longbench-limit", type=int, default=160)
    parser.add_argument("--long-context-limit", type=int, default=128)
    parser.add_argument("--ruler-squad-limit", type=int, default=128)
    parser.add_argument("--public-shared-prefix-groups", type=int, default=24)
    parser.add_argument("--public-shared-prefix-per-group", type=int, default=8)
    parser.add_argument("--synthetic-shared-prefix-limit", type=int, default=256)
    args = parser.parse_args()

    random.seed(args.seed)
    args.root = args.root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    counts = {
        "gsm8k_public_correctness": make_gsm8k(
            args.root, args.output_dir / "gsm8k_public_correctness.jsonl", args.gsm8k_limit
        ),
        "cmmlu_public_correctness": make_cmmlu(
            args.root,
            args.output_dir / "cmmlu_public_correctness.jsonl",
            args.cmmlu_limit,
            DEFAULT_CMMLU_SUBJECTS,
        ),
        "longbench_public_qa": make_longbench_qa(
            args.root,
            args.output_dir / "longbench_public_qa.jsonl",
            args.longbench_limit,
            DEFAULT_LONGBENCH_TASKS,
        ),
        "longbench_long_context_pressure": make_long_context_pressure(
            args.root,
            args.output_dir / "longbench_long_context_pressure.jsonl",
            args.long_context_limit,
            DEFAULT_LONGBENCH_TASKS,
        ),
        "public_shared_prefix": make_public_shared_prefix(
            args.root,
            args.output_dir / "public_shared_prefix.jsonl",
            groups=args.public_shared_prefix_groups,
            prompts_per_group=args.public_shared_prefix_per_group,
            max_context_chars=22000,
        ),
        "ruler_squad_qa": make_ruler_squad(
            args.root, args.output_dir / "ruler_squad_qa.jsonl", args.ruler_squad_limit
        ),
        "synthetic_shared_prefix": copy_synthetic_shared_prefix(
            args.root,
            args.output_dir / "synthetic_shared_prefix.jsonl",
            args.synthetic_shared_prefix_limit,
        ),
    }
    write_manifest(args.output_dir, counts, args)
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
