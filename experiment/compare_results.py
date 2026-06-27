#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = int(round((len(values) - 1) * p))
    return values[idx]


def summary(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    ok = [r for r in rows if r.get("ok")]
    ttft = [r["ttft_s"] for r in ok if r.get("ttft_s") is not None]
    e2e = [r["e2e_s"] for r in ok if r.get("e2e_s") is not None]
    tpot = [r["tpot_s"] for r in ok if r.get("tpot_s") is not None]
    chunks = sum(int(r.get("output_chunks", 0)) for r in ok)
    wall = max((r["end_time"] for r in ok), default=0) - min(
        (r["start_time"] for r in ok), default=0
    )
    return {
        "num": len(rows),
        "ok": len(ok),
        "failed": len(rows) - len(ok),
        "wall_s": wall,
        "rps": len(ok) / wall if wall > 0 else 0.0,
        "chunks_per_s": chunks / wall if wall > 0 else 0.0,
        "ttft_avg_s": statistics.mean(ttft) if ttft else 0.0,
        "ttft_p50_s": percentile(ttft, 0.50),
        "ttft_p90_s": percentile(ttft, 0.90),
        "e2e_avg_s": statistics.mean(e2e) if e2e else 0.0,
        "e2e_p50_s": percentile(e2e, 0.50),
        "e2e_p90_s": percentile(e2e, 0.90),
        "tpot_avg_s": statistics.mean(tpot) if tpot else 0.0,
        "tpot_p50_s": percentile(tpot, 0.50),
        "tpot_p90_s": percentile(tpot, 0.90),
    }


def ratio(candidate: float, baseline: float) -> str:
    if baseline == 0:
        return "n/a"
    return f"{candidate / baseline:.3f}x"


def print_table(base: Dict[str, float], cand: Dict[str, float]) -> None:
    keys = [
        "ok",
        "failed",
        "rps",
        "chunks_per_s",
        "ttft_avg_s",
        "ttft_p50_s",
        "ttft_p90_s",
        "e2e_avg_s",
        "e2e_p50_s",
        "e2e_p90_s",
        "tpot_avg_s",
        "tpot_p50_s",
        "tpot_p90_s",
    ]
    print("| metric | baseline | candidate | candidate/baseline |")
    print("| --- | ---: | ---: | ---: |")
    for key in keys:
        b = base[key]
        c = cand[key]
        print(f"| {key} | {b:.6g} | {c:.6g} | {ratio(c, b)} |")


def print_text_diff(base_rows: List[Dict[str, Any]], cand_rows: List[Dict[str, Any]]) -> None:
    base_map = {r.get("id"): r for r in base_rows}
    cand_map = {r.get("id"): r for r in cand_rows}
    print("\n## Text comparison")
    for req_id in sorted(set(base_map) & set(cand_map)):
        b = base_map[req_id].get("output_text", "")
        c = cand_map[req_id].get("output_text", "")
        same = b == c
        print(f"\n### {req_id} same={same}")
        print("baseline:")
        print(b[:1000])
        print("candidate:")
        print(c[:1000])


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two miniSGLang benchmark result JSONLs.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--show-text", action="store_true")
    args = parser.parse_args()

    base_rows = load_jsonl(args.baseline)
    cand_rows = load_jsonl(args.candidate)
    base = summary(base_rows)
    cand = summary(cand_rows)
    print_table(base, cand)
    if args.show_text:
        print_text_diff(base_rows, cand_rows)


if __name__ == "__main__":
    main()
