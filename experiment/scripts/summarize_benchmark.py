#!/usr/bin/env python3
"""Summarize ZipCache benchmark results into a comparison report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_experiment_dirs(log_root: Path, pattern: str) -> List[Path]:
    return sorted(
        [d for d in log_root.iterdir() if d.is_dir() and pattern in d.name],
        key=lambda d: d.name,
    )


def extract_metrics(summary: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "num_requests",
        "num_ok",
        "request_throughput_rps",
        "output_chunk_throughput_cps",
        "ttft_avg_s",
        "ttft_p50_s",
        "ttft_p90_s",
        "ttft_p99_s",
        "e2e_avg_s",
        "tpot_avg_s",
        "gpu_memory_used_mb_max",
    ]
    return {k: summary.get(k) for k in keys if k in summary}


def fmt_val(val: Any, unit: str = "") -> str:
    if val is None:
        return "n/a"
    if isinstance(val, float):
        return f"{val:.4g}{unit}"
    return f"{val}{unit}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize ZipCache benchmark results")
    parser.add_argument("--log-root", type=Path, default=Path("experiment/logs"))
    parser.add_argument("--pattern", default="8b", help="Pattern to match experiment dirs")
    parser.add_argument("--output", type=Path, default=None, help="Output markdown file")
    args = parser.parse_args()

    dirs = find_experiment_dirs(args.log_root, args.pattern)
    if not dirs:
        print(f"No experiment directories found matching '{args.pattern}' in {args.log_root}")
        sys.exit(1)

    lines = [
        "# ZipCache V3 Benchmark Summary",
        "",
        f"Pattern: `{args.pattern}`",
        f"Directories found: {len(dirs)}",
        "",
    ]

    experiments: Dict[str, List[Dict[str, Any]]] = {}
    for d in dirs:
        for summary_file in sorted(d.glob("*_summary.json")):
            exp_name = summary_file.stem.replace("_summary", "")
            summary = load_json(summary_file)
            metrics = extract_metrics(summary)
            metrics["_dir"] = str(d)
            metrics["_mode"] = d.name.split("_", 2)[-1] if "_" in d.name else d.name
            experiments.setdefault(exp_name, []).append(metrics)

    for exp_name, runs in sorted(experiments.items()):
        lines.extend([
            f"## {exp_name}",
            "",
            "| mode | rps | chunks/s | ttft avg | ttft p90 | ttft p99 | e2e avg | gpu max MB |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for run in runs:
            mode = run.get("_mode", "unknown")
            lines.append(
                "| {mode} | {rps} | {cps} | {ttft} | {ttft90} | {ttft99} | {e2e} | {gpu} |".format(
                    mode=mode,
                    rps=fmt_val(run.get("request_throughput_rps")),
                    cps=fmt_val(run.get("output_chunk_throughput_cps")),
                    ttft=fmt_val(run.get("ttft_avg_s"), "s"),
                    ttft90=fmt_val(run.get("ttft_p90_s"), "s"),
                    ttft99=fmt_val(run.get("ttft_p99_s"), "s"),
                    e2e=fmt_val(run.get("e2e_avg_s"), "s"),
                    gpu=fmt_val(run.get("gpu_memory_used_mb_max"), "MB"),
                )
            )
        lines.append("")

    output = "\n".join(lines) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
        print(f"Summary written to: {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
