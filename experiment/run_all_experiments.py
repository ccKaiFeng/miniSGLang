#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List


EXPERIMENTS = [
    {
        "name": "shared_prefix",
        "dataset": "data/shared_prefix.jsonl",
        "concurrency": 4,
        "repeat": 2,
        "max_tokens": 256,
        "description": "共享长前缀测试，主要观察长上下文、prefix cache 和 ZipCache 压缩开销。",
    },
    {
        "name": "realistic_long_context",
        "dataset": "data/realistic_long_context.jsonl",
        "concurrency": 8,
        "repeat": 3,
        "max_tokens": 512,
        "description": "真实长上下文压力测试：较多长 prompt、高并发、较长输出，用于观察 KV cache 显存压力和端到端性能。",
    },
    {
        "name": "zipcache_restore_probe",
        "dataset": "data/zipcache_restore_probe.jsonl",
        "concurrency": 1,
        "repeat": 8,
        "max_tokens": 256,
        "description": "ZipCache v2 强命中测试：顺序重复同一个长 prompt，专门观察 compressed hit 和 restore 是否触发。",
    },
    {
        "name": "zipcache_restore_pressure",
        "dataset": "data/zipcache_restore_pressure.jsonl",
        "concurrency": 1,
        "repeat": 6,
        "max_tokens": 384,
        "description": "ZipCache v2 restore 压力测试：多个共享超长前缀请求顺序重复，增加 compressed hit/restore 触发概率。",
    },
    {
        "name": "mixed_length",
        "dataset": "data/mixed_length.jsonl",
        "concurrency": 4,
        "repeat": 2,
        "max_tokens": 256,
        "description": "混合长度测试，模拟普通在线服务负载。",
    },
    {
        "name": "correctness",
        "dataset": "data/correctness.jsonl",
        "concurrency": 1,
        "repeat": 1,
        "max_tokens": 96,
        "description": "正确性冒烟测试，低并发 greedy 输出，便于人工对比 main/ZipCache 输出。",
    },
    {
        "name": "gsm8k_correctness",
        "dataset": "data/gsm8k_correctness.jsonl",
        "concurrency": 1,
        "repeat": 1,
        "max_tokens": 256,
        "description": "GSM8K 数学题正确性测试，使用标准数字答案自动计算 accuracy。",
        "evaluate": "gsm8k",
    },
]


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def check_server(base_url: str, timeout: float = 5.0) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/v1"
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
        return {
            "ok": True,
            "url": url,
            "latency_s": time.perf_counter() - start,
            "response": body[:500],
        }
    except Exception as e:
        return {
            "ok": False,
            "url": url,
            "latency_s": time.perf_counter() - start,
            "error": repr(e),
        }


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def run_command(cmd: List[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        proc.wait()
        log.write(f"\n[exit_code] {proc.returncode}\n")
        return int(proc.returncode)


def summarize_for_report(summary: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "num_requests",
        "num_ok",
        "num_failed",
        "request_throughput_rps",
        "output_chunk_throughput_cps",
        "ttft_avg_s",
        "ttft_p50_s",
        "ttft_p90_s",
        "e2e_avg_s",
        "e2e_p50_s",
        "e2e_p90_s",
        "tpot_avg_s",
        "tpot_p50_s",
        "tpot_p90_s",
        "gpu_memory_used_mb_min",
        "gpu_memory_used_mb_max",
        "gpu_memory_used_mb_avg",
    ]
    return {key: summary.get(key) for key in keys if key in summary}


def write_markdown_report(
    path: Path,
    *,
    mode: str,
    base_url: str,
    run_dir: Path,
    manifest: Dict[str, Any],
    results: List[Dict[str, Any]],
    zipcache_stats: Dict[str, Any] | None,
) -> None:
    lines = [
        f"# miniSGLang Experiment Report: {mode}",
        "",
        f"- mode: `{mode}`",
        f"- base_url: `{base_url}`",
        f"- run_dir: `{run_dir}`",
        f"- started_at: `{manifest['started_at']}`",
        f"- git_branch: `{manifest.get('git_branch', 'unknown')}`",
        f"- git_commit: `{manifest.get('git_commit', 'unknown')}`",
        "",
        "## Server Check",
        "",
        "```json",
        json.dumps(manifest["server_check"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Experiments",
        "",
        "| experiment | ok/total | rps | chunks/s | ttft avg | ttft p90 | e2e avg | tpot avg | gpu max MB |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        summary = row["summary"]
        ok_total = f"{summary.get('num_ok', 0)}/{summary.get('num_requests', 0)}"
        lines.append(
            "| {name} | {ok_total} | {rps:.4g} | {cps:.4g} | {ttft:.4g} | {ttft90:.4g} | {e2e:.4g} | {tpot:.4g} | {gpu} |".format(
                name=row["name"],
                ok_total=ok_total,
                rps=float(summary.get("request_throughput_rps", 0)),
                cps=float(summary.get("output_chunk_throughput_cps", 0)),
                ttft=float(summary.get("ttft_avg_s", 0)),
                ttft90=float(summary.get("ttft_p90_s", 0)),
                e2e=float(summary.get("e2e_avg_s", 0)),
                tpot=float(summary.get("tpot_avg_s", 0)),
                gpu=summary.get("gpu_memory_used_mb_max", "n/a"),
            )
        )
    lines.extend(["", "## Result Files", ""])
    for row in results:
        lines.append(f"- `{row['name']}` results: `{row['result_file']}`")
        lines.append(f"- `{row['name']}` summary: `{row['summary_file']}`")
        lines.append(f"- `{row['name']}` log: `{row['log_file']}`")
        if row.get("eval_file") is not None:
            lines.append(f"- `{row['name']}` correctness eval: `{row['eval_file']}`")

    eval_rows = [row for row in results if row.get("eval_summary") is not None]
    if eval_rows:
        lines.extend(
            [
                "",
                "## Correctness Evaluation",
                "",
                "| experiment | judged | correct | accuracy |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for row in eval_rows:
            ev = row["eval_summary"]
            lines.append(
                "| {name} | {judged} | {correct} | {acc:.4g} |".format(
                    name=row["name"],
                    judged=int(ev.get("num_judged", 0)),
                    correct=int(ev.get("num_correct", 0)),
                    acc=float(ev.get("accuracy", 0.0)),
                )
            )

    if zipcache_stats is not None:
        lines.extend(
            [
                "",
                "## ZipCache Stats",
                "",
                "```json",
                json.dumps(zipcache_stats, ensure_ascii=False, indent=2),
                "```",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def get_git_info(repo_root: Path) -> Dict[str, str]:
    def run(args: List[str]) -> str:
        try:
            return subprocess.check_output(args, cwd=repo_root, text=True).strip()
        except Exception:
            return "unknown"

    return {
        "git_branch": run(["git", "branch", "--show-current"]),
        "git_commit": run(["git", "rev-parse", "--short", "HEAD"]),
        "git_describe": run(["git", "describe", "--tags", "--always", "--dirty"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all miniSGLang comparison experiments.")
    parser.add_argument("--mode", required=True, help="Experiment mode label, e.g. main or zipcache.")
    parser.add_argument("--base-url", required=True, help="Running miniSGLang server base URL.")
    parser.add_argument("--log-root", type=Path, default=Path("experiment/logs"))
    parser.add_argument("--model", default="minisgl")
    parser.add_argument("--gpu-sample-interval", type=float, default=0.5)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--server-log", type=Path, default=None, help="Optional ZipCache server log.")
    parser.add_argument("--skip-server-check", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    exp_root = Path(__file__).resolve().parent
    run_dir = args.log_root / f"{now_stamp()}_{args.mode}"
    run_dir.mkdir(parents=True, exist_ok=True)

    server_check = (
        {"ok": True, "skipped": True}
        if args.skip_server_check
        else check_server(args.base_url)
    )
    if not server_check.get("ok"):
        write_json(run_dir / "server_check_failed.json", server_check)
        raise SystemExit(
            f"Server check failed for {args.base_url}. "
            f"Details saved to {run_dir / 'server_check_failed.json'}"
        )

    manifest: Dict[str, Any] = {
        "mode": args.mode,
        "base_url": args.base_url,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_dir": str(run_dir),
        "server_check": server_check,
        "experiments": EXPERIMENTS,
    }
    manifest.update(get_git_info(repo_root))
    write_json(run_dir / "manifest.json", manifest)

    all_results: List[Dict[str, Any]] = []
    for exp in EXPERIMENTS:
        name = exp["name"]
        result_file = run_dir / f"{name}.jsonl"
        summary_file = run_dir / f"{name}_summary.json"
        log_file = run_dir / f"{name}.log"
        cmd = [
            sys.executable,
            str(exp_root / "bench_openai_stream.py"),
            "--base-url",
            args.base_url,
            "--dataset",
            str(exp_root / exp["dataset"]),
            "--output",
            str(result_file),
            "--summary",
            str(summary_file),
            "--concurrency",
            str(exp["concurrency"]),
            "--repeat",
            str(exp["repeat"]),
            "--model",
            args.model,
            "--timeout",
            str(args.timeout),
            "--gpu-sample-interval",
            str(args.gpu_sample_interval),
        ]
        if exp["max_tokens"] is not None:
            cmd.extend(["--max-tokens", str(exp["max_tokens"])])
        print(f"\n===== Running {name} ({args.mode}) =====")
        exit_code = run_command(cmd, log_file)
        if exit_code != 0:
            raise SystemExit(f"Experiment {name} failed, see {log_file}")
        summary = load_json(summary_file)
        row_result = {
            "name": name,
            "description": exp["description"],
            "result_file": str(result_file),
            "summary_file": str(summary_file),
            "log_file": str(log_file),
            "summary": summary,
            "compact_summary": summarize_for_report(summary),
        }
        if exp.get("evaluate") == "gsm8k":
            eval_file = run_dir / f"{name}_eval.json"
            eval_cmd = [
                sys.executable,
                str(exp_root / "evaluate_correctness.py"),
                "--dataset",
                str(exp_root / exp["dataset"]),
                "--results",
                str(result_file),
                "--output",
                str(eval_file),
            ]
            print(f"\n===== Evaluating {name} correctness =====")
            eval_exit = run_command(eval_cmd, run_dir / f"{name}_eval.log")
            if eval_exit == 0 and eval_file.exists():
                row_result["eval_file"] = str(eval_file)
                row_result["eval_summary"] = load_json(eval_file)
        all_results.append(row_result)

    zipcache_stats = None
    if args.server_log is not None:
        zipcache_output = run_dir / "zipcache_stats_summary.json"
        cmd = [
            sys.executable,
            str(exp_root / "parse_zipcache_log.py"),
            "--log",
            str(args.server_log),
            "--output",
            str(zipcache_output),
        ]
        print("\n===== Parsing ZipCache stats =====")
        exit_code = run_command(cmd, run_dir / "parse_zipcache_log.log")
        if exit_code == 0 and zipcache_output.exists():
            zipcache_stats = load_json(zipcache_output)

    final_summary = {
        "mode": args.mode,
        "base_url": args.base_url,
        "run_dir": str(run_dir),
        "manifest": manifest,
        "experiments": all_results,
        "zipcache_stats": zipcache_stats,
    }
    write_json(run_dir / "all_results_summary.json", final_summary)
    write_markdown_report(
        run_dir / "report.md",
        mode=args.mode,
        base_url=args.base_url,
        run_dir=run_dir,
        manifest=manifest,
        results=all_results,
        zipcache_stats=zipcache_stats,
    )
    print(f"\nAll experiments finished. Report: {run_dir / 'report.md'}")


if __name__ == "__main__":
    main()
