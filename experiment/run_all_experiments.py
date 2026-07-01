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
        "name": "gsm8k_public_correctness",
        "dataset": "workloads/gsm8k_public_correctness.jsonl",
        "concurrency": 1,
        "repeat": 1,
        "max_tokens": 1024,
        "ignore_eos": False,
        "description": "GSM8K 公开测试集数学推理正确性测试，使用标准数字答案自动计算 accuracy。",
        "evaluate": "auto",
    },
    {
        "name": "cmmlu_public_correctness",
        "dataset": "workloads/cmmlu_public_correctness.jsonl",
        "concurrency": 1,
        "repeat": 1,
        "max_tokens": 128,
        "ignore_eos": False,
        "description": "CMMLU 公开中文多学科选择题正确性测试，自动抽取 A/B/C/D 选项并计算 accuracy。",
        "evaluate": "auto",
    },
    {
        "name": "longbench_public_qa",
        "dataset": "workloads/longbench_public_qa.jsonl",
        "concurrency": 4,
        "repeat": 1,
        "max_tokens": 512,
        "ignore_eos": False,
        "description": "LongBench 公开长上下文问答任务，用于同时观察长输入性能和近似 answer contains 正确性。",
        "evaluate": "auto",
    },
    {
        "name": "longbench_long_context_pressure",
        "dataset": "workloads/longbench_long_context_pressure.jsonl",
        "concurrency": 8,
        "repeat": 1,
        "max_tokens": 256,
        "description": "LongBench 派生长上下文压力测试，优先选择较长样本，用于压高 KV cache 显存和 prefill 压力。",
    },
    {
        "name": "public_shared_prefix",
        "dataset": "workloads/public_shared_prefix.jsonl",
        "concurrency": 4,
        "repeat": 2,
        "max_tokens": 160,
        "description": "由 LongBench 公开长上下文派生的共享前缀 workload，用于观察 radix/prefix cache 与 ZipCache restore 命中。",
    },
    {
        "name": "public_shared_prefix_serial",
        "dataset": "workloads/public_shared_prefix.jsonl",
        "concurrency": 1,
        "repeat": 3,
        "max_tokens": 128,
        "description": "顺序重复共享前缀测试，用于提高 finished prefix demote 后再次命中 compressed entry 的概率。",
    },
    {
        "name": "ruler_squad_qa",
        "dataset": "workloads/ruler_squad_qa.jsonl",
        "concurrency": 2,
        "repeat": 1,
        "max_tokens": 256,
        "ignore_eos": False,
        "description": "RULER helper 下载得到的 SQuAD 问答数据，用于长上下文检索类正确性近似评估。",
        "evaluate": "auto",
    },
    {
        "name": "synthetic_shared_prefix",
        "dataset": "workloads/synthetic_shared_prefix.jsonl",
        "concurrency": 8,
        "repeat": 1,
        "max_tokens": 256,
        "description": "本地生成的 generated-shared-prefix 压测负载，保留用于强 prefix cache 压力测试。",
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


def select_experiments(only: str | None) -> List[Dict[str, Any]]:
    if only is None or not only.strip():
        return EXPERIMENTS
    wanted = {name.strip() for name in only.split(",") if name.strip()}
    selected = [exp for exp in EXPERIMENTS if exp["name"] in wanted]
    missing = sorted(wanted - {exp["name"] for exp in selected})
    if missing:
        names = ", ".join(exp["name"] for exp in EXPERIMENTS)
        raise SystemExit(f"Unknown experiment(s): {missing}. Available: {names}")
    return selected


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
        "finish_reasons",
        "num_reached_max_tokens",
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
        "| experiment | ok/total | maxed | rps | chunks/s | ttft avg | ttft p90 | e2e avg | tpot avg | gpu max MB |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        summary = row["summary"]
        ok_total = f"{summary.get('num_ok', 0)}/{summary.get('num_requests', 0)}"
        lines.append(
            "| {name} | {ok_total} | {maxed} | {rps:.4g} | {cps:.4g} | {ttft:.4g} | {ttft90:.4g} | {e2e:.4g} | {tpot:.4g} | {gpu} |".format(
                name=row["name"],
                ok_total=ok_total,
                maxed=summary.get("num_reached_max_tokens", "n/a"),
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
    parser.add_argument(
        "--only",
        default=None,
        help="Comma-separated experiment names to run. Default: run all public workloads.",
    )
    parser.add_argument("--list-experiments", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    exp_root = Path(__file__).resolve().parent
    experiments = select_experiments(args.only)
    if args.list_experiments:
        for exp in experiments:
            print(f"{exp['name']}: {exp['description']}")
        return

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
        "experiments": experiments,
    }
    manifest.update(get_git_info(repo_root))
    write_json(run_dir / "manifest.json", manifest)

    all_results: List[Dict[str, Any]] = []
    for exp in experiments:
        name = exp["name"]
        dataset_path = exp_root / exp["dataset"]
        if not dataset_path.exists():
            raise SystemExit(
                f"Dataset for experiment {name} does not exist: {dataset_path}. "
                f"Run: python experiment/prepare_public_workloads.py"
            )
        result_file = run_dir / f"{name}.jsonl"
        summary_file = run_dir / f"{name}_summary.json"
        log_file = run_dir / f"{name}.log"
        cmd = [
            sys.executable,
            str(exp_root / "bench_openai_stream.py"),
            "--base-url",
            args.base_url,
            "--dataset",
            str(dataset_path),
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
        if exp.get("ignore_eos") is not None:
            cmd.extend(["--ignore-eos", "true" if exp["ignore_eos"] else "false"])
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
        if exp.get("evaluate") == "auto":
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
