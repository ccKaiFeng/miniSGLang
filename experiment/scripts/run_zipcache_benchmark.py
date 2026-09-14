"""Run isolated Main/ZipCache workloads and fail promptly on server errors.

Uses the active Python environment. Source snapshots avoid switching the user's
checkout; process groups restrict cleanup to children created by this runner.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKLOADS = [
    ("public_shared_prefix_serial", 64, 192),
    ("longbench_long_context_pressure", 32, 32),
    ("gsm8k_public_correctness", 64, 64),
]


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def stop_group(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        # Children may survive their group leader's exit.
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def check_server(proc: subprocess.Popen, log: Path) -> None:
    if proc.poll() is not None:
        raise RuntimeError(f"Server exited ({proc.returncode}); see {log}")
    content = log.read_text(errors="replace")
    if any(
        marker in content
        for marker in ("Traceback (most recent call last)", "CUDA out of memory", "AssertionError:")
    ):
        raise RuntimeError(f"Server exception detected; see {log}")


def snapshot(ref: str, destination: Path) -> tuple[Path, str]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{ref}^{{commit}}"], cwd=ROOT, text=True
    ).strip()
    destination.mkdir()
    archive = destination / "source.tar"
    subprocess.run(
        ["git", "archive", "--output", str(archive), commit, "python"], cwd=ROOT, check=True
    )
    subprocess.run(["tar", "-xf", str(archive), "-C", str(destination)], check=True)
    archive.unlink()
    return destination, commit


def validate_results(directory: Path, expected: int, name: str) -> None:
    summaries = list(directory.glob("*/all_results_summary.json"))
    if len(summaries) != 1:
        raise RuntimeError(f"Missing or ambiguous result summary: {directory}")
    rows = json.loads(summaries[0].read_text())["experiments"]
    if len(rows) != 1 or rows[0]["name"] != name:
        raise RuntimeError("Unexpected workload result")
    row = rows[0]
    summary = row["summary"]
    if (summary["num_requests"], summary["num_ok"], summary["num_failed"]) != (
        expected,
        expected,
        0,
    ):
        raise RuntimeError(f"Incomplete/failed requests: {summary}")
    if name == "gsm8k_public_correctness":
        if row.get("eval_summary", {}).get("num_judged") != expected:
            raise RuntimeError("Missing or incomplete GSM8K evaluation")


def run_workload(args, source, metadata, directory, name, limit, expected, pool_mb):
    directory.mkdir()
    server_log = directory / "server.log"
    metadata_file = directory / "server_metadata.json"
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "minisgl",
        "--model-path",
        args.model,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--cache-type",
        "radix",
        "--max-running-requests",
        "8",
        "--max-prefill-length",
        "4096",
    ]
    if pool_mb is None:
        cmd += ["--memory-ratio", "0.52"]
    else:
        cmd += [
            "--enable-zipcache-v3",
            "--zipcache-v3-normal-pool-pages",
            "94000",
            "--zipcache-v3-compressed-pool-mb",
            str(pool_mb),
            "--zipcache-v3-q4-pool-ratio",
            "0.66",
            "--zipcache-v3-q2-pool-ratio",
            "0.22",
            "--zipcache-v3-scale-pool-ratio",
            "0.09",
            "--zipcache-v3-ids-pool-ratio",
            "0.03",
            "--zipcache-stats-interval",
            "10",
        ]
    write_json(metadata_file, dict(metadata, command=cmd, source=str(source)))
    write_json(directory / "status.json", {"status": "running", "workload": name})
    env = dict(os.environ, PYTHONPATH=str(source / "python"), PYTHONUNBUFFERED="1")
    server = client = None
    try:
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", args.port))
        with server_log.open("w") as output:
            server = subprocess.Popen(
                cmd,
                cwd=source,
                env=env,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        base_url = f"http://127.0.0.1:{args.port}"
        deadline = time.monotonic() + args.startup_timeout
        while True:
            check_server(server, server_log)
            try:
                with urllib.request.urlopen(base_url + "/v1", timeout=2) as response:
                    ready = json.load(response).get("status") == "ok"
                if ready:
                    break
            except (OSError, ValueError):
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError("Server startup timed out")
            time.sleep(1)
        client_cmd = [
            sys.executable,
            "-u",
            str(ROOT / "experiment/run_all_experiments.py"),
            "--mode",
            directory.name,
            "--base-url",
            base_url,
            "--server-log",
            str(server_log),
            "--server-metadata",
            str(metadata_file),
            "--only",
            name,
            "--max-samples",
            str(limit),
            "--log-root",
            str(directory),
            "--timeout",
            str(args.request_timeout),
        ]
        with (directory / "client.log").open("w") as output:
            client = subprocess.Popen(
                client_cmd,
                cwd=ROOT,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        deadline = time.monotonic() + args.workload_timeout
        while client.poll() is None:
            check_server(server, server_log)
            if time.monotonic() >= deadline:
                raise TimeoutError("Workload timed out")
            time.sleep(1)
        check_server(server, server_log)
        if client.returncode:
            raise RuntimeError(f"Benchmark client exited ({client.returncode})")
        validate_results(directory, expected, name)
        write_json(directory / "status.json", {"status": "complete", "workload": name})
    except BaseException as exc:
        write_json(directory / "status.json", {"status": "failed", "error": str(exc)})
        raise
    finally:
        try:
            stop_group(client)
        finally:
            stop_group(server)


def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model", nargs="?", default="/root/autodl-tmp/modelscope-cache/models/Qwen/Qwen3-8B"
    )
    parser.add_argument("zipcache_ref", nargs="?", help="Omit to test working-tree fixes")
    parser.add_argument("--main-ref", default="main")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--startup-timeout", type=float, default=300)
    parser.add_argument("--request-timeout", type=float, default=120)
    parser.add_argument("--workload-timeout", type=float, default=3600)
    parser.add_argument("--log-root", type=Path, default=ROOT / "experiment/logs")
    args = parser.parse_args()
    args.model = str(Path(args.model).resolve())
    directory = args.log_root.resolve() / time.strftime("bench_8b_%Y%m%d_%H%M%S")
    directory.mkdir(parents=True)
    status = {
        "status": "running",
        "completed": [],
        "model": args.model,
        "isolation": "fresh_server_per_workload",
    }
    write_json(directory / "status.json", status)
    try:
        if not Path(args.model).is_dir():
            raise FileNotFoundError(f"Model directory not found: {args.model}")
        hardware = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"],
            text=True,
        )
        (directory / "hardware.csv").write_text(hardware)
        with tempfile.TemporaryDirectory(prefix="minisgl-bench-") as temporary:
            baseline, main_commit = snapshot(args.main_ref, Path(temporary) / "main")
            if args.zipcache_ref:
                zip_source, zip_commit = snapshot(args.zipcache_ref, Path(temporary) / "zipcache")
            else:
                zip_source = Path(temporary) / "zipcache"
                shutil.copytree(
                    ROOT / "python",
                    zip_source / "python",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                )
                # Preserve untracked source files too; git diff alone misses them.
                with tarfile.open(directory / "zipcache_source.tar.gz", "w:gz") as archive:
                    archive.add(zip_source / "python", arcname="python")
                zip_commit = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip()
                diff = subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT, text=True)
                (directory / "working_tree.patch").write_text(diff)
            metadata = {"git_commit": zip_commit, "git_branch": args.zipcache_ref or "working-tree"}
            stages = [
                (
                    "main",
                    baseline,
                    {"git_commit": main_commit, "git_branch": args.main_ref},
                    None,
                    WORKLOADS,
                ),
                ("zipcache_13600mb", zip_source, metadata, 13600, WORKLOADS),
                ("zipcache_3400mb", zip_source, metadata, 3400, WORKLOADS[:1]),
                ("zipcache_6800mb", zip_source, metadata, 6800, WORKLOADS[:1]),
            ]
            for label, source, meta, pool, workloads in stages:
                for name, limit, expected in workloads:
                    stage = f"{label}_{name}"
                    print(f"Running {stage}; logs: {directory / stage}", flush=True)
                    run_workload(args, source, meta, directory / stage, name, limit, expected, pool)
                    status["completed"].append(stage)
                    write_json(directory / "status.json", status)
        status["status"] = "complete"
        (directory / "summary.txt").write_text(
            "BENCHMARK COMPLETE\n" + "\n".join(status["completed"]) + "\n"
        )
    except BaseException as exc:
        status.update(status="failed", error=str(exc))
        raise
    finally:
        write_json(directory / "status.json", status)
        print(f"Benchmark {status['status']}: {directory}", flush=True)


if __name__ == "__main__":
    main()
