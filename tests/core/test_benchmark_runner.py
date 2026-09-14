"""Verify completion markers reflect request/evaluation success, not process exit alone."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "benchmark_runner",
    Path(__file__).resolve().parents[2] / "experiment/scripts/run_zipcache_benchmark.py",
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def result(tmp_path, summary, evaluation=None):
    run = tmp_path / "run"
    run.mkdir()
    row = {"name": "gsm8k_public_correctness", "summary": summary}
    if evaluation is not None:
        row["eval_summary"] = evaluation
    (run / "all_results_summary.json").write_text(json.dumps({"experiments": [row]}))


@pytest.mark.parametrize(
    "summary,evaluation",
    [
        ({"num_requests": 64, "num_ok": 63, "num_failed": 1}, {"num_judged": 64}),
        ({"num_requests": 4, "num_ok": 4, "num_failed": 0}, {"num_judged": 4}),
        ({"num_requests": 64, "num_ok": 64, "num_failed": 0}, None),
    ],
)
def test_incomplete_experiment_rejected(tmp_path, summary, evaluation):
    result(tmp_path, summary, evaluation)
    with pytest.raises(RuntimeError):
        runner.validate_results(tmp_path, 64, "gsm8k_public_correctness")


def test_complete_experiment_accepted(tmp_path):
    result(tmp_path, {"num_requests": 64, "num_ok": 64, "num_failed": 0}, {"num_judged": 64})
    runner.validate_results(tmp_path, 64, "gsm8k_public_correctness")


def test_dead_scheduler_detected_while_frontend_alive(tmp_path):
    log = tmp_path / "server.log"
    log.write_text("Process minisgl-TP0-scheduler:\nTraceback (most recent call last):\n")
    frontend = SimpleNamespace(poll=lambda: None)
    with pytest.raises(RuntimeError, match="Server exception"):
        runner.check_server(frontend, log)


def test_pool_pressure_without_crash_is_not_fatal(tmp_path):
    log = tmp_path / "server.log"
    log.write_text("num_demote_rejected_pool_full: 100\n")
    runner.check_server(SimpleNamespace(poll=lambda: None), log)


@pytest.mark.parametrize("fail_at", [None, 5])
def test_suite_completion_requires_all_eight_workloads(tmp_path, monkeypatch, fail_at):
    monkeypatch.setattr(
        sys, "argv", ["runner", str(tmp_path), "fixed-ref", "--log-root", str(tmp_path)]
    )
    monkeypatch.setattr(runner, "snapshot", lambda ref, destination: (tmp_path, ref))
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **kw: "GPU,memory,driver\n")
    calls = []

    def workload(*args):
        calls.append(args)
        if len(calls) == fail_at:
            raise RuntimeError("scheduler failed")

    monkeypatch.setattr(runner, "run_workload", workload)
    # Do not alter the pytest process's signal handlers.
    monkeypatch.setattr(runner.signal, "signal", lambda *args: None)
    if fail_at is None:
        runner.main()
    else:
        with pytest.raises(RuntimeError, match="scheduler failed"):
            runner.main()
    suite = next(tmp_path.glob("bench_8b_*"))
    status = json.loads((suite / "status.json").read_text())
    assert status["status"] == ("complete" if fail_at is None else "failed")
    assert len(status["completed"]) == (8 if fail_at is None else fail_at - 1)
    assert (suite / "summary.txt").exists() == (fail_at is None)
