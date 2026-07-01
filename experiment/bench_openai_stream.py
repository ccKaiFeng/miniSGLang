#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List


def parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value!r}")


def load_dataset(
    path: Path,
    repeat: int,
    max_tokens_override: int | None,
    ignore_eos_override: bool | None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    expanded = []
    for r in range(repeat):
        for row in rows:
            item = dict(row)
            item["id"] = f"{row.get('id', 'req')}_r{r}"
            if max_tokens_override is not None:
                item["max_tokens"] = max_tokens_override
            elif "max_tokens" not in item and "output_len" in item:
                item["max_tokens"] = int(item["output_len"])
            if ignore_eos_override is not None:
                item["ignore_eos"] = ignore_eos_override
            expanded.append(item)
    return expanded


class GpuSampler:
    def __init__(self, interval: float):
        self.interval = interval
        self.samples: List[Dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self.interval <= 0:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def summary(self) -> Dict[str, Any]:
        values = [int(s["memory_used_mb"]) for s in self.samples if "memory_used_mb" in s]
        if not values:
            return {"gpu_samples": 0}
        return {
            "gpu_samples": len(values),
            "gpu_memory_used_mb_min": min(values),
            "gpu_memory_used_mb_max": max(values),
            "gpu_memory_used_mb_avg": statistics.mean(values),
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    stderr=subprocess.DEVNULL,
                    timeout=2,
                )
                values = [int(x.strip()) for x in out.splitlines() if x.strip()]
                for gpu_id, value in enumerate(values):
                    self.samples.append(
                        {"time": time.time(), "gpu_id": gpu_id, "memory_used_mb": value}
                    )
            except Exception:
                self.samples.append({"time": time.time(), "error": "nvidia-smi unavailable"})
                return
            self._stop.wait(self.interval)


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = int(round((len(values) - 1) * p))
    return values[idx]


def summarize(results: List[Dict[str, Any]], gpu_summary: Dict[str, Any]) -> Dict[str, Any]:
    ok = [r for r in results if r.get("ok")]
    failed = [r for r in results if not r.get("ok")]
    ttft = [r["ttft_s"] for r in ok if r.get("ttft_s") is not None]
    e2e = [r["e2e_s"] for r in ok]
    tpot = [r["tpot_s"] for r in ok if r.get("tpot_s") is not None]
    chunks = [r["output_chunks"] for r in ok]
    total_time = max((r["end_time"] for r in ok), default=0) - min(
        (r["start_time"] for r in ok), default=0
    )
    output_chunks = sum(chunks)
    summary = {
        "num_requests": len(results),
        "num_ok": len(ok),
        "num_failed": len(failed),
        "total_wall_time_s": total_time,
        "request_throughput_rps": len(ok) / total_time if total_time > 0 else 0.0,
        "output_chunk_throughput_cps": output_chunks / total_time if total_time > 0 else 0.0,
        "output_chunks": output_chunks,
        "ttft_avg_s": statistics.mean(ttft) if ttft else 0.0,
        "ttft_p50_s": percentile(ttft, 0.50),
        "ttft_p90_s": percentile(ttft, 0.90),
        "ttft_p99_s": percentile(ttft, 0.99),
        "e2e_avg_s": statistics.mean(e2e) if e2e else 0.0,
        "e2e_p50_s": percentile(e2e, 0.50),
        "e2e_p90_s": percentile(e2e, 0.90),
        "e2e_p99_s": percentile(e2e, 0.99),
        "tpot_avg_s": statistics.mean(tpot) if tpot else 0.0,
        "tpot_p50_s": percentile(tpot, 0.50),
        "tpot_p90_s": percentile(tpot, 0.90),
        "tpot_p99_s": percentile(tpot, 0.99),
        "finish_reasons": {},
        "num_reached_max_tokens": 0,
    }
    for row in ok:
        reason = str(row.get("finish_reason") or "unknown")
        summary["finish_reasons"][reason] = summary["finish_reasons"].get(reason, 0) + 1
        if int(row.get("output_chunks") or 0) >= int(row.get("max_tokens") or 0):
            summary["num_reached_max_tokens"] += 1
    summary.update(gpu_summary)
    return summary


def iter_sse_lines(response: Any) -> Iterable[str]:
    pending = b""
    while True:
        chunk = response.read(1)
        if not chunk:
            if pending:
                yield pending.decode("utf-8", errors="replace")
            return
        pending += chunk
        if pending.endswith(b"\n"):
            line = pending.decode("utf-8", errors="replace").strip()
            pending = b""
            if line:
                yield line


def run_one(base_url: str, model: str, item: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    start = time.perf_counter()
    start_wall = time.time()
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": item["prompt"]}],
        "max_tokens": int(item.get("max_tokens", 128)),
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "ignore_eos": bool(item.get("ignore_eos", True)),
        "stream": True,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    output_parts: List[str] = []
    chunk_times: List[float] = []
    first_token_time = None
    finish_reason = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            for line in iter_sse_lines(response):
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                choice = obj["choices"][0]
                if choice.get("finish_reason") is not None:
                    finish_reason = choice.get("finish_reason")
                delta = choice.get("delta", {})
                text = delta.get("content", "")
                if text:
                    now = time.perf_counter()
                    if first_token_time is None:
                        first_token_time = now
                    chunk_times.append(now - start)
                    output_parts.append(text)
        end = time.perf_counter()
        output_chunks = len(output_parts)
        return {
            "id": item.get("id"),
            "ok": True,
            "prompt_chars": len(item["prompt"]),
            "max_tokens": payload["max_tokens"],
            "ignore_eos": payload["ignore_eos"],
            "finish_reason": finish_reason,
            "output_chunks": output_chunks,
            "output_text": "".join(output_parts),
            "ttft_s": None if first_token_time is None else first_token_time - start,
            "e2e_s": end - start,
            "tpot_s": (
                None
                if first_token_time is None or output_chunks <= 1
                else (end - first_token_time) / (output_chunks - 1)
            ),
            "chunk_times_s": chunk_times,
            "start_time": start_wall,
            "end_time": time.time(),
        }
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, Exception) as e:
        return {
            "id": item.get("id"),
            "ok": False,
            "error": repr(e),
            "prompt_chars": len(item.get("prompt", "")),
            "max_tokens": payload["max_tokens"],
            "ignore_eos": payload["ignore_eos"],
            "start_time": start_wall,
            "end_time": time.time(),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark miniSGLang OpenAI streaming API.")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--model", default="minisgl")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument(
        "--ignore-eos",
        type=parse_bool,
        default=None,
        help="Override request ignore_eos. Use false for correctness tests and true for fixed-length performance tests.",
    )
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--gpu-sample-interval", type=float, default=0.0)
    args = parser.parse_args()

    items = load_dataset(args.dataset, args.repeat, args.max_tokens, args.ignore_eos)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    sampler = GpuSampler(args.gpu_sample_interval)
    sampler.start()
    results: List[Dict[str, Any]] = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [
                pool.submit(run_one, args.base_url, args.model, item, args.timeout)
                for item in items
            ]
            for i, fut in enumerate(concurrent.futures.as_completed(futures), start=1):
                result = fut.result()
                results.append(result)
                status = "ok" if result.get("ok") else "failed"
                print(f"[{i}/{len(futures)}] {result.get('id')} {status}")
    finally:
        sampler.stop()

    with args.output.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = summarize(results, sampler.summary())
    with args.summary.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
