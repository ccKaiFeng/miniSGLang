#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, List


STATS_RE = re.compile(r"\[(ZipCacheV[12])\] stats:\s*(\{.*\})")


def parse_stats(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            match = STATS_RE.search(line)
            if not match:
                continue
            version = match.group(1)
            raw = match.group(2)
            try:
                row = ast.literal_eval(raw)
            except Exception:
                try:
                    row = json.loads(raw)
                except Exception:
                    continue
            if isinstance(row, dict):
                row["_zipcache_version"] = version
                rows.append(row)
    return rows


def _first_number(row: Dict[str, Any], keys: List[str], default: float = 0.0) -> float:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return float(value)
    return default


def _first_int(row: Dict[str, Any], keys: List[str], default: int = 0) -> int:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return int(value)
    return default


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse [ZipCacheV1]/[ZipCacheV2] stats from server log.")
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows = parse_stats(args.log)
    if not rows:
        summary = {"num_stats": 0, "message": "No [ZipCacheV1]/[ZipCacheV2] stats found."}
    else:
        last = rows[-1]
        versions = sorted({str(r.get("_zipcache_version", "unknown")) for r in rows})
        ratios = [
            _first_number(
                r,
                [
                    "active_compression_ratio",
                    "active_storage_compression_ratio",
                    "active_estimated_compression_ratio",
                ],
            )
            for r in rows
            if any(
                r.get(k) is not None
                for k in (
                    "active_compression_ratio",
                    "active_storage_compression_ratio",
                    "active_estimated_compression_ratio",
                )
            )
        ]
        summary = {
            "num_stats": len(rows),
            "versions": versions,
            "last": last,
            "max_active_compression_ratio": max(ratios) if ratios else 0,
            "last_active_compression_ratio": _first_number(
                last,
                [
                    "active_compression_ratio",
                    "active_storage_compression_ratio",
                    "active_estimated_compression_ratio",
                ],
            ),
            "max_active_original_estimated_bytes": max(
                int(r.get("active_original_estimated_bytes", 0)) for r in rows
            ),
            "max_active_compressed_estimated_bytes": max(
                _first_int(
                    r,
                    [
                        "active_compressed_estimated_bytes",
                        "active_compressed_storage_bytes",
                        "active_compressed_estimated_bytes_4bit",
                    ],
                )
                for r in rows
            ),
            "max_num_compressions_or_demotions": max(
                _first_int(r, ["num_compressions", "num_demotions"]) for r in rows
            ),
            "max_num_decompressions_or_restores": max(
                _first_int(r, ["num_decompressions", "num_restore_success"]) for r in rows
            ),
        }

    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
