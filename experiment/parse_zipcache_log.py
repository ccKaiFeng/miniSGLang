#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, List


STATS_RE = re.compile(r"\[ZipCacheV1\] stats:\s*(\{.*\})")


def parse_stats(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            match = STATS_RE.search(line)
            if not match:
                continue
            raw = match.group(1)
            try:
                rows.append(ast.literal_eval(raw))
            except Exception:
                try:
                    rows.append(json.loads(raw))
                except Exception:
                    pass
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse [ZipCacheV1] stats from server log.")
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows = parse_stats(args.log)
    if not rows:
        summary = {"num_stats": 0, "message": "No [ZipCacheV1] stats found."}
    else:
        last = rows[-1]
        ratios = [
            float(r.get("active_compression_ratio", 0))
            for r in rows
            if r.get("active_compression_ratio") is not None
        ]
        summary = {
            "num_stats": len(rows),
            "last": last,
            "max_active_compression_ratio": max(ratios) if ratios else 0,
            "last_active_compression_ratio": last.get("active_compression_ratio", 0),
            "max_active_original_estimated_bytes": max(
                int(r.get("active_original_estimated_bytes", 0)) for r in rows
            ),
            "max_active_compressed_estimated_bytes": max(
                int(r.get("active_compressed_estimated_bytes", 0)) for r in rows
            ),
            "max_num_compressions": max(int(r.get("num_compressions", 0)) for r in rows),
            "max_num_decompressions": max(int(r.get("num_decompressions", 0)) for r in rows),
        }

    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
