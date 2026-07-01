#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence


LOGGER = logging.getLogger("download_eval_datasets")

CMMLU_REPOS = ("haonan-li/cmmlu", "lmlmcat/cmmlu", "XiaHan19/cmmlu")
LONGBENCH_REPOS = ("THUDM/LongBench", "zai-org/LongBench", "yanbingzheng/LongBench")
RULER_GIT_URL = "https://github.com/NVIDIA/RULER.git"

WORD_BANK = (
    "attention",
    "cache",
    "prefix",
    "request",
    "decode",
    "prefill",
    "latency",
    "throughput",
    "reasoning",
    "context",
    "retrieval",
    "evidence",
    "memory",
    "compression",
    "quantization",
    "scheduler",
    "radix",
    "token",
    "sequence",
    "benchmark",
    "analysis",
    "answer",
    "instruction",
    "document",
    "summary",
    "calculation",
    "constraint",
    "comparison",
    "experiment",
    "performance",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare evaluation datasets and synthetic workloads for miniSGLang "
            "KV cache compression experiments."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd() / "experiment",
        help="Dataset root directory. Default: current working directory / experiment.",
    )

    parser.add_argument("--skip-hf", action="store_true", help="Skip all Hugging Face datasets.")
    parser.add_argument("--skip-gsm8k", action="store_true", help="Skip GSM8K.")
    parser.add_argument("--skip-cmmlu", action="store_true", help="Skip CMMLU.")
    parser.add_argument("--skip-longbench", action="store_true", help="Skip LongBench.")
    parser.add_argument("--skip-ruler", action="store_true", help="Skip NVIDIA RULER clone/setup.")
    parser.add_argument("--skip-synthetic", action="store_true", help="Skip synthetic workloads.")
    parser.add_argument(
        "--ruler-helper-timeout",
        type=int,
        default=300,
        help="Timeout in seconds for each optional RULER data helper script.",
    )

    parser.add_argument(
        "--cmmlu-max-configs",
        type=int,
        default=None,
        help="Only download the first N CMMLU configs/subjects.",
    )
    parser.add_argument(
        "--longbench-max-configs",
        type=int,
        default=None,
        help="Only download the first N LongBench configs/tasks.",
    )
    parser.add_argument(
        "--no-longbench-e",
        action="store_true",
        help="Skip LongBench-E tasks or task names ending with _e/-e.",
    )

    parser.add_argument("--gsp-groups", type=int, default=64)
    parser.add_argument("--gsp-prompts-per-group", type=int, default=16)
    parser.add_argument("--gsp-prefix-len-words", type=int, default=2048)
    parser.add_argument("--gsp-question-len-words", type=int, default=128)
    parser.add_argument("--gsp-output-len", type=int, default=256)

    parser.add_argument("--random-num-prompts", type=int, default=1024)
    parser.add_argument("--random-min-input-len", type=int, default=100)
    parser.add_argument("--random-max-input-len", type=int, default=1024)
    parser.add_argument("--random-min-output-len", type=int, default=100)
    parser.add_argument("--random-max-output-len", type=int, default=1024)
    parser.add_argument("--random-vocab-size", type=int, default=10000)

    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument(
        "--overwrite-synthetic",
        action="store_true",
        help="Regenerate synthetic JSONL files even when they already exist.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def has_existing_payload(path: Path) -> bool:
    return path.exists() and any(path.iterdir()) if path.is_dir() else path.exists()


def sanitize_name(name: str) -> str:
    return name.replace("/", "__").replace("\\", "__").replace(" ", "_")


def limited(items: Sequence[str], limit: int | None) -> Sequence[str]:
    if limit is None or limit <= 0:
        return items
    return items[:limit]


def import_datasets_module() -> Any:
    try:
        import datasets
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'datasets'. Install with: "
            "python -m pip install -U datasets huggingface_hub pandas pyarrow tqdm"
        ) from exc
    return datasets


def get_config_names(repo_id: str) -> list[str]:
    datasets = import_datasets_module()
    try:
        return list(datasets.get_dataset_config_names(repo_id, trust_remote_code=True))
    except TypeError:
        return list(datasets.get_dataset_config_names(repo_id))


def load_dataset(repo_id: str, *args: Any, **kwargs: Any) -> Any:
    datasets = import_datasets_module()
    try:
        return datasets.load_dataset(repo_id, *args, trust_remote_code=True, **kwargs)
    except TypeError:
        return datasets.load_dataset(repo_id, *args, **kwargs)


def save_dataset_to_disk(dataset: Any, output_dir: Path) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_dir))


def write_metadata(output_dir: Path, metadata: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "_minisgl_dataset_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def download_gsm8k(root: Path) -> bool:
    output_dir = root / "gsm8k" / "main"
    if has_existing_payload(output_dir):
        LOGGER.info("GSM8K already exists, skip: %s", output_dir)
        return True

    LOGGER.info("Downloading GSM8K openai/gsm8k config=main -> %s", output_dir)
    try:
        dataset = load_dataset("openai/gsm8k", "main")
        save_dataset_to_disk(dataset, output_dir)
        write_metadata(output_dir, {"repo": "openai/gsm8k", "config": "main", "splits": "all"})
        return True
    except Exception:
        LOGGER.exception("Failed to download GSM8K openai/gsm8k main")
        return False


def choose_first_working_repo(repos: Iterable[str]) -> tuple[str, list[str]] | None:
    for repo in repos:
        try:
            configs = get_config_names(repo)
        except Exception:
            LOGGER.exception("Failed to list configs for %s", repo)
            continue
        if configs:
            LOGGER.info("Using dataset repo %s with %d configs", repo, len(configs))
            return repo, configs
        LOGGER.warning("Dataset repo %s returned no configs", repo)
    return None


def download_cmmlu(root: Path, max_configs: int | None) -> bool:
    selected = choose_first_working_repo(CMMLU_REPOS)
    if selected is None:
        LOGGER.error("Failed to locate a usable CMMLU repo from: %s", ", ".join(CMMLU_REPOS))
        return False

    repo, configs = selected
    ok = True
    for config in limited(configs, max_configs):
        output_dir = root / "cmmlu" / sanitize_name(config)
        if has_existing_payload(output_dir):
            LOGGER.info("CMMLU config %s already exists, skip: %s", config, output_dir)
            continue
        LOGGER.info("Downloading CMMLU %s config=%s -> %s", repo, config, output_dir)
        try:
            dataset = load_dataset(repo, config)
            save_dataset_to_disk(dataset, output_dir)
            write_metadata(output_dir, {"repo": repo, "config": config, "splits": "all"})
        except Exception:
            ok = False
            LOGGER.exception("Failed to download CMMLU %s config=%s", repo, config)
    return ok


def is_longbench_e_task(name: str) -> bool:
    lower = name.lower()
    return "longbench-e" in lower or lower.endswith("_e") or lower.endswith("-e")


def download_longbench(root: Path, max_configs: int | None, skip_longbench_e: bool) -> bool:
    selected = choose_first_working_repo(LONGBENCH_REPOS)
    if selected is None:
        LOGGER.error(
            "Failed to locate a usable LongBench repo from: %s",
            ", ".join(LONGBENCH_REPOS),
        )
        return False

    repo, configs = selected
    if skip_longbench_e:
        before = len(configs)
        configs = [config for config in configs if not is_longbench_e_task(config)]
        LOGGER.info("Filtered LongBench-E tasks: %d -> %d configs", before, len(configs))

    ok = True
    for config in limited(configs, max_configs):
        output_dir = root / "longbench" / sanitize_name(config)
        if has_existing_payload(output_dir):
            LOGGER.info("LongBench task %s already exists, skip: %s", config, output_dir)
            continue
        LOGGER.info("Downloading LongBench %s config=%s split=test -> %s", repo, config, output_dir)
        try:
            dataset = load_dataset(repo, config, split="test")
            save_dataset_to_disk(dataset, output_dir)
            write_metadata(output_dir, {"repo": repo, "config": config, "split": "test"})
        except Exception:
            ok = False
            LOGGER.exception("Failed to download LongBench %s config=%s split=test", repo, config)
    return ok


def run_optional_command(
    cmd: list[str],
    cwd: Path,
    description: str,
    *,
    timeout: int | None = None,
) -> bool:
    LOGGER.info("Running %s: %s", description, " ".join(cmd))
    try:
        subprocess.run(cmd, cwd=str(cwd), check=True, timeout=timeout)
        return True
    except FileNotFoundError:
        LOGGER.warning("Command not found while running %s: %s", description, cmd[0])
    except subprocess.CalledProcessError as exc:
        LOGGER.warning("%s failed with exit code %s", description, exc.returncode)
    except subprocess.TimeoutExpired:
        LOGGER.warning("%s timed out after %s seconds", description, timeout)
    return False


def prepare_ruler(root: Path, *, helper_timeout: int) -> bool:
    ruler_dir = root / "ruler" / "NVIDIA_RULER"
    ruler_dir.parent.mkdir(parents=True, exist_ok=True)

    if ruler_dir.exists():
        LOGGER.info("RULER repo already exists, skip clone: %s", ruler_dir)
    else:
        LOGGER.info("Cloning NVIDIA RULER -> %s", ruler_dir)
        try:
            subprocess.run(["git", "clone", RULER_GIT_URL, str(ruler_dir)], check=True)
        except Exception:
            LOGGER.exception("Failed to clone RULER from %s", RULER_GIT_URL)
            return False

    data_scripts = [
        (
            ruler_dir / "scripts" / "data" / "synthetic" / "json" / "download_paulgraham_essay.py",
            [sys.executable],
            "RULER Paul Graham essay download",
        ),
        (
            ruler_dir / "scripts" / "data" / "synthetic" / "json" / "download_qa_dataset.sh",
            ["bash"],
            "RULER QA dataset download",
        ),
    ]

    found_any = False
    for script_path, prefix, description in data_scripts:
        if not script_path.exists():
            LOGGER.warning(
                "RULER helper script not found: %s. Please check the current RULER README.",
                script_path,
            )
            continue
        found_any = True
        run_optional_command(
            prefix + [str(script_path.resolve())],
            cwd=ruler_dir,
            description=description,
            timeout=helper_timeout,
        )

    if not found_any:
        LOGGER.warning(
            "No known RULER data helper scripts were found. "
            "The RULER repository layout may have changed; please inspect its README manually."
        )
    return True


def make_words(rng: random.Random, count: int, *, group_id: int, salt: str) -> str:
    words: list[str] = []
    for idx in range(count):
        bank_word = WORD_BANK[(idx + group_id * 7 + rng.randrange(len(WORD_BANK))) % len(WORD_BANK)]
        words.append(f"{bank_word}_{salt}_{idx % 97}")
    return " ".join(words)


def make_generated_shared_prefix(
    root: Path,
    *,
    groups: int,
    prompts_per_group: int,
    prefix_len_words: int,
    question_len_words: int,
    output_len: int,
    seed: int,
    overwrite: bool,
) -> Path:
    output_path = root / "synthetic" / "generated_shared_prefix.jsonl"
    if output_path.exists() and not overwrite:
        LOGGER.info("generated-shared-prefix already exists, skip: %s", output_path)
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    LOGGER.info("Generating shared-prefix workload -> %s", output_path)

    row_id = 0
    with output_path.open("w", encoding="utf-8") as f:
        for group_id in range(groups):
            prefix_body = make_words(
                rng,
                prefix_len_words,
                group_id=group_id,
                salt=f"g{group_id}",
            )
            shared_prefix = (
                f"System prompt group {group_id}. You are evaluating long-context KV cache "
                f"reuse. Keep all facts from this shared prefix available. {prefix_body}"
            )
            for prompt_idx in range(prompts_per_group):
                question_body = make_words(
                    rng,
                    question_len_words,
                    group_id=group_id + prompt_idx,
                    salt=f"q{prompt_idx}",
                )
                prompt = (
                    f"{shared_prefix}\n\n"
                    f"Question {prompt_idx}: use the shared context and answer the following "
                    f"request. {question_body}"
                )
                row = {
                    "id": row_id,
                    "group_id": group_id,
                    "prompt": prompt,
                    "input_len_words_approx": len(prompt.split()),
                    "output_len": output_len,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                row_id += 1
    LOGGER.info("Generated %d shared-prefix prompts", row_id)
    return output_path


def make_random_token_ids(
    root: Path,
    *,
    num_prompts: int,
    min_input_len: int,
    max_input_len: int,
    min_output_len: int,
    max_output_len: int,
    vocab_size: int,
    seed: int,
    overwrite: bool,
) -> Path:
    output_path = root / "synthetic" / "random_token_ids.jsonl"
    if output_path.exists() and not overwrite:
        LOGGER.info("random token ids already exists, skip: %s", output_path)
        return output_path

    if min_input_len <= 0 or max_input_len < min_input_len:
        raise ValueError("Invalid random input length range")
    if min_output_len <= 0 or max_output_len < min_output_len:
        raise ValueError("Invalid random output length range")
    if vocab_size <= 0:
        raise ValueError("random-vocab-size must be positive")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed + 1)
    LOGGER.info("Generating random token id workload -> %s", output_path)
    with output_path.open("w", encoding="utf-8") as f:
        for row_id in range(num_prompts):
            input_len = rng.randint(min_input_len, max_input_len)
            output_len = rng.randint(min_output_len, max_output_len)
            token_ids = [rng.randrange(vocab_size) for _ in range(input_len)]
            row = {
                "id": row_id,
                "prompt_token_ids": token_ids,
                "input_len": input_len,
                "output_len": output_len,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    LOGGER.info("Generated %d random-token-id prompts", num_prompts)
    return output_path


def prepare_synthetic(root: Path, args: argparse.Namespace) -> bool:
    try:
        make_generated_shared_prefix(
            root,
            groups=args.gsp_groups,
            prompts_per_group=args.gsp_prompts_per_group,
            prefix_len_words=args.gsp_prefix_len_words,
            question_len_words=args.gsp_question_len_words,
            output_len=args.gsp_output_len,
            seed=args.seed,
            overwrite=args.overwrite_synthetic,
        )
        make_random_token_ids(
            root,
            num_prompts=args.random_num_prompts,
            min_input_len=args.random_min_input_len,
            max_input_len=args.random_max_input_len,
            min_output_len=args.random_min_output_len,
            max_output_len=args.random_max_output_len,
            vocab_size=args.random_vocab_size,
            seed=args.seed,
            overwrite=args.overwrite_synthetic,
        )
        return True
    except Exception:
        LOGGER.exception("Failed to generate synthetic workloads")
        return False


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    root = args.root.expanduser()
    root.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Dataset root: %s", root)

    ok = True
    if args.skip_hf:
        LOGGER.info("Skipping all Hugging Face datasets because --skip-hf is set")
    else:
        if args.skip_gsm8k:
            LOGGER.info("Skipping GSM8K")
        else:
            ok = download_gsm8k(root) and ok

        if args.skip_cmmlu:
            LOGGER.info("Skipping CMMLU")
        else:
            ok = download_cmmlu(root, args.cmmlu_max_configs) and ok

        if args.skip_longbench:
            LOGGER.info("Skipping LongBench")
        else:
            ok = download_longbench(root, args.longbench_max_configs, args.no_longbench_e) and ok

    if args.skip_ruler:
        LOGGER.info("Skipping RULER")
    else:
        ok = prepare_ruler(root, helper_timeout=args.ruler_helper_timeout) and ok

    if args.skip_synthetic:
        LOGGER.info("Skipping synthetic workloads")
    else:
        ok = prepare_synthetic(root, args) and ok

    if ok:
        LOGGER.info("Dataset preparation finished")
        return 0
    LOGGER.error("Dataset preparation finished with errors")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
