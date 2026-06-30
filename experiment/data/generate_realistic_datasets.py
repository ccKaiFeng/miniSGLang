#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


BASE_BACKGROUND = """
你是一名严谨的 LLM serving 系统性能分析助手。请先给结论，再给依据，最后给实验建议。

固定背景资料：
miniSGLang 是一个用于大语言模型推理服务的实验性系统，请求路径包括 OpenAI 兼容 API、tokenizer、scheduler、engine、attention backend、paged KV cache、radix prefix cache 和 detokenizer。scheduler 会把请求拆分成 prefill 和 decode 阶段。prefill 负责处理输入 prompt 并生成历史 token 的 K/V cache；decode 每次生成一个新 token，并复用历史 K/V cache，避免重复计算历史 token 的 K/V。

KV cache 是推理服务中显存占用最大的部分之一。对于长上下文、多并发请求，KV cache 会随 batch 中活跃 token 数增长。paged KV cache 把 K/V 切成 page 或 slot 管理，page table 记录请求 token 到物理 KV page 的映射。radix prefix cache 用 token 前缀作为 key，当新请求和历史请求共享前缀时，可以复用已经计算好的 KV，减少 prefill 计算。

ZipCache v2 的目标是在不修改 CUDA attention kernel、不修改 attention 数学公式、不让 attention 直接读取 int4/int2 KV 的前提下，把不活跃或即将释放的 prefix KV cache 从 normal fp16/bf16 KV pool demote 到 GPU compressed pool。compressed pool 使用统一 4bit packed uint8 存储量化后的 K/V，同时保存 scale、token id、shape 和 metadata。未来如果 radix cache 再次命中 compressed prefix，ZipCache v2 会在 attention 前把 compressed KV restore 回 normal KV page，然后继续走原始 attention 路径。

实验时需要对比 main 分支和 ZipCache v2 分支。main 分支表示原始 miniSGLang，不带任何 ZipCache 参数。ZipCache v2 分支需要带 --enable-zipcache-v2，通常还需要用 --num-pages 限制 normal KV pool，为 compressed pool 留出显存。性能指标包括 TTFT、TPOT、E2E、request throughput、output chunks/s 和 GPU memory used。ZipCache v2 内部指标包括 num_demotions、num_compressed_hits、num_restore_attempts、num_restore_success、num_restore_fallback、active_storage_compression_ratio、compressed_pool_used_bytes 和 compressed_pool_utilization。

请注意：如果只看到 num_demotions 大于 0，而 num_compressed_hits 和 num_restore_success 等于 0，说明当前测试只验证了压缩保存，没有验证压缩缓存复用。要验证 restore，需要构造重复长前缀请求，让第一次请求结束后触发 demote，后续请求再次命中同一前缀。
""".strip()


LONG_CASES = [
    "分析 ZipCache v2 在长上下文共享前缀场景下可能降低显存占用的原因，并指出为什么 compressed pool 预分配过大也可能抵消收益。",
    "设计一个完整实验，对比 main 和 ZipCache v2 在 TTFT、TPOT、E2E、吞吐、显存峰值、压缩率和输出正确性上的差异。",
    "解释为什么不修改 attention kernel 时，压缩后的 KV cache 需要在 attention 前 restore 回 normal KV page，而不是让 attention 直接读取 packed 4bit 数据。",
    "分析 num_demotions、num_compressed_hits、num_restore_success 三个指标之间的关系，并说明如何根据它们判断 workload 是否有效。",
    "如果 ZipCache v2 的 active_storage_compression_ratio 接近 3.7，但 nvidia-smi 中显存峰值没有下降，请列出可能原因。",
    "如果 ZipCache v2 的 TTFT 明显高于 main，但 TPOT 接近 main，请分析可能的工程原因和下一步定位方法。",
    "从 scheduler 和 radix cache 的角度解释 compressed node 的命中路径，并说明为什么命中 token prefix 不等于 normal KV page 一定可直接复用。",
    "给出一个排查 restore 失败的 checklist，包括 radix node 状态、page 分配、pool 容量、shape/dtype 和 fallback 行为。",
    "分析将 compressed pool 设计成固定大小 GPU pool 的优缺点，并说明为什么需要用 --num-pages 给它预留显存。",
    "请说明在多请求并发下，prefill 和 decode 如何竞争 KV cache 显存，以及 ZipCache v2 demote 对这种竞争的影响。",
    "从正确性角度分析有损 KV 量化可能带来的输出差异，并说明如何区分可接受的语义差异和严重错误。",
    "请写一份实验报告摘要，总结 ZipCache v2 在长上下文场景下的预期收益、实际风险和后续优化方向。",
]


RESTORE_QUESTIONS = [
    "请判断当前测试是否足以验证 compressed KV restore，并给出更强 workload 的设计。",
    "请解释为什么顺序重复完全相同的长 prompt 比高并发同时请求更容易触发 restore。",
    "请分析如果 num_compressed_hits 仍然为 0，应该优先检查哪些代码路径。",
    "请说明 restore 成功后为什么仍然可能看不到明显吞吐提升。",
]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_long_context_rows() -> list[dict]:
    rows = []
    repeated_notes = "\n".join(
        [
            f"补充资料 {i}: 在真实线上服务中，用户请求可能包含长 system prompt、工具说明、历史对话、检索文档和代码片段。"
            f"这些内容会增加 prefill token 数和 KV cache 占用，也会放大 prefix cache 命中或失效对性能的影响。"
            for i in range(1, 11)
        ]
    )
    for i, question in enumerate(LONG_CASES, start=1):
        prompt = (
            f"{BASE_BACKGROUND}\n\n"
            f"{repeated_notes}\n\n"
            f"本次任务编号：LONG-{i:03d}\n"
            f"具体问题：{question}\n"
            "请输出一份结构化分析，至少包含：结论、关键指标、可能瓶颈、实验步骤、预期现象、异常排查。"
        )
        rows.append(
            {
                "id": f"realistic_long_{i:03d}",
                "prompt": prompt,
                "max_tokens": 512,
            }
        )
    return rows


def make_restore_rows() -> list[dict]:
    shared_prefix = (
        f"{BASE_BACKGROUND}\n\n"
        "下面是一段固定的长前缀，用于测试 radix prefix cache 和 ZipCache v2 compressed restore。"
        "这一段在多个请求中保持完全一致，目的是让第一次请求结束后形成可 demote 的 prefix，后续请求再次访问同一前缀。"
        "\n"
        + "\n".join(
            [
                f"固定段落 {i}: ZipCache v2 应该先把冷 prefix 的 KV cache 压缩保存在 GPU compressed pool，"
                f"后续相同 prefix 请求进入时通过 radix match 找到 compressed entry，再 restore 到 normal KV page。"
                for i in range(1, 16)
            ]
        )
    )
    rows = []
    for i, question in enumerate(RESTORE_QUESTIONS, start=1):
        rows.append(
            {
                "id": f"restore_pressure_{i:03d}",
                "prompt": (
                    f"{shared_prefix}\n\n"
                    f"本次请求问题：{question}\n"
                    "请按“结论、原因、验证方法、下一步”回答。"
                ),
                "max_tokens": 384,
            }
        )
    return rows


def main() -> None:
    write_jsonl(ROOT / "realistic_long_context.jsonl", make_long_context_rows())
    write_jsonl(ROOT / "zipcache_restore_pressure.jsonl", make_restore_rows())


if __name__ == "__main__":
    main()
