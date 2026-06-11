from __future__ import annotations

# 这个文件实现 Engine。
#
# Engine 是真正执行模型推理的对象。每个 TP rank 的 Scheduler 内部都会创建一个 Engine。
# 它负责：
# - 初始化 CUDA device 和 stream；
# - 初始化 torch.distributed / PyNCCL；
# - 创建模型并加载权重；
# - 分配 KV cache 和 page_table；
# - 创建 attention / MoE backend；
# - capture CUDA Graph；
# - 执行 batch forward 并采样 next token。

from datetime import timedelta
from typing import Any, Dict, NamedTuple, Tuple

import torch
from minisgl.attention import create_attention_backend
from minisgl.core import Batch, Context, Req, set_global_ctx
from minisgl.distributed import destroy_distributed, enable_pynccl_distributed, set_tp_info
from minisgl.kvcache import create_kvcache_pool
from minisgl.layers import set_rope_device
from minisgl.models import create_model, load_weight
from minisgl.moe import create_moe_backend
from minisgl.utils import div_even, init_logger, is_sm90_supported, is_sm100_supported, torch_dtype

from .config import EngineConfig
from .graph import GraphRunner, get_free_memory, mem_GB
from .sample import BatchSamplingArgs, Sampler

logger = init_logger(__name__)


class ForwardOutput(NamedTuple):
    """Engine.forward_batch() 的输出。"""

    # GPU 上的 next token，用于写回 token_pool。
    next_tokens_gpu: torch.Tensor

    # CPU 上的 next token，用于 scheduler 发给 detokenizer。
    next_tokens_cpu: torch.Tensor

    # GPU->CPU 异步拷贝完成事件。
    copy_done_event: torch.cuda.Event


class Engine:
    """单个 TP rank 上的模型执行引擎。"""

    def __init__(self, config: EngineConfig):
        """初始化 Engine 所需的全部运行时资源。"""

        # Engine 必须在 CUDA 初始化前设置 device/进程组，避免 CUDA context 在错误设备上创建。
        assert not torch.cuda.is_initialized()

        # 记录当前进程的 TP rank/size，供 layers/distributed 代码读取。
        set_tp_info(rank=config.tp_info.rank, size=config.tp_info.size)

        # 根据硬件和模型类型自动修正 attention/MoE/page_size 等配置。
        _adjust_config(config)

        # 每个 TP rank 使用一张 GPU，rank 号直接对应 cuda device id。
        self.device = torch.device(f"cuda:{config.tp_info.rank}")
        torch.cuda.set_device(self.device)
        torch.manual_seed(42)

        # Engine 使用自己的 CUDA stream 跑模型 forward。
        self.stream = torch.cuda.Stream()
        torch.cuda.set_stream(self.stream)
        self.dtype = config.dtype

        # 创建并注册全局 Context。后续模型 layer 会通过 get_global_ctx() 访问它。
        self.ctx = Context(config.page_size)
        set_global_ctx(self.ctx)

        # 初始化 TP 通信，并记录模型加载前的可用显存。
        self.tp_cpu_group = self._init_communication(config)
        init_free_memory = self._sync_get_memory()[1]
        logger.info_rank0(f"Free memory before loading model: {mem_GB(init_free_memory)}")

        # ======================= Model initialization ========================
        # RoPE cache 放在当前 device。
        set_rope_device(self.device)

        # torch.device("meta") 表示先创建“无真实内存”的模型结构，
        # 再通过 load_state_dict 加载实际权重，减少初始化峰值内存。
        with torch.device("meta"), torch_dtype(config.dtype):
            self.model = create_model(config.model_config)
        self.model.load_state_dict(self._load_weight_state_dict(config))

        # ======================= KV cache initialization ========================
        # 根据剩余显存估算可分配多少 KV cache page。
        self.num_pages = self._determine_num_pages(init_free_memory, config)
        num_tokens = self.num_pages * config.page_size
        self.ctx.kv_cache = self.kv_cache = create_kvcache_pool(
            model_config=config.model_config,
            num_pages=self.num_pages + 1,  # +1 for dummy page
            page_size=config.page_size,
            device=self.device,
            dtype=self.dtype,
        )

        # ======================= Page table initialization ========================
        # NOTE: 1. aligned to 128 bytes; 2. store raw locations instead of pages
        # max_seq_len 不能超过 KV cache 能容纳的 token 总数。
        self.max_seq_len = min(config.max_seq_len, num_tokens)

        # 对齐到 32，便于某些 CUDA kernel/graph buffer 使用规整长度。
        aligned_max_seq_len = _align_up_32(self.max_seq_len)

        # page_table[table_idx, token_pos] = KV cache 物理 token index。
        # +1 是给 dummy request 预留的槽位，CUDA graph padding 会用它。
        self.ctx.page_table = self.page_table = torch.zeros(  # + 1 for dummy request
            (config.max_running_req + 1, aligned_max_seq_len),
            dtype=torch.int32,
            device=self.device,
        )

        # ======================= Attention & MoE backend initialization ========================
        # 创建 attention backend，例如 FlashAttention、FlashInfer、TensorRT-LLM。
        self.ctx.attn_backend = self.attn_backend = create_attention_backend(
            config.attention_backend, config.model_config
        )
        if config.model_config.is_moe:
            # MoE 模型才需要 MoE backend。
            self.ctx.moe_backend = self.moe_backend = create_moe_backend(config.moe_backend)

        # ======================= Sampler initialization ========================
        # Sampler 把 logits 转成 next token。
        self.sampler = Sampler(self.device, config.model_config.vocab_size)

        post_free_memory = self._sync_get_memory()[0]
        logger.info_rank0(f"Free memory after initialization: {mem_GB(post_free_memory)}")

        # ======================= Graph capture initialization ========================
        # dummy_req 用于 CUDA graph padding。它不对应真实用户请求。
        self.dummy_req = Req(
            input_ids=torch.tensor([0], dtype=torch.int32, device="cpu"),
            table_idx=config.max_running_req,
            cached_len=0,
            output_len=1,
            uid=-1,
            sampling_params=None,  # type: ignore
            cache_handle=None,  # type: ignore
        )

        # dummy request 的 page table 指向最后额外分配的 dummy page。
        self.page_table[self.dummy_req.table_idx].fill_(num_tokens)  # point to dummy page

        # 初始化并 capture CUDA Graph。
        self.graph_runner = GraphRunner(
            stream=self.stream,
            device=self.device,
            model=self.model,
            attn_backend=self.attn_backend,
            cuda_graph_bs=config.cuda_graph_bs,
            cuda_graph_max_bs=config.cuda_graph_max_bs,
            free_memory=init_free_memory,
            max_seq_len=aligned_max_seq_len,
            vocab_size=config.model_config.vocab_size,
            dummy_req=self.dummy_req,
        )

    def _init_communication(self, config: EngineConfig) -> torch.distributed.ProcessGroup:
        """初始化 TP 通信。

        CPU 侧控制同步使用 gloo。GPU tensor 通信有两种路径：
        - use_pynccl=True：torch.distributed 建 gloo 组 + 自定义 PyNCCL 做 GPU 通信；
        - use_pynccl=False：torch.distributed nccl 组做 GPU 通信，再建 gloo CPU 组。
        """

        if config.tp_info.size == 1 or config.use_pynccl:
            torch.distributed.init_process_group(
                backend="gloo",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.group.WORLD
            assert tp_cpu_group is not None
            max_bytes = (
                config.max_forward_len * config.model_config.hidden_size * self.dtype.itemsize
            )
            # 初始化自定义 PyNCCL communicator。
            enable_pynccl_distributed(config.tp_info, tp_cpu_group, max_bytes)
        else:
            torch.distributed.init_process_group(
                backend="nccl",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.new_group(backend="gloo")
            assert tp_cpu_group is not None
        return tp_cpu_group

    def _load_weight_state_dict(self, config: EngineConfig) -> Dict[str, torch.Tensor]:
        """加载模型权重，返回可传给 model.load_state_dict 的字典。"""

        if config.use_dummy_weight:
            # 测试模式：生成形状匹配的随机权重。
            return {
                k: torch.randn_like(v, device=self.device)
                for k, v in self.model.state_dict().items()
            }
        else:
            # 正常模式：从模型目录读取权重，并转成目标 dtype。
            return {k: v.to(self.dtype) for k, v in load_weight(config.model_path, self.device)}

    def _determine_num_pages(self, old_free_memory: int, config: EngineConfig) -> int:
        """根据显存估算 KV cache page 数量。"""

        new_free_memory = self._sync_get_memory()[1]

        # 一个 page 的 KV cache 显存开销：
        # K/V 两份 * head_dim * 本 rank KV head 数 * page_size * dtype 字节数 * 层数。
        cache_per_page = (
            2  # key + value
            * config.model_config.head_dim
            * div_even(config.model_config.num_kv_heads, config.tp_info.size, allow_replicate=True)
            * config.page_size
            * self.dtype.itemsize
            * config.model_config.num_layers
        )
        num_pages = config.num_page_override
        if num_pages is None:
            # 模型加载消耗 = 加载前空闲 - 加载后空闲。
            model_memory = old_free_memory - new_free_memory

            # 可给 KV cache 使用的显存 = memory_ratio * 初始空闲 - 模型消耗。
            available_memory = int(config.memory_ratio * old_free_memory) - model_memory
            num_pages = available_memory // cache_per_page

        assert num_pages > 1, "Not enough memory for KV cache, try reducing --num-pages"
        num_tokens = num_pages * config.page_size
        real_kv_size = num_pages * cache_per_page
        logger.info(f"Allocating {num_tokens} tokens for KV cache, K + V = {mem_GB(real_kv_size)}")
        return num_pages

    def _sync_get_memory(self) -> Tuple[int, int]:
        """Get the min and max free memory across TP ranks."""

        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        free_memory = get_free_memory(self.device)

        # all_reduce 取所有 TP rank 的最小/最大空闲显存。
        # 第二个值用负数配合 MIN reduce，等价于求最大值。
        free_mem_tensor = torch.tensor([free_memory, -free_memory], device="cpu", dtype=torch.int64)
        torch.distributed.all_reduce(
            free_mem_tensor, op=torch.distributed.ReduceOp.MIN, group=self.tp_cpu_group
        )
        min_free_memory = int(free_mem_tensor[0].item())
        max_free_memory = -int(free_mem_tensor[1].item())
        if max_free_memory - min_free_memory > 2 * 1024 * 1024 * 1024:
            logger.error(
                f"Memory across TP ranks are imbalanced:"
                f" min {mem_GB(min_free_memory)}, max {mem_GB(max_free_memory)}"
            )
            raise RuntimeError("Memory across TP ranks are imbalanced")

        return min_free_memory, max_free_memory

    def forward_batch(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
        """执行一次 batch forward，并采样下一个 token。"""

        assert torch.cuda.current_stream() == self.stream

        # 设置当前 batch 到全局 Context，让模型内部 layer 可以读取 batch metadata。
        with self.ctx.forward_batch(batch):
            if self.graph_runner.can_use_cuda_graph(batch):
                # decode 小 batch 可使用 CUDA Graph replay。
                logits = self.graph_runner.replay(batch)
            else:
                # prefill 或不满足 graph 条件时直接普通 forward。
                logits = self.model.forward()

        # forward 完成后，每个请求的 device_len/cached_len 前进一个 token。
        for req in batch.reqs:
            req.complete_one()

        # 从 logits 采样 next token。
        next_tokens_gpu = self.sampler.sample(logits[: batch.size], args).to(torch.int32)

        # 异步拷贝到 CPU，Scheduler 后续会等待 copy_done_event。
        next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
        copy_done_event = torch.cuda.Event()
        copy_done_event.record(self.stream)
        return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)

    def shutdown(self) -> None:
        """释放 Engine 持有的通信和 graph 资源。"""

        self.graph_runner.destroy_cuda_graphs()
        torch.distributed.destroy_process_group()
        destroy_distributed()


def _align_up_32(num: int) -> int:
    """向上对齐到 32 的整数倍。"""

    return (num + 31) // 32 * 32


def _adjust_config(config: EngineConfig):
    """根据硬件能力和模型类型自动修正配置。

    注意：EngineConfig 是 frozen dataclass，正常不应修改。这里通过 object.__setattr__
    强行覆盖字段，所以只在初始化阶段谨慎使用。
    """

    def override(attr: str, value: Any):  # this is dangerous, use with caution
        object.__setattr__(config, attr, value)

    if config.attention_backend == "auto":
        # Blackwell/SM100 优先 TRTLLM；Hopper/SM90 默认 prefill 用 FA、decode 用 FI；
        # 其他架构默认 FlashInfer。
        backend = "trtllm" if is_sm100_supported() else ("fa,fi" if is_sm90_supported() else "fi")
        override("attention_backend", backend)
        logger.info_rank0(f"Auto-selected attention backend: {config.attention_backend}")

    if "trtllm" in config.attention_backend and config.page_size not in [16, 32, 64]:
        # TRTLLM backend 只支持特定 page size。
        override("page_size", 64)
        logger.warning_rank0("Page size is overridden to 64 for TRTLLM backend")

    if config.model_config.is_moe and config.moe_backend == "auto":
        # MoE 模型默认使用 fused MoE backend。
        override("moe_backend", "fused")
        logger.info_rank0(f"Auto-selected MoE backend: {config.moe_backend}")
