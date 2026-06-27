from __future__ import annotations

# 这个文件是后端调度器的核心。
#
# Scheduler 的主要职责：
# 1. 从 tokenizer 接收新请求 UserMsg / 取消请求 AbortBackendMsg；
# 2. 把新请求放入 prefill 队列；
# 3. 在 prefill 和 decode 请求中选择下一批 batch；
# 4. 准备模型 forward 需要的 input_ids、positions、KV cache 写入位置；
# 5. 调用 Engine.forward_batch() 执行模型；
# 6. 处理模型生成的 next token，更新请求状态；
# 7. 把 next token 发给 detokenizer，再返回给前端。

from typing import TYPE_CHECKING, List, NamedTuple, NoReturn, Set, Tuple, TypeAlias

import torch
from minisgl.core import Batch, Req
from minisgl.env import ENV
from minisgl.message import (
    AbortBackendMsg,
    BaseBackendMsg,
    BatchBackendMsg,
    DetokenizeMsg,
    ExitMsg,
    UserMsg,
)
from minisgl.utils import init_logger, load_tokenizer

from .cache import CacheManager
from .config import SchedulerConfig
from .decode import DecodeManager
from .io import SchedulerIOMixin
from .prefill import ChunkedReq, PrefillManager
from .table import TableManager

if TYPE_CHECKING:
    from minisgl.engine import BatchSamplingArgs, ForwardOutput


logger = init_logger(__name__)

Indice2D: TypeAlias = Tuple[torch.Tensor, torch.Tensor]


# For overlap scheduling, we also need to cache some other data to avoid IMA
class ForwardInput(NamedTuple):
    """一次 Engine forward 前需要准备好的所有输入。"""

    batch: Batch
    sample_args: BatchSamplingArgs

    # input_tuple 用来从 token_pool 取出本轮要送进模型的 token。
    input_tuple: Indice2D  # (token_mapping, positions)

    # write_tuple 用来把本轮生成的新 token 写回 token_pool。
    write_tuple: Indice2D  # (req_mapping, seq_lens or -1)


ForwardData: TypeAlias = "Tuple[ForwardInput, ForwardOutput]"


class Scheduler(SchedulerIOMixin):
    """miniSGLang 后端调度器。

    一个 Scheduler 对应一个 TP rank。rank 0 负责和 tokenizer/detokenizer 直接通信，
    其他 rank 通过 SchedulerIOMixin 接收 rank 0 广播的请求。
    """

    def __init__(self, config: SchedulerConfig):
        """初始化 Engine、cache/table/decode/prefill manager 和通信链路。"""

        from minisgl.engine import Engine

        # Engine 持有模型、KV cache pool、attention backend、CUDA graph、sampler。
        self.engine = Engine(config)

        # use another stream to overlap metadata processing with computation
        # Scheduler 使用自己的 CUDA stream 准备 metadata；Engine 使用 engine.stream 跑模型。
        # overlap scheduling 会让 CPU 调度/metadata 准备和 GPU forward 尽量重叠。
        self.device = self.engine.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        torch.cuda.set_stream(self.stream)

        # initialize other managers
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        self.cache_manager = CacheManager(
            self.engine.num_pages, config.page_size, self.engine.page_table, config.cache_type
        )
        self.decode_manager = DecodeManager(config.page_size)
        self.prefill_manager = PrefillManager(
            self.cache_manager, self.table_manager, self.decode_manager
        )

        # some alias for easy access
        self.finished_reqs: Set[Req] = set()
        self.tokenizer = load_tokenizer(config.model_path)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.token_pool = self.table_manager.token_pool
        self.prefill_budget = config.max_extend_tokens
        # self.config = config

        # Initialize the I/O mixin
        super().__init__(config, self.engine.tp_cpu_group)

    def run_when_idle(self) -> None:
        """Called when the scheduler is idle to perform background tasks."""
        logger.info_rank0("Scheduler is idle, waiting for new reqs...")
        # 空闲时做一次 cache 完整性检查，尽早发现 page 泄漏或统计错误。
        self.cache_manager.check_integrity()
        if self.engine.zipcache_manager is not None:
            self.engine.zipcache_manager.log_stats()

    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """
        overlap scheduling 主循环。

        它会先启动当前 batch 的 GPU forward，再处理上一轮 forward 的结果。
        这样 CPU 处理上一轮 next token / cache 释放时，GPU 可以同时跑当前 batch。
        """

        # 如果没有上一轮结果要处理、也没有可运行请求，就阻塞等待新消息。
        blocking = not (
            last_data is not None  # don't block if we have a batch to be processed
            or self.prefill_manager.runnable
            or self.decode_manager.runnable
        )

        # 接收并处理 tokenizer/rank0 广播来的消息。
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        # 选择下一轮要 forward 的 batch，并准备 metadata。
        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            with self.engine_stream_ctx:  # run the batch in the engine's stream
                # 确保 Engine stream 等待 Scheduler stream 上的 metadata 准备完成。
                self.engine.stream.wait_stream(self.stream)
                ongoing_data = (forward_input, self._forward(forward_input))

        # 当前 batch 已经提交到 GPU 后，处理上一轮数据。
        self._process_last_data(last_data)
        return ongoing_data

    def normal_loop(self) -> None:
        """非 overlap 模式主循环。

        按顺序执行：收消息 -> 调度 -> forward -> 处理结果。
        """

        blocking = not (self.prefill_manager.runnable or self.decode_manager.runnable)
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(ongoing_data)

    @torch.inference_mode()
    def run_forever(self) -> NoReturn:
        """Scheduler 主循环，正常情况下不会返回。"""

        if ENV.DISABLE_OVERLAP_SCHEDULING:
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                while True:
                    self.normal_loop()
        else:
            assert torch.cuda.current_stream() == self.stream
            data = None
            while True:
                data = self.overlap_loop(data)

    def shutdown(self) -> None:
        """关闭 Scheduler/Engine 前的同步和资源释放。"""

        torch.cuda.synchronize(self.device)
        self.sync_all_ranks()
        self.engine.shutdown()

    def _process_last_data(self, last_data: ForwardData | None) -> None:
        """处理上一轮 Engine forward 的输出。

        包括：
        - 等 next_tokens 从 GPU 拷到 CPU；
        - 把新 token 追加到 Req.input_ids；
        - 判断请求是否完成；
        - prefill 完成后把 prefix 放入 cache；
        - 完成请求释放 table slot/cache handle；
        - 向 detokenizer 发送 DetokenizeMsg。
        """

        if last_data is None:
            return

        batch, (_, next_tokens_cpu, copy_done) = last_data[0].batch, last_data[1]

        # 等待异步 GPU->CPU token 拷贝完成。
        copy_done.synchronize()
        reply: List[DetokenizeMsg] = []
        new_finished_reqs: Set[Req] = set()
        with self.cache_manager.lazy_free_region():
            for i, req in enumerate(batch.reqs):
                if isinstance(req, ChunkedReq):
                    # ChunkedReq 还没完成完整 prompt prefill，不产生输出 token。
                    continue
                next_token = next_tokens_cpu[i]
                req.append_host(next_token.unsqueeze(0))
                next_token = int(next_token.item())

                # 达到最大长度则完成；如果不 ignore_eos，生成 EOS 也完成。
                finished = not req.can_decode
                if not req.sampling_params.ignore_eos:
                    finished |= next_token == self.eos_token_id
                reply.append(DetokenizeMsg(uid=req.uid, next_token=next_token, finished=finished))

                # NOTE: overlap scheduling may make the request freed twice, skip second free
                if finished and req not in self.finished_reqs:
                    # 请求结束，移出 decode manager 并释放资源。
                    self.decode_manager.remove_req(req)
                    self._free_req_resources(req)
                    new_finished_reqs.add(req)
                elif batch.is_prefill:  # for prefill, non-chunk req, cache the prefix
                    # prefill 完成但请求还没结束，将已计算前缀插入 prefix cache。
                    self.cache_manager.cache_req(req, finished=False)

        self.finished_reqs = new_finished_reqs
        self.send_result(reply)

    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        """处理一条后端消息。"""

        if isinstance(msg, BatchBackendMsg):
            # 批量消息递归展开处理。
            for msg in msg.data:
                self._process_one_msg(msg)
        elif isinstance(msg, ExitMsg):
            # 用 KeyboardInterrupt 复用上层退出逻辑。
            raise KeyboardInterrupt
        elif isinstance(msg, UserMsg):
            logger.debug_rank0("Received user msg: %s", msg)
            input_len, max_seq_len = len(msg.input_ids), self.engine.max_seq_len
            max_output_len = max_seq_len - input_len
            if max_output_len <= 0:
                # prompt 已经超过模型最大上下文，直接丢弃。
                return logger.warning_rank0(
                    f"Input sequence length {input_len} exceeds {max_seq_len}, "
                    f"request {msg.uid} is dropped."
                )
            if msg.sampling_params.max_tokens > max_output_len:
                # 用户要求的生成长度超过剩余上下文，自动截断。
                msg.sampling_params.max_tokens = max_output_len
                logger.warning_rank0(
                    f"Adjust max_tokens to {max_output_len} for request {msg.uid}."
                )
            # 合法新请求进入 prefill 队列。
            self.prefill_manager.add_one_req(msg)
        elif isinstance(msg, AbortBackendMsg):
            logger.debug_rank0("Aborting request %d", msg.uid)
            # 取消请求可能处于 pending/prefill，也可能已经处于 decode。
            req_to_free = self.prefill_manager.abort_req(msg.uid)
            req_to_free = req_to_free or self.decode_manager.abort_req(msg.uid)
            if req_to_free is not None:
                self._free_req_resources(req_to_free)
        else:
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    def _free_req_resources(self, req: Req) -> None:
        """释放一个请求占用的 table slot，并把可缓存部分交给 cache manager。"""

        if self.engine.zipcache_manager is not None:
            self.engine.zipcache_manager.free_request(req.uid)
        self.table_manager.free(req.table_idx)
        self.cache_manager.cache_req(req, finished=True)

    def _prepare_batch(self, batch: Batch) -> ForwardInput:
        """为一个 batch 准备 Engine forward 所需的 tensor 和 metadata。"""

        # 如果能用 CUDA Graph，把 batch 补齐到已 capture 的大小。
        self.engine.graph_runner.pad_batch(batch)

        # 为本轮要计算的 token 分配 KV cache page。
        self.cache_manager.allocate_paged(batch.reqs)

        # positions 是每个 token 在原始序列里的位置。
        batch.positions = _make_positions(batch, self.device)

        # input_mapping 用来从 token_pool 中 gather 本轮 input_ids。
        input_mapping = _make_input_tuple(batch, self.device)

        # write_mapping 用来把 next token 写回 token_pool 对应位置。
        write_mapping = _make_write_tuple(batch, self.device)

        # out_loc 是本轮每个输入 token 对应的 KV cache 写入位置。
        batch.out_loc = self.engine.page_table[input_mapping]

        # attention backend 根据 batch/page table 准备自己的 metadata。
        self.engine.attn_backend.prepare_metadata(batch)
        return ForwardInput(
            batch=batch,
            sample_args=self.engine.sampler.prepare(batch),
            input_tuple=input_mapping,
            write_tuple=write_mapping,
        )

    def _schedule_next_batch(self) -> ForwardInput | None:
        """选择下一轮 batch。当前策略是 prefill 优先，其次 decode。"""

        # TODO: support other policies: e.g. DECODE first
        batch = (
            self.prefill_manager.schedule_next_batch(self.prefill_budget)
            or self.decode_manager.schedule_next_batch()
        )
        return self._prepare_batch(batch) if batch else None

    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        """调用 Engine 执行一次 batch forward，并更新 token_pool/decode manager。"""

        batch, sample_args, input_mapping, output_mapping = forward_input

        # 根据 input_mapping 从 token_pool 取出本轮输入 token。
        batch.input_ids = self.token_pool[input_mapping]
        forward_output = self.engine.forward_batch(batch, sample_args)

        # 把新生成 token 写回 token_pool，供后续 decode 使用。
        self.token_pool[output_mapping] = forward_output.next_tokens_gpu

        # 可 decode 的请求加入/保留在 decode manager。
        self.decode_manager.filter_reqs(forward_input.batch.reqs)
        return forward_output


def _make_positions(batch: Batch, device: torch.device) -> torch.Tensor:
    """生成 batch 中每个输入 token 的 position。"""

    needed_size = sum(r.extend_len for r in batch.padded_reqs)
    indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        torch.arange(
            req.cached_len,
            req.device_len,
            dtype=torch.int32,
            out=indices_host[offset : offset + length],
        )
        offset += length
    return indices_host.to(device, non_blocking=True)


def _make_input_tuple(batch: Batch, device: torch.device) -> Indice2D:
    """生成从 token_pool 读取输入 token 的二维索引。

    返回值是 (table_idx_tensor, position_tensor)，可用于：
        token_pool[table_idx_tensor, position_tensor]
    """

    mapping_host = torch.empty(len(batch.positions), dtype=torch.int64, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        mapping_host[offset : offset + length].fill_(req.table_idx)
        offset += length
    return mapping_host.to(device, non_blocking=True), batch.positions.to(torch.int64)


def _make_write_tuple(batch: Batch, device: torch.device) -> Indice2D:
    """生成写回 next token 的二维索引。

    对每个真实请求，写入位置是 req.device_len。
    如果 req 不能继续 decode，就写 -1；这些位置不会再被正常读取。
    """

    mapping_list = [req.table_idx for req in batch.reqs]
    mapping_host = torch.tensor(mapping_list, dtype=torch.int64, pin_memory=True)
    write_list = [(req.device_len if req.can_decode else -1) for req in batch.reqs]
    write_host = torch.tensor(write_list, dtype=torch.int64, pin_memory=True)
    return mapping_host.to(device, non_blocking=True), write_host.to(device, non_blocking=True)
