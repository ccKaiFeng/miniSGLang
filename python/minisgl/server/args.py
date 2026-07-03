from __future__ import annotations

# 这个文件负责解析命令行参数，并把它们整理成 ServerArgs 配置对象。
#
# 用户启动服务时会执行类似：
#   python -m minisgl --model Qwen/Qwen3-0.6B --tp-size 2 --port 1919
#
# parse_args() 会把这些字符串参数转换成后端真正需要的 Python 配置：
# - 模型路径；
# - dtype；
# - TP 并行大小；
# - KV cache 参数；
# - attention backend；
# - ZMQ 通信地址；
# - 是否进入 shell 模式等。

import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple

import torch
from minisgl.distributed import DistributedInfo
from minisgl.scheduler import SchedulerConfig
from minisgl.utils import init_logger


@dataclass(frozen=True)
class ServerArgs(SchedulerConfig):
    """服务启动配置。

    ServerArgs 继承 SchedulerConfig，而 SchedulerConfig 又继承 EngineConfig。
    所以这个类里既有 HTTP Server 参数，也有 Scheduler/Engine 参数。

    frozen=True 表示创建后不允许直接修改字段。需要改字段时使用
    dataclasses.replace() 复制一份新配置。
    """

    server_host: str = "127.0.0.1"
    server_port: int = 1919
    num_tokenizer: int = 0
    silent_output: bool = False

    @property
    def share_tokenizer(self) -> bool:
        """是否让 tokenize 和 detokenize 共用同一个 tokenizer worker。

        num_tokenizer == 0 时，不额外启动 tokenizer 进程，只启动一个 detokenizer
        worker，并让前端 tokenize 请求也发给它处理。
        """

        return self.num_tokenizer == 0

    @property
    def zmq_frontend_addr(self) -> str:
        """detokenizer 发回 API Server 的 ZMQ 地址。"""

        return "ipc:///tmp/minisgl_3" + self._unique_suffix

    @property
    def zmq_tokenizer_addr(self) -> str:
        """API Server 发 tokenize 请求的 ZMQ 地址。"""

        if self.share_tokenizer:
            # 共用 worker 时，tokenize 和 detokenize 都走同一个地址。
            return self.zmq_detokenizer_addr
        result = "ipc:///tmp/minisgl_4" + self._unique_suffix
        assert result != self.zmq_detokenizer_addr
        return result

    @property
    def tokenizer_create_addr(self) -> bool:
        """tokenizer worker 是否负责创建 ZMQ endpoint。"""

        return self.share_tokenizer

    @property
    def backend_create_detokenizer_link(self) -> bool:
        """后端是否负责创建连接 detokenizer 的 ZMQ endpoint。"""

        return not self.share_tokenizer

    @property
    def frontend_create_tokenizer_link(self) -> bool:
        """前端是否负责创建连接 tokenizer 的 ZMQ endpoint。"""

        return not self.share_tokenizer

    @property
    def distributed_addr(self) -> str:
        """torch.distributed / TP 通信用的 TCP 地址。"""

        return f"tcp://127.0.0.1:{self.server_port + 1}"


def parse_args(args: List[str], run_shell: bool = False) -> Tuple[ServerArgs, bool]:
    """
    解析命令行参数并返回 ServerArgs。

    Args:
        args: 命令行参数，例如 sys.argv[1:]。
        run_shell: 外部是否已经要求 shell 模式。

    Returns:
        (ServerArgs, run_shell)：
        - ServerArgs 是完整服务配置；
        - run_shell 表示最终是否启动交互式 shell，而不是 HTTP server。
    """

    from minisgl.attention import validate_attn_backend
    from minisgl.kvcache import SUPPORTED_CACHE_MANAGER
    from minisgl.moe import SUPPORTED_MOE_BACKENDS

    parser = argparse.ArgumentParser(description="MiniSGL Server Arguments")

    # 模型路径是必需参数，可以是本地目录，也可以是 HuggingFace repo id。
    parser.add_argument(
        "--model-path",
        "--model",
        type=str,
        required=True,
        help="The path of the model weights. This can be a local folder or a Hugging Face repo ID.",
    )

    # dtype 控制权重和激活的数据类型。auto 会根据模型 config 自动推断。
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Data type for model weights and activations. 'auto' will use FP16 for FP32/FP16 models and BF16 for BF16 models.",
    )

    # Tensor Parallel 大小，也就是把模型切到多少张 GPU 上。
    parser.add_argument(
        "--tensor-parallel-size",
        "--tp-size",
        type=int,
        default=1,
        help="The tensor parallelism size.",
    )

    # Scheduler 同时运行的最大请求数。
    parser.add_argument(
        "--max-running-requests",
        type=int,
        dest="max_running_req",
        default=ServerArgs.max_running_req,
        help="The maximum number of running requests.",
    )

    # 手动覆盖模型最大序列长度，常用于测试或限制显存。
    parser.add_argument(
        "--max-seq-len-override",
        type=int,
        default=ServerArgs.max_seq_len_override,
        help="The maximum sequence length override.",
    )

    # 用多少比例的 GPU 显存作为 KV cache。
    parser.add_argument(
        "--memory-ratio",
        type=float,
        default=ServerArgs.memory_ratio,
        help="The fraction of GPU memory to use for KV cache.",
    )

    assert ServerArgs.use_dummy_weight == False
    # dummy weight 用于测试链路，不加载真实模型权重。
    parser.add_argument(
        "--dummy-weight",
        action="store_true",
        dest="use_dummy_weight",
        help="Use dummy weights for testing.",
    )

    assert ServerArgs.use_pynccl == True
    # PyNCCL 是本工程自定义 NCCL binding；可通过参数关闭。
    parser.add_argument(
        "--disable-pynccl",
        action="store_false",
        dest="use_pynccl",
        help="Disable PyNCCL for tensor parallelism.",
    )

    # HTTP Server 监听地址。
    parser.add_argument(
        "--host",
        type=str,
        dest="server_host",
        default=ServerArgs.server_host,
        help="The host address for the server.",
    )

    # HTTP Server 监听端口。distributed_addr 默认会使用 port + 1。
    parser.add_argument(
        "--port",
        type=int,
        dest="server_port",
        default=ServerArgs.server_port,
        help="The port number for the server to listen on.",
    )

    # CUDA Graph capture 的最大 batch size。用于降低 decode 阶段 CPU launch 开销。
    parser.add_argument(
        "--cuda-graph-max-bs",
        "--graph",
        type=int,
        default=ServerArgs.cuda_graph_max_bs,
        help="The maximum batch size for CUDA graph capture. None means auto-tuning based on the GPU memory.",
    )

    # tokenizer 进程数量。0 表示和 detokenizer 共用一个 worker。
    parser.add_argument(
        "--num-tokenizer",
        "--tokenizer-count",
        type=int,
        default=ServerArgs.num_tokenizer,
        help="The number of tokenizer processes to launch. 0 means the tokenizer is shared with the detokenizer.",
    )

    # chunked prefill 每次最多处理多少 token。
    parser.add_argument(
        "--max-prefill-length",
        "--max-extend-length",
        type=int,
        dest="max_extend_tokens",
        default=ServerArgs.max_extend_tokens,
        help="Chunk Prefill maximum chunk size in tokens.",
    )

    # 手动设置 KV cache page 数量。
    parser.add_argument(
        "--num-pages",
        dest="num_page_override",
        type=int,
        default=ServerArgs.num_page_override,
        help="Set the maximum number of pages for KVCache.",
    )

    # KV cache 页大小。
    parser.add_argument(
        "--page-size",
        type=int,
        default=ServerArgs.page_size,
        help="Set the page size for system management.",
    )

    # attention backend 可以是 fa/fi/trtllm/auto，支持 prefill 和 decode 使用不同 backend。
    parser.add_argument(
        "--attention-backend",
        "--attn",
        type=validate_attn_backend,
        default=ServerArgs.attention_backend,
        help="The attention backend to use. If two backends are specified,"
        " the first one is used for prefill and the second one for decode.",
    )

    # 模型下载来源。modelscope 分支会先把模型下载到本地。
    parser.add_argument(
        "--model-source",
        type=str,
        default="huggingface",
        choices=["huggingface", "modelscope"],
        help="The source to download model from. Either 'huggingface' or 'modelscope'.",
    )

    # KV cache 管理策略，例如 naive 或 radix。
    parser.add_argument(
        "--cache-type",
        type=str,
        default=ServerArgs.cache_type,
        choices=SUPPORTED_CACHE_MANAGER.supported_names(),
        help="The KV cache management strategy.",
    )

    # MoE backend 选择。
    parser.add_argument(
        "--moe-backend",
        default=ServerArgs.moe_backend,
        choices=["auto"] + SUPPORTED_MOE_BACKENDS.supported_names(),
        help="The MoE backend to use.",
    )

    parser.add_argument(
        "--enable-zipcache-v3",
        action="store_true",
        default=ServerArgs.enable_zipcache_v3,
        help="Enable experimental ZipCache v3 GPU compressed prefix archive.",
    )
    parser.add_argument(
        "--enable-zipcache-cuda-graph",
        action="store_true",
        default=ServerArgs.enable_zipcache_cuda_graph,
        help=(
            "Allow ZipCache v3 to use CUDA Graph for decode forward. "
            "Restore/demote still run outside CUDA Graph. Experimental."
        ),
    )
    parser.add_argument(
        "--zipcache-unimportant-ratio",
        type=float,
        default=ServerArgs.zipcache_unimportant_ratio,
        help="The fraction of KV tokens quantized with the lower ZipCache bit width.",
    )
    parser.add_argument(
        "--zipcache-k-important-bit",
        type=int,
        default=ServerArgs.zipcache_k_important_bit,
        help="Quantization bit width for salient Key cache tokens.",
    )
    parser.add_argument(
        "--zipcache-k-unimportant-bit",
        type=int,
        default=ServerArgs.zipcache_k_unimportant_bit,
        help="Quantization bit width for non-salient Key cache tokens.",
    )
    parser.add_argument(
        "--zipcache-v-important-bit",
        type=int,
        default=ServerArgs.zipcache_v_important_bit,
        help="Quantization bit width for salient Value cache tokens.",
    )
    parser.add_argument(
        "--zipcache-v-unimportant-bit",
        type=int,
        default=ServerArgs.zipcache_v_unimportant_bit,
        help="Quantization bit width for non-salient Value cache tokens.",
    )
    parser.add_argument(
        "--zipcache-protect-recent-tokens",
        type=int,
        default=ServerArgs.zipcache_protect_recent_tokens,
        help="Always keep the most recent N tokens out of the low-bit group.",
    )
    parser.add_argument(
        "--zipcache-stats-interval",
        type=float,
        default=ServerArgs.zipcache_stats_interval,
        help="Seconds between ZipCache stats logs. Set <= 0 to disable periodic logs.",
    )
    parser.add_argument(
        "--zipcache-v3-normal-pool-pages",
        type=int,
        default=ServerArgs.zipcache_v3_normal_pool_pages,
        help="Normal fp16/bf16 KV pool pages for ZipCache v3. 0 keeps the normal policy.",
    )
    parser.add_argument(
        "--zipcache-v3-demote-on-finish",
        action=argparse.BooleanOptionalAction,
        default=ServerArgs.zipcache_v3_demote_on_finish,
        help="Demote finished radix cache nodes into the ZipCache v3 compressed pool.",
    )
    parser.add_argument(
        "--zipcache-v3-compressed-pool-mb",
        type=int,
        default=ServerArgs.zipcache_v3_compressed_pool_mb,
        help="Fixed GPU compressed pool size for ZipCache v3. 0 derives from ratio.",
    )
    parser.add_argument(
        "--zipcache-v3-compressed-pool-ratio",
        type=float,
        default=ServerArgs.zipcache_v3_compressed_pool_ratio,
        help="ZipCache v3 compressed pool bytes as a ratio of the normal KV pool when MB is 0.",
    )
    parser.add_argument(
        "--zipcache-v3-q4-pool-ratio",
        type=float,
        default=ServerArgs.zipcache_v3_q4_pool_ratio,
        help="ZipCache v3 q4 buffer ratio inside compressed pool.",
    )
    parser.add_argument(
        "--zipcache-v3-q2-pool-ratio",
        type=float,
        default=ServerArgs.zipcache_v3_q2_pool_ratio,
        help="ZipCache v3 q2 buffer ratio inside compressed pool.",
    )
    parser.add_argument(
        "--zipcache-v3-scale-pool-ratio",
        type=float,
        default=ServerArgs.zipcache_v3_scale_pool_ratio,
        help="ZipCache v3 scale/min buffer ratio inside compressed pool.",
    )
    parser.add_argument(
        "--zipcache-v3-ids-pool-ratio",
        type=float,
        default=ServerArgs.zipcache_v3_ids_pool_ratio,
        help="ZipCache v3 token-id buffer ratio inside compressed pool.",
    )
    parser.add_argument(
        "--zipcache-v3-keep-compressed-after-restore",
        action=argparse.BooleanOptionalAction,
        default=ServerArgs.zipcache_v3_keep_compressed_after_restore,
        help="Keep compressed entries after temporary restore in ZipCache v3.",
    )
    parser.add_argument(
        "--zipcache-v3-min-restore-tokens",
        type=int,
        default=ServerArgs.zipcache_v3_min_restore_tokens,
        help="Only restore compressed radix nodes with at least this many tokens.",
    )

    # shell 模式：不启动 HTTP 服务，而是在当前终端里交互聊天。
    parser.add_argument(
        "--shell-mode",
        action="store_true",
        help="Run the server in shell mode.",
    )

    # Parse arguments
    # argparse 输出 Namespace，这里转成 dict，方便传给 ServerArgs(**kwargs)。
    kwargs = parser.parse_args(args).__dict__.copy()

    # resolve some arguments
    # run_shell 可能来自命令行，也可能由外部入口 python -m minisgl.shell 指定。
    run_shell |= kwargs.pop("shell_mode")
    if run_shell:
        # shell 模式一次只服务一个终端请求，所以把图和运行请求数限制为 1。
        kwargs["cuda_graph_max_bs"] = 1
        kwargs["max_running_req"] = 1
        kwargs["silent_output"] = True

    zipcache_enabled = kwargs["enable_zipcache_v3"]
    zipcache_cuda_graph_allowed = (
        kwargs["enable_zipcache_cuda_graph"] and kwargs["enable_zipcache_v3"]
    )
    if kwargs["enable_zipcache_cuda_graph"] and not zipcache_cuda_graph_allowed:
        parser.error("--enable-zipcache-cuda-graph requires --enable-zipcache-v3.")
    if zipcache_enabled and not zipcache_cuda_graph_allowed:
        # v3 的 restore/demote 在 graph 外执行。默认关闭 graph，方便先看清
        # 压缩路径本身的开销；需要公平对比时可显式开启。
        kwargs["cuda_graph_max_bs"] = 0
    elif zipcache_cuda_graph_allowed and kwargs["cuda_graph_max_bs"] is None:
        # main 会根据显存自动 capture 到 160/256。ZipCache v3 还要给
        # compressed pool 和临时 restore pages 留空间，因此默认采用保守上限。
        # 需要更大 decode graph 时可显式传 --cuda-graph-max-bs。
        kwargs["cuda_graph_max_bs"] = 16

    # 展开用户目录路径，例如 ~/models/qwen。
    if kwargs["model_path"].startswith("~"):
        kwargs["model_path"] = os.path.expanduser(kwargs["model_path"])

    # 如果模型来源是 ModelScope，且 model_path 不是本地目录，则先下载模型。
    if kwargs["model_source"] == "modelscope":
        model_path = kwargs["model_path"]
        if not os.path.isdir(model_path):
            from modelscope import snapshot_download

            ignore_patterns = []
            if kwargs["use_dummy_weight"]:
                ignore_patterns = ["*.bin", "*.safetensors", "*.pt", "*.ckpt"]
            model_path = snapshot_download(model_path, ignore_patterns=ignore_patterns)
            kwargs["model_path"] = model_path
    del kwargs["model_source"]

    # dtype=auto 时，从模型 config 里读取实际 dtype。
    if (dtype_str := kwargs["dtype"]) == "auto":
        from minisgl.utils import cached_load_hf_config

        dtype_str = cached_load_hf_config(kwargs["model_path"]).dtype

    # 把字符串 dtype 转成 torch.dtype 对象，后续 Engine 初始化直接使用。
    DTYPE_MAP = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    kwargs["dtype"] = DTYPE_MAP[dtype_str] if isinstance(dtype_str, str) else dtype_str

    # rank 0 的初始 TP 信息。真正启动多进程时，launch.py 会为每个 rank 替换 tp_info。
    kwargs["tp_info"] = DistributedInfo(0, kwargs["tensor_parallel_size"])
    del kwargs["tensor_parallel_size"]

    # 用解析后的参数构造不可变 ServerArgs。
    result = ServerArgs(**kwargs)
    logger = init_logger(__name__)
    logger.info(f"Parsed arguments:\n{result}")
    return result, run_shell
