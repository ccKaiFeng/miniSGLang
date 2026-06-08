# miniSGLang 工程目录说明

本文档基于当前仓库源码、`README.md`、`docs/structures.md`、`docs/features.md` 和各 Python/CUDA/C++ 文件中的类、函数定义整理，用于快速理解 miniSGLang 的文件层次与模块职责。

## 1. 工程总体作用

miniSGLang 是一个轻量级 LLM 推理/Serving 框架，目标是用较小代码量展示现代大模型服务系统的核心机制。工程主体位于 `python/minisgl`，包含：

- OpenAI 兼容 API Server；
- tokenizer / detokenizer 工作进程；
- scheduler 调度进程；
- 单 GPU/TP rank 上的 engine；
- KV cache、radix prefix cache、attention backend、MoE backend；
- Llama、Qwen、Mistral 等模型实现；
- 自定义 CUDA/C++ kernel 及 Python binding；
- offline / online benchmark 和 pytest 测试。

核心请求链路为：

1. 用户请求进入 `server/api_server.py`；
2. API Server 通过 ZMQ 发送给 tokenizer；
3. tokenizer 将文本转成 token 后发送给 rank 0 scheduler；
4. scheduler 负责 prefill/decode 调度，并驱动本 rank 的 engine；
5. engine 调用模型、attention backend、KV cache、CUDA graph、采样逻辑完成推理；
6. rank 0 scheduler 将输出 token 发给 detokenizer；
7. detokenizer 还原文本并回传 API Server；
8. API Server 以流式或非流式方式返回用户。

多 GPU Tensor Parallel 场景下，每个 TP rank 有一个 scheduler/engine 进程，控制消息主要经 ZMQ，GPU 间张量通信使用 torch.distributed / PyNCCL。

## 2. 顶层文件与目录

```text
.
├── README.md                  # 项目介绍、安装、运行 server/shell、benchmark 示例
├── LICENSE                    # MIT License
├── Dockerfile                 # Linux/CUDA 容器构建与默认启动环境
├── pyproject.toml             # Python 包元数据、依赖、setuptools、black/ruff/pytest/mypy 配置
├── .gitignore                 # Git 忽略规则
├── .dockerignore -> .gitignore# Docker 忽略规则复用 .gitignore
├── .pre-commit-config.yaml    # pre-commit hook 配置，主要用于代码格式/静态检查
├── assets/                    # README/文档使用的静态资源
├── docs/                      # 架构和功能说明文档
├── benchmark/                 # 可独立运行的 offline/online benchmark 脚本
├── python/                    # Python package 源码根目录
└── tests/                     # pytest 测试用例
```

## 3. `docs/`

```text
docs/
├── features.md
└── structures.md
```

- `features.md`：说明 online serving、shell mode、TP、支持模型、chunked prefill、page size、attention backend、CUDA graph、radix cache、overlap scheduling 等功能。
- `structures.md`：说明 miniSGLang 的进程架构、请求生命周期、ZMQ/NCCL 通信关系，以及 `minisgl` 各源码子包职责。

## 4. `assets/`

```text
assets/
└── logo.png
```

- `logo.png`：项目 logo，当前主要被 `README.md` 引用。

## 5. `benchmark/`

```text
benchmark/
├── offline/
│   ├── bench.py
│   └── bench_wildchat.py
└── online/
    ├── bench_qwen.py
    └── bench_simple.py
```

- `benchmark/offline/bench.py`：离线推理 benchmark 入口，直接使用 `minisgl.llm.LLM` 批量生成，统计吞吐/延迟。
- `benchmark/offline/bench_wildchat.py`：基于 WildChat 数据集的离线 benchmark；包含数据下载、prompt token 长度过滤和长度统计逻辑。
- `benchmark/online/bench_simple.py`：简单 online benchmark 客户端，面向已启动的 OpenAI 兼容服务发送请求。
- `benchmark/online/bench_qwen.py`：面向 Qwen trace 的 online benchmark；包含 Qwen trace 下载和异步压测入口。

## 6. `python/minisgl/` 总览

```text
python/minisgl/
├── __main__.py        # python -m minisgl 入口
├── shell.py           # python -m minisgl.shell 入口，启动交互 shell
├── core.py            # 请求、batch、采样参数、全局 Context
├── env.py             # 环境变量封装
├── attention/         # attention backend 抽象和 FlashAttention/FlashInfer/TensorRT-LLM 实现
├── benchmark/         # benchmark 客户端与性能测量工具
├── distributed/       # TP 信息和 all-reduce/all-gather 通信封装
├── engine/            # 单 TP rank 推理引擎、CUDA graph、采样、配置
├── kernel/            # Python kernel wrapper、Triton kernel、C++/CUDA 源码
├── kvcache/           # KV cache pool 与 prefix cache 管理
├── layers/            # 模型基础算子层
├── llm/               # Python 直接调用接口
├── message/           # ZMQ 进程间消息类型和序列化
├── models/            # 模型结构、配置、权重加载
├── moe/               # MoE backend
├── scheduler/         # prefill/decode 调度、cache 分配、进程 IO
├── server/            # CLI 参数、server 启动、FastAPI API Server
├── tokenizer/         # tokenizer/detokenizer worker
└── utils/             # 日志、ZMQ 队列、HF 下载、数学工具、CUDA arch 工具
```

### 6.1 入口与核心数据结构

- `python/minisgl/__main__.py`：`python -m minisgl` 的入口，调用 `server.launch_server()` 启动服务。
- `python/minisgl/shell.py`：交互式 shell 入口，调用 `launch_server(run_shell=True)`。
- `python/minisgl/core.py`：定义核心 dataclass：
  - `SamplingParams`：用户采样参数，如 `max_tokens`、`temperature`、`top_p` 等；
  - `Req`：单个请求的运行状态，包括输入 token、输出 token、采样参数、KV cache handle 等；
  - `Batch`：一批请求的集合；
  - `Context`：engine 运行时全局上下文，保存 model config、KV cache pool、attention backend、MoE backend、CUDA graph 标记等；
  - `set_global_ctx()` / `get_global_ctx()`：设置和读取全局上下文。
- `python/minisgl/env.py`：封装环境变量读取，支持类型转换和默认值；用于控制 overlap scheduling、kernel 路径等运行时选项。

### 6.2 `server/`：服务启动与 HTTP 前端

```text
server/
├── __init__.py
├── api_server.py
├── args.py
└── launch.py
```

- `server/__init__.py`：导出 `launch_server`。
- `server/args.py`：定义 `ServerArgs`，继承 scheduler/engine 配置；`parse_args()` 解析 CLI 参数，例如 model、tp、host、port、cache、attn、shell 等。
- `server/launch.py`：启动 miniSGLang 后端子进程，包括 scheduler worker、tokenizer worker、detokenizer worker，并启动 API Server。
- `server/api_server.py`：FastAPI HTTP 前端：
  - 定义 `/generate`、`/v1/completions`、`/v1/chat/completions`、`/v1/models` 等接口；
  - `FrontendManager` 维护前端请求、ZMQ 队列和请求状态；
  - 支持 OpenAI 兼容流式输出；
  - 包含 shell completion 与交互式 shell 支持。

### 6.3 `tokenizer/`：文本与 token 转换进程

```text
tokenizer/
├── __init__.py
├── detokenize.py
├── server.py
└── tokenize.py
```

- `tokenizer/__init__.py`：导出 `tokenize_worker`。
- `tokenizer/server.py`：`tokenize_worker()` 进程入口；从 API Server/Scheduler 收发消息，分发 tokenize、detokenize、abort 等请求。
- `tokenizer/tokenize.py`：`TokenizeManager`，将用户 prompt/messages 编码为 token，并生成发往 scheduler 的请求消息。
- `tokenizer/detokenize.py`：`DetokenizeManager` 和 `DecodeStatus`，将输出 token 增量解码为可打印文本；`find_printable_text()` 处理不完整 unicode/空白片段。

### 6.4 `message/`：进程间消息与序列化

```text
message/
├── __init__.py
├── backend.py
├── frontend.py
├── tokenizer.py
└── utils.py
```

- `message/__init__.py`：集中导出 backend/frontend/tokenizer 消息类型。
- `message/frontend.py`：API Server 与 detokenizer/frontend 方向的消息，如 `UserReply`、`BatchFrontendMsg`。
- `message/tokenizer.py`：tokenizer 相关消息，如 `TokenizeMsg`、`DetokenizeMsg`、`AbortMsg`、`BatchTokenizerMsg`。
- `message/backend.py`：scheduler/backend 相关消息，如 `UserMsg`、`AbortBackendMsg`、`ExitMsg`、`BatchBackendMsg`。
- `message/utils.py`：通用序列化/反序列化工具，处理 dataclass、torch tensor、numpy 等对象。

### 6.5 `scheduler/`：请求调度与 KV cache 分配

```text
scheduler/
├── __init__.py
├── cache.py
├── config.py
├── decode.py
├── io.py
├── prefill.py
├── scheduler.py
├── table.py
└── utils.py
```

- `scheduler/__init__.py`：导出 `SchedulerConfig` 和 `Scheduler`。
- `scheduler/config.py`：`SchedulerConfig`，在 `EngineConfig` 基础上增加 ZMQ 地址、TP rank 信息、进程名后缀等调度侧配置。
- `scheduler/scheduler.py`：核心 `Scheduler`：
  - 接收 tokenizer 请求；
  - 维护 waiting/running/finished 请求；
  - 组织 prefill、decode；
  - 调用 engine forward；
  - rank 0 与 tokenizer/detokenizer 交互，多 rank 间广播请求；
  - `_make_positions()`、`_make_input_tuple()`、`_make_write_tuple()` 生成模型输入和 KV 写入索引。
- `scheduler/prefill.py`：`PrefillManager` 和 `PrefillAdder`，实现 chunked prefill 的请求选择、切分、token budget 管理；`ChunkedReq` 表示被切块的请求。
- `scheduler/decode.py`：`DecodeManager`，管理 decode 阶段 batch 选择、请求完成状态和持续解码集合。
- `scheduler/cache.py`：`CacheManager`，负责 prefix cache match/insert、KV page 分配、驱逐和 page table 写入；`_write_page_table()` 写入请求到物理 KV page 的映射。
- `scheduler/table.py`：`TableManager`，维护 token 到 KV cache page/index 的表。
- `scheduler/io.py`：`SchedulerIOMixin`，封装 scheduler 与 tokenizer、detokenizer、其他 TP scheduler 之间的 ZMQ push/pull/pub/sub 通信。
- `scheduler/utils.py`：调度辅助数据结构，如 `PendingReq`、`ScheduleResult`。

### 6.6 `engine/`：单 rank 推理执行

```text
engine/
├── __init__.py
├── config.py
├── engine.py
├── graph.py
└── sample.py
```

- `engine/__init__.py`：导出 `EngineConfig`、`Engine`、`ForwardOutput`、`BatchSamplingArgs`。
- `engine/config.py`：`EngineConfig`，包含 model path、dtype、device、TP 信息、attention/cache/backend 选择、CUDA graph 参数等。
- `engine/engine.py`：核心 `Engine`：
  - 初始化 TP、CUDA device、模型、权重、KV cache pool、attention backend、MoE backend；
  - 设置全局 `Context`；
  - 执行 prefill/decode forward；
  - 管理 CUDA graph capture/replay；
  - 调用 sampler 生成下一个 token；
  - `ForwardOutput` 保存 logits/sample 输出。
- `engine/graph.py`：CUDA graph 相关：
  - `GraphCaptureBuffer` 保存 capture 用 buffer；
  - `GraphRunner` 负责 capture 和 replay；
  - `get_free_memory()`、`mem_GB()`、`_determine_cuda_graph_bs()` 辅助 graph batch size 和显存判断。
- `engine/sample.py`：采样逻辑：
  - `BatchSamplingArgs` 保存 batch 级采样参数张量；
  - `Sampler` 根据 logits、temperature、top_p 等生成 token；
  - `sample_impl()` 是实际采样实现。

### 6.7 `models/`：模型结构与权重

```text
models/
├── __init__.py
├── base.py
├── config.py
├── llama.py
├── mistral.py
├── qwen2.py
├── qwen3.py
├── qwen3_moe.py
├── register.py
├── utils.py
└── weight.py
```

- `models/__init__.py`：导出 `BaseLLMModel`、`ModelConfig`、`RotaryConfig`、`load_weight()`，并提供 `create_model()`。
- `models/base.py`：`BaseLLMModel` 抽象基类，定义模型 forward、load weight 等接口。
- `models/config.py`：`ModelConfig` 和 `RotaryConfig`，从 HuggingFace config 中提取层数、hidden size、head 数、rope 参数、vocab size 等模型结构信息。
- `models/register.py`：根据 HuggingFace `architectures` 字段动态导入并选择模型类。
- `models/weight.py`：加载 safetensors 权重，并按 TP rank 做 tensor sharding、QKV/MLP merged weight 处理、MoE expert stack 处理。
- `models/utils.py`：模型通用组件：
  - `GatedMLP`：dense 模型 MLP；
  - `MoEMLP`：MoE MLP；
  - `RopeAttn`：带 RoPE 的 attention block 封装。
- `models/llama.py`：Llama decoder layer、model body 和 causal LM 实现。
- `models/mistral.py`：Mistral decoder layer、model body 和 causal LM 实现。
- `models/qwen2.py`：Qwen2 decoder layer、model body 和 causal LM 实现。
- `models/qwen3.py`：Qwen3 dense decoder layer、model body 和 causal LM 实现。
- `models/qwen3_moe.py`：Qwen3 MoE decoder layer、model body 和 causal LM 实现。

### 6.8 `layers/`：模型基础算子

```text
layers/
├── __init__.py
├── activation.py
├── attention.py
├── base.py
├── embedding.py
├── linear.py
├── moe.py
├── norm.py
└── rotary.py
```

- `layers/__init__.py`：集中导出 activation、attention、base、embedding、linear、MoE、norm、rotary 组件。
- `layers/base.py`：`BaseOP`、`StateLessOP`、`OPList`，定义参数加载、命名前缀、子模块列表等基础机制。
- `layers/activation.py`：`silu_and_mul()`、`gelu_and_mul()`，调用 `torch.ops.sgl_kernel` 的 fused activation。
- `layers/attention.py`：`AttentionLayer`，封装 Q/K/V reshape、RoPE、KV cache 写入、调用 attention backend。
- `layers/embedding.py`：`VocabParallelEmbedding` 和 `ParallelLMHead`，实现 vocab 维度 TP 切分、embedding lookup、LM head all-gather/all-reduce。
- `layers/linear.py`：TP 线性层实现，包括 replicated、column parallel merged、QKV merged、O projection、row parallel。
- `layers/moe.py`：`MoELayer`，实现 MoE gate、expert 权重组织、TP 通信和 MoE backend 调用。
- `layers/norm.py`：`RMSNorm` 和 `RMSNormFused`。
- `layers/rotary.py`：RoPE 缓存、Yarn/rope scaling 相关参数处理，提供 `get_rope()` 和 `set_rope_device()`。

### 6.9 `attention/`：Attention backend

```text
attention/
├── __init__.py
├── base.py
├── fa.py
├── fi.py
├── trtllm.py
└── utils.py
```

- `attention/__init__.py`：backend registry，提供 `validate_attn_backend()`、`create_attention_backend()`，支持 auto/fa/fi/trtllm 选择。
- `attention/base.py`：`BaseAttnMetadata`、`BaseAttnBackend` 抽象类，以及 prefill/decode 可组合的 `HybridBackend`。
- `attention/utils.py`：`BaseCaptureData`，保存 CUDA graph capture 过程中复用的 attention metadata 张量。
- `attention/fa.py`：FlashAttention backend，定义 `FAMetadata`、`FACaptureData`、`FlashAttentionBackend`，处理 prefill/decode 所需 metadata 并调用 FA kernel。
- `attention/fi.py`：FlashInfer backend，定义 `FIMetadata`、`FICaptureData`、`FlashInferBackend`，处理 paged KV cache、decode wrapper、CUDA graph capture。
- `attention/trtllm.py`：TensorRT-LLM FMHA backend，定义 `TRTLLMMetadata`、`TRTLLMCaptureData`、`TensorRTLLMBackend`。

### 6.10 `kvcache/`：KV cache 与 prefix cache

```text
kvcache/
├── __init__.py
├── base.py
├── mha_pool.py
├── naive_cache.py
└── radix_cache.py
```

- `kvcache/__init__.py`：KV cache/prefix cache factory，提供 `create_kvcache_pool()`、`create_prefix_cache()`、`create_naive_cache()`、`create_radix_cache()`。
- `kvcache/base.py`：抽象接口和结果类型：
  - `BaseKVCachePool`；
  - `BaseCacheHandle`；
  - `BasePrefixCache`；
  - `SizeInfo`、`InsertResult`、`MatchResult`。
- `kvcache/mha_pool.py`：`MHAKVCache`，为 multi-head attention 管理 K/V cache tensor，按 TP rank 切分 head。
- `kvcache/naive_cache.py`：`NaivePrefixCache` 和 `NaiveCacheHandle`，不做复杂 prefix 复用的 cache 管理实现。
- `kvcache/radix_cache.py`：`RadixPrefixCache`、`RadixTreeNode`、`RadixCacheHandle`，用 radix tree 进行共享前缀匹配、插入、驱逐和引用计数管理。

### 6.11 `moe/`：MoE backend

```text
moe/
├── __init__.py
├── base.py
└── fused.py
```

- `moe/__init__.py`：MoE backend registry，提供 `create_moe_backend()` 和默认 fused backend。
- `moe/base.py`：`BaseMoeBackend` 抽象接口。
- `moe/fused.py`：fused MoE 实现：
  - `fused_topk()` 选择 top-k expert；
  - `moe_align_block_size()` 对 token/expert block 做对齐；
  - `fused_experts_impl()` 调用 fused expert kernel；
  - `FusedMoe` 封装完整 MoE backend。

### 6.12 `distributed/`：Tensor Parallel 通信

```text
distributed/
├── __init__.py
├── impl.py
└── info.py
```

- `distributed/__init__.py`：导出 TP info 和通信接口。
- `distributed/info.py`：`DistributedInfo`，保存 `rank`、`size`、local rank 等 TP 信息；提供 `set_tp_info()`、`get_tp_info()`、`try_get_tp_info()`。
- `distributed/impl.py`：
  - `DistributedImpl` 抽象通信接口；
  - `TorchDistributedImpl` 使用 torch.distributed；
  - `PyNCCLDistributedImpl` 使用自定义 PyNCCL wrapper；
  - `DistributedCommunicator` 统一暴露 all-reduce/all-gather；
  - `enable_pynccl_distributed()`、`destroy_distributed()` 管理通信后端生命周期。

### 6.13 `kernel/`：Python/CUDA/C++ kernel

```text
kernel/
├── __init__.py
├── __main__.py
├── index.py
├── moe_impl.py
├── pynccl.py
├── radix.py
├── store.py
├── tensor.py
├── triton/
│   └── fused_moe.py
└── csrc/
    ├── include/minisgl/
    │   ├── nccl227.h
    │   ├── tensor.h
    │   ├── utils.cuh
    │   ├── utils.h
    │   └── warp.cuh
    ├── jit/
    │   ├── index.cu
    │   └── store.cu
    └── src/
        ├── pynccl.cu
        ├── radix.cpp
        └── tensor.cpp
```

- `kernel/__init__.py`：导出 `indexing`、`store_cache`、`fast_compare_key`、`test_tensor`、PyNCCL 和 MoE Triton wrapper。
- `kernel/__main__.py`：生成 `.clangd` 的辅助入口，会读取 GPU compute capability 并写入 CUDA/C++ include flags。
- `kernel/utils.py`：TVM FFI AOT/JIT 编译加载工具：
  - `KernelConfig` 描述 kernel 名称、源码、编译参数；
  - `load_aot()` 加载 AOT module；
  - `load_jit()` JIT 编译并缓存 CUDA/C++ kernel；
  - `make_cpp_args()` 生成 C++ 模板参数。
- `kernel/index.py`：Python wrapper，JIT 编译 `csrc/jit/index.cu`，提供 `indexing()`，用于按 index 从 weight/cache 中 gather 数据。
- `kernel/store.py`：Python wrapper，JIT 编译 `csrc/jit/store.cu`，提供 `store_cache()`，用于将 K/V 写入 KV cache。
- `kernel/radix.py`：加载 AOT `radix` module，提供 `fast_compare_key()`，用于 radix cache 的 CPU key 快速比较。
- `kernel/tensor.py`：加载 AOT test tensor module，提供 `test_tensor()` 测试 C++/CUDA tensor binding。
- `kernel/pynccl.py`：加载 AOT PyNCCL module，提供 `init_pynccl()` 和 `PyNCCLCommunicator` wrapper。
- `kernel/moe_impl.py`：Python 侧 Triton MoE wrapper，提供 fused MoE kernel 和 sum reduce kernel 调用。
- `kernel/triton/fused_moe.py`：Triton kernel 定义，包括 `fused_moe_kernel()` 和 `moe_sum_reduce_kernel()`。
- `kernel/csrc/jit/index.cu`：JIT CUDA index kernel，每个 warp 复制一个或一段 element，支持 mask 范围外置零。
- `kernel/csrc/jit/store.cu`：JIT CUDA KV cache store kernel，每个 warp 将一条 K/V 写入指定 cache index。
- `kernel/csrc/src/radix.cpp`：TVM FFI 导出的 `fast_compare_key()`，比较两个 1D CPU int tensor 的共同前缀长度。
- `kernel/csrc/src/tensor.cpp`：TVM FFI tensor 示例/测试 module。
- `kernel/csrc/src/pynccl.cu`：PyNCCL C++/CUDA binding，实现 NCCL communicator 相关封装。
- `kernel/csrc/include/minisgl/tensor.h`：TVM/DLPack tensor 相关 C++ helper。
- `kernel/csrc/include/minisgl/utils.h`：host 侧 C++ 工具，如 runtime check、shape/dtype/device matcher、通用数学函数。
- `kernel/csrc/include/minisgl/utils.cuh`：device/host CUDA 工具，如 kernel launch、指针 offset、PDL 支持等。
- `kernel/csrc/include/minisgl/warp.cuh`：warp 级 copy/reset 等工具函数。
- `kernel/csrc/include/minisgl/nccl227.h`：NCCL 2.27 相关头文件兼容/声明。

### 6.14 `llm/`：Python 直接调用接口

```text
llm/
├── __init__.py
└── llm.py
```

- `llm/__init__.py`：导出 `LLM`。
- `llm/llm.py`：`LLM` 继承 `Scheduler`，提供 Python 侧直接调用接口；`RequestStatus` 跟踪请求状态，`RequestAllFinished` 表示批量请求完成。

### 6.15 `benchmark/` 包：benchmark 公共工具

```text
python/minisgl/benchmark/
├── client.py
└── perf.py
```

- `benchmark/client.py`：
  - 定义 `BenchmarkTrace`、`BenchOneResult`、`RawResult`、`BenchmarkResult`；
  - 提供异步 OpenAI 客户端压测函数 `benchmark_one()`、`benchmark_one_batch()`、`benchmark_trace()`；
  - 提供 Qwen/Mooncake trace 读取、trace 缩放、结果统计函数。
- `benchmark/perf.py`：CUDA 性能测量工具，包含 `perf_cuda()` 和 `compare_memory_kernel_perf()`。

### 6.16 `utils/`：通用工具

```text
utils/
├── __init__.py
├── arch.py
├── hf.py
├── logger.py
├── misc.py
├── mp.py
├── registry.py
└── torch_utils.py
```

- `utils/__init__.py`：集中导出常用工具。
- `utils/arch.py`：读取 torch CUDA 版本并判断 SM90/SM100 等架构支持。
- `utils/hf.py`：HuggingFace / ModelScope 相关工具，包含 tokenizer/config 加载、权重下载、禁用 tqdm helper。
- `utils/logger.py`：`init_logger()`，统一日志格式和等级。
- `utils/misc.py`：数学/小工具，如 `div_even()`、`div_ceil()`、`align_ceil()`、`align_down()`、`call_if_main()`、`Unset/UNSET`。
- `utils/mp.py`：基于 ZMQ/msgpack 的进程间队列封装，包括同步/异步 push/pull、pub/sub。
- `utils/registry.py`：通用 `Registry`，用于 attention/MoE/cache backend 注册和创建。
- `utils/torch_utils.py`：torch dtype context、NVTX 标注等工具。

## 7. `tests/`

```text
tests/
├── core/
│   ├── test_cache_allocate.py
│   └── test_scheduler.py
├── kernel/
│   ├── test_comm.py
│   ├── test_index.py
│   ├── test_store.py
│   └── test_tensor.py
└── misc/
    └── test_serialize.py
```

- `tests/core/test_cache_allocate.py`：测试 `CacheManager` 的 page 分配、驱逐、page 对齐、不重叠等行为。
- `tests/core/test_scheduler.py`：启动 scheduler 相关最小流程，验证 scheduler 配置和进程运行路径。
- `tests/kernel/test_index.py`：测试 `kernel.indexing()`，包含普通 indexing 和 mask indexing，与 PyTorch reference 对比。
- `tests/kernel/test_store.py`：测试 `kernel.store_cache()` 写入 KV cache 的正确性。
- `tests/kernel/test_tensor.py`：测试自定义 tensor binding / `test_tensor()`。
- `tests/kernel/test_comm.py`：测试 TP 通信/PyNCCL 或 distributed 相关路径。
- `tests/misc/test_serialize.py`：测试 message/dataclass 序列化与反序列化。

## 8. 阅读和修改建议

- 想理解服务启动：从 `python/minisgl/__main__.py` → `server/launch.py` → `server/api_server.py` 看起。
- 想理解一次请求的调度：从 `message/*.py` → `tokenizer/server.py` → `scheduler/scheduler.py` → `engine/engine.py` 看起。
- 想理解 prefill/decode：重点看 `scheduler/prefill.py`、`scheduler/decode.py`、`scheduler/cache.py` 和 `engine/engine.py`。
- 想理解 KV cache/radix cache：重点看 `kvcache/base.py`、`kvcache/radix_cache.py`、`scheduler/cache.py`、`kernel/radix.py`。
- 想理解 attention backend：重点看 `layers/attention.py`、`attention/base.py`、`attention/fa.py`、`attention/fi.py`、`attention/trtllm.py`。
- 想理解模型权重加载和 TP 切分：重点看 `models/config.py`、`models/weight.py`、`layers/linear.py`、`layers/embedding.py`。
- 想调 kernel：重点看 `kernel/utils.py`、`kernel/index.py`、`kernel/store.py`、`kernel/csrc/jit/*.cu` 和 `tests/kernel/`。

