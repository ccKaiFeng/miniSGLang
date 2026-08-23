# Mini-SGLang 的结构

## 系统架构

Mini-SGLang 被设计为一个分布式系统，用于高效执行大语言模型（Large Language Model，LLM）推理。系统由多个相互独立的进程协作完成一次请求。

### 关键组件

- **API Server（API 服务）**：用户请求的入口。它提供兼容 OpenAI 的 API，例如 `/v1/chat/completions`，负责接收 prompt 并返回生成的文本。
- **Tokenizer Worker（分词工作进程）**：将输入文本转换为模型能够理解的数字，即 token id（词元编号）。
- **Detokenizer Worker（反分词工作进程）**：将模型生成的 token id 转换回人类可读的文本。
- **Scheduler Worker（调度工作进程）**：系统的核心工作进程。在多 GPU 配置下，每块 GPU 对应一个 Scheduler Worker；这里的编号称为 **TP Rank**，其中 TP 是 Tensor Parallelism（张量并行）的缩写。每个 Scheduler 管理其所在 GPU 的计算与资源分配。

### 数据流

各组件使用 **ZeroMQ（ZMQ）** 传递控制消息；多张 GPU 之间需要交换大量张量数据时，则通过 **NCCL**（由 `torch.distributed` 调用）通信。

![进程总览图](https://lmsys.org/images/blog/minisgl/design.drawio.png)

**请求生命周期：**

1. **用户**向 **API Server** 发送请求。
2. **API Server** 将请求转发给 **Tokenizer**。
3. **Tokenizer** 将文本转换为 token，并发送给 **Scheduler（Rank 0）**。
4. 若使用多 GPU，**Scheduler（Rank 0）** 会将该请求广播给其余 Scheduler。
5. **所有 Scheduler** 对请求进行调度，并触发本地 **Engine** 计算下一个 token。
6. **Scheduler（Rank 0）** 收集输出 token，并将其发送给 **Detokenizer**。
7. **Detokenizer** 将 token 转换为文本，再发送回 **API Server**。
8. **API Server** 以流式方式将结果返回给 **用户**。

## 代码组织（`minisgl` 包）

源代码位于 `python/minisgl`。下面按模块说明其面向开发者的职责：

- `minisgl.core`：提供核心数据类 `Req` 与 `Batch`，用于表示请求状态和批次状态；提供保存推理全局上下文的 `Context`；`SamplingParams` 用于保存用户传入的采样参数。
- `minisgl.distributed`：提供张量并行中的 all-reduce、all-gather 通信接口；`DistributedInfo` 数据类保存一个 TP 工作进程的张量并行信息。
- `minisgl.layers`：实现支持 TP 的大语言模型基础层，包括线性层、LayerNorm、Embedding、RoPE 等；这些层共享定义在 `minisgl.layers.base` 中的基础类。
- `minisgl.models`：实现具体的大语言模型，例如 Llama 和 Qwen3；同时提供从 HuggingFace 加载权重和切分权重的工具。
- `minisgl.attention`：提供 attention backend 的抽象接口，并实现 `flashattention`、`flashinfer` 等后端。它们由 `AttentionLayer` 调用，并读取存放在 `Context` 中的元数据。
- `minisgl.kvcache`：提供 KV Cache pool 和 KV Cache manager 的抽象接口，并实现 `MHAKVCache`、`NaiveCacheManager` 与 `RadixCacheManager`。KV Cache 保存模型注意力机制中需要复用的历史 Key/Value 张量。
- `minisgl.utils`：提供通用工具集合，包括日志初始化和对 ZMQ 的封装。
- `minisgl.engine`：实现 `Engine` 类。一个进程中的 TP worker 对应一个 Engine；它管理模型、上下文、KV Cache、attention backend 和 CUDA Graph replay。
- `minisgl.message`：定义 API Server、tokenizer、detokenizer 与 scheduler 之间通过 ZMQ 交换的消息。所有消息类型都支持自动序列化和反序列化。
- `minisgl.scheduler`：实现运行在每个 TP worker 进程中的 `Scheduler`，并管理对应的 `Engine`。Rank 0 Scheduler 从 tokenizer 接收消息，与其他 TP worker 上的 Scheduler 通信，并将结果发送给 detokenizer。
- `minisgl.server`：定义命令行参数以及用于启动 Mini-SGLang 全部子进程的 `launch_server`；还在 `minisgl.server.api_server` 中实现作为前端的 FastAPI 服务，提供 `/v1/chat/completions` 等端点。
- `minisgl.tokenizer`：实现 `tokenize_worker`，负责处理分词和反分词请求。
- `minisgl.llm`：提供 `LLM` 类，作为便于通过 Python 代码直接调用 Mini-SGLang 的接口。
- `minisgl.kernel`：实现自定义 CUDA kernel，并由 `tvm-ffi` 提供 Python binding（Python 绑定）与 JIT 接口。
- `minisgl.benchmark`：提供性能基准测试工具。
