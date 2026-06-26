# Compressed KV Cache Demotion

这是一个 Python/runtime 层的实验原型，用来验证“KV cache 即将释放或被驱逐时，先保存压缩归档；未来命中后尝试恢复，失败则回退到原始 prefill/recompute”的流程。

当前版本不修改 CUDA kernel，不修改 attention 数学逻辑，也不会让 attention 直接读取 int8 KV。默认 `mock` codec 只保存 metadata 和日志，不保存真实 K/V tensor，因此 restore 一定 fallback 到 recompute，生成正确性不依赖 compressed archive。

## 启动方式

```bash
python -m minisgl \
  --model-path Qwen/Qwen2.5-0.5B-Instruct \
  --host 0.0.0.0 \
  --port 30000 \
  --enable-compressed-kv-cache \
  --compressed-kv-cache-dir /root/autodl-tmp/kv_archive \
  --compressed-kv-cache-codec mock \
  --compressed-kv-cache-max-size-mb 4096 \
  --compressed-kv-cache-restore-policy cost
```

如果要用项目原来的默认端口，也可以省略 `--host` 和 `--port`。

## 参数说明

- `--enable-compressed-kv-cache`：打开实验功能。不开启时不会创建 archive 目录，也不会打印 `[CompressedKV]` 日志。
- `--compressed-kv-cache-dir`：metadata 保存目录。每个 entry 保存为 `entry_xxx.json`。
- `--compressed-kv-cache-codec`：当前支持 `mock` 和 `int8_cpu`。`int8_cpu` 第一版只保留 TODO metadata，不保存真实 tensor。
- `--compressed-kv-cache-max-size-mb`：归档大小上限。mock 不保存 tensor，通常不会触发容量回收。
- `--compressed-kv-cache-restore-policy`：restore 尝试策略，支持 `cost`、`always`、`never`。

## 触发 demotion

KV cache demotion 接在两个位置：

1. 请求结束或 prefix cache 复用导致当前请求私有 page 要释放时；
2. free page 不够，radix prefix cache 发生 eviction 时。

正常观察日志：

```text
[CompressedKV] eviction detected: entry_id=..., num_tokens=..., estimated_bytes=...
[CompressedKV] demoted: entry_id=..., codec=mock, original_bytes=..., compressed_bytes=0
```

如果想更容易触发 eviction，可以调小 KV cache page 数：

```bash
python -m minisgl \
  --model-path Qwen/Qwen2.5-0.5B-Instruct \
  --num-pages 128 \
  --enable-compressed-kv-cache \
  --compressed-kv-cache-codec mock
```

也可以构造多个长 prompt 请求，让 prefix cache 挤占更多 page。

## 构造 shared-prefix 请求

发送多次包含相同长前缀的请求，例如相同 system prompt 或相同长文档开头。第一次请求结束或 cache eviction 后会生成 archive metadata；后续请求如果前缀 hash 命中，会看到：

```text
[CompressedKV] compressed hit: entry_id=..., num_tokens=...
[CompressedKV] restore attempt: entry_id=..., policy=cost
[CompressedKV] restore fallback to recompute: entry_id=...
```

当前 mock codec 不恢复真实 KV，所以命中后仍继续走原始 prefix cache / prefill 路径。

## 验证 feature flag 关闭

不带 `--enable-compressed-kv-cache` 启动：

```bash
python -m minisgl --model-path Qwen/Qwen2.5-0.5B-Instruct
```

预期行为：

- 不创建 compressed KV archive 目录；
- 日志中不出现 `[CompressedKV]`；
- KV page 分配、prefix cache 命中、eviction、attention forward 行为保持原样。

## 当前限制

- `mock` codec 只保存 metadata，不保存真实 K/V tensor。
- `int8_cpu` codec 目前没有真实 tensor dump/restore，原因是第一版没有实现安全写回原 KV pool 的路径。
- restore 返回 `None` 时必须 fallback 到 recompute；当前代码就是这个行为。
- radix eviction 会记录被驱逐节点的 token 前缀和 KV indices，用于 request-level compressed hit 检测，但不会阻止 GPU KV page 被释放。

## 后续 TODO

1. 在明确 KV pool layout 和写回接口后，为 `int8_cpu` 实现 tensor dump。
2. 为 restore 增加“分配新 page -> 解压到 GPU -> 写 page_table -> 返回可用 handle”的闭环。
3. 增加更精确的容量管理，按 `last_access_time` 删除冷归档。
4. 增加单元测试覆盖 mock demotion、metadata reload、compressed hit 和 fallback。
