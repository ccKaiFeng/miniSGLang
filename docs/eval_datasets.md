# miniSGLang 评测数据集说明

本文档说明当前仓库中 `experiment/` 目录下已经准备好的评测数据集和合成负载。数据下载脚本已经删除；云服务器只需要拉取当前分支即可获得这些数据。

默认数据目录：

```text
./experiment
```

## 已包含的数据

| 数据集 / 负载 | 保存位置 | 主要用途 |
| --- | --- | --- |
| GSM8K | `experiment/gsm8k/main` | 数学推理准确性测试 |
| CMMLU | `experiment/cmmlu/<subject_name>` | 中文多学科选择题准确性测试 |
| LongBench | `experiment/longbench/<task_name>` | 长上下文真实任务准确性和性能测试 |
| RULER SQuAD 数据 | `experiment/ruler/squad.json` | 长上下文问答 / 检索类测试数据 |
| generated-shared-prefix | `experiment/synthetic/generated_shared_prefix.jsonl` | 共享长 prefix 的服务性能负载，用于观察 prefix/radix cache 和 compressed KV 命中 |
| random token ids | `experiment/synthetic/random_token_ids.jsonl` | 纯性能压测负载，不用于准确率 |

## 云服务器同步方式

在云服务器仓库目录中执行：

```bash
git fetch origin
git checkout ZipCache
git pull origin ZipCache
```

拉取完成后，数据应位于：

```text
experiment/gsm8k/
experiment/cmmlu/
experiment/longbench/
experiment/ruler/
experiment/synthetic/
```

## 准确性测试数据

建议优先使用：

```text
GSM8K
CMMLU
LongBench
RULER SQuAD
```

这些数据用于比较原始 miniSGLang 与 ZipCache 版本在压缩 KV cache 后的回答正确性是否发生明显变化。

## 性能测试数据

建议优先使用：

```text
experiment/synthetic/generated_shared_prefix.jsonl
experiment/synthetic/random_token_ids.jsonl
LongBench 长输入任务
```

其中：

- `generated_shared_prefix.jsonl`：同组请求共享很长的前缀，适合观察 radix cache / prefix cache / compressed KV cache 的复用效果。
- `random_token_ids.jsonl`：绕开自然语言语义，主要用于压力测试吞吐、TTFT、TPOT、E2E、GPU 显存占用。
- LongBench：更接近真实长上下文任务，适合同时观察正确性和长输入性能。

## RULER 说明

RULER 原始仓库是一个独立 Git 仓库，当前没有把 `experiment/ruler/NVIDIA_RULER/` 作为外层仓库内容提交，避免形成嵌套 Git 仓库或误提交 RULER 源码。

当前外层仓库只提交了已经下载成功且可直接使用的数据文件：

```text
experiment/ruler/squad.json
```

如果后续需要完整 RULER benchmark，应在云服务器上单独 clone NVIDIA/RULER，并按照其 README 生成更多长上下文任务数据。

## Git 注意事项

本次为了方便云服务器直接 `git pull` 复现实验，`experiment/` 下的评测数据已经被提交到当前分支。

`.gitignore` 仍然会忽略以下内容，避免把运行日志、实验输出、嵌套 RULER 仓库继续提交进去：

```text
experiment/logs/
experiment/results/
experiment/ruler/NVIDIA_RULER/
datasets/
data/
```
