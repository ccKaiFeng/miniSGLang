# miniSGLang 评测数据集准备说明

本文档说明如何为 miniSGLang KV cache 压缩实验准备准确性测试数据和性能测试负载。

默认保存目录是当前项目目录下的：

```text
./experiment
```

也可以通过 `--root` 指定其他目录，例如：

```bash
python scripts/download_eval_datasets.py --root /root/autodl-tmp/datasets/minisgl_eval
```

## 数据集用途

| 数据集 / 负载 | 默认保存位置 | 主要用途 |
| --- | --- | --- |
| GSM8K | `experiment/gsm8k/main` | 数学推理准确性测试 |
| CMMLU | `experiment/cmmlu/<subject_name>` | 中文多学科选择题准确性测试 |
| LongBench | `experiment/longbench/<task_name>` | 长上下文真实任务准确性和长输入性能测试 |
| RULER | `experiment/ruler/NVIDIA_RULER` | 长上下文合成检索、多跳、聚合任务 |
| generated-shared-prefix | `experiment/synthetic/generated_shared_prefix.jsonl` | 共享长 prefix 的服务性能负载，用于观察 prefix/radix cache 和 compressed KV 命中 |
| random token ids | `experiment/synthetic/random_token_ids.jsonl` | 纯性能压测负载，不用于准确率 |

准确性测试重点使用：

```text
GSM8K
CMMLU
LongBench
RULER
```

性能测试重点使用：

```text
generated_shared_prefix.jsonl
random_token_ids.jsonl
LongBench 长输入任务
RULER 不同上下文长度任务
```

## 一键准备

安装依赖并下载/生成数据：

```bash
bash scripts/setup_eval_datasets.sh ./experiment
```

这个脚本会先安装：

```text
datasets
huggingface_hub
pandas
pyarrow
tqdm
```

然后调用：

```bash
python scripts/download_eval_datasets.py --root ./experiment
```

如果没有传 root 参数，默认也是 `./experiment`：

```bash
bash scripts/setup_eval_datasets.sh
```

## 直接运行 Python 脚本

只下载少量 CMMLU / LongBench config，跳过 RULER，适合先验证脚本：

```bash
python scripts/download_eval_datasets.py \
  --root ./experiment \
  --cmmlu-max-configs 3 \
  --longbench-max-configs 3 \
  --skip-ruler
```

只生成 synthetic 负载，不下载 Hugging Face 数据集，也不 clone RULER：

```bash
python scripts/download_eval_datasets.py \
  --root ./experiment \
  --skip-hf \
  --skip-ruler
```

如果不传 `--root`，脚本会使用当前执行目录下的 `experiment/`：

```bash
python scripts/download_eval_datasets.py \
  --cmmlu-max-configs 3 \
  --longbench-max-configs 3 \
  --skip-ruler
```

## Hugging Face 镜像

国内或云服务器访问 Hugging Face 较慢时，可以设置 `HF_ENDPOINT`：

```bash
HF_ENDPOINT=https://hf-mirror.com \
python scripts/download_eval_datasets.py \
  --root ./experiment \
  --cmmlu-max-configs 3 \
  --longbench-max-configs 3 \
  --skip-ruler
```

`setup_eval_datasets.sh` 也会继承当前环境变量：

```bash
HF_ENDPOINT=https://hf-mirror.com bash scripts/setup_eval_datasets.sh ./experiment
```

## 可跳过项

脚本支持按数据集跳过：

```text
--skip-hf          跳过所有 Hugging Face 数据集
--skip-gsm8k       跳过 GSM8K
--skip-cmmlu       跳过 CMMLU
--skip-longbench   跳过 LongBench
--skip-ruler       跳过 RULER clone 和数据脚本
--skip-synthetic   跳过 synthetic 负载生成
```

CMMLU / LongBench 支持快速调试参数：

```text
--cmmlu-max-configs 3
--longbench-max-configs 3
--no-longbench-e
```

## Synthetic 负载参数

`generated_shared_prefix.jsonl` 默认参数：

```text
--gsp-groups 64
--gsp-prompts-per-group 16
--gsp-prefix-len-words 2048
--gsp-question-len-words 128
--gsp-output-len 256
```

每一行格式：

```json
{
  "id": 0,
  "group_id": 0,
  "prompt": "...",
  "input_len_words_approx": 2300,
  "output_len": 256
}
```

同一个 `group_id` 内的 prompt 共享长 prefix，适合测试：

```text
prefix cache
radix cache
compressed KV cache hit
TTFT / E2E / TPOT
```

`random_token_ids.jsonl` 默认参数：

```text
--random-num-prompts 1024
--random-min-input-len 100
--random-max-input-len 1024
--random-min-output-len 100
--random-max-output-len 1024
--random-vocab-size 10000
```

每一行格式：

```json
{
  "id": 0,
  "prompt_token_ids": [123, 456, 789],
  "input_len": 512,
  "output_len": 256
}
```

这个文件主要用于性能压测，不用于判断模型回答正确性。

## RULER 注意事项

RULER 来自 NVIDIA 的长上下文评测仓库：

```text
https://github.com/NVIDIA/RULER.git
```

脚本会 clone 到：

```text
experiment/ruler/NVIDIA_RULER
```

clone 后会尝试运行当前已知的数据准备脚本：

```text
scripts/data/synthetic/json/download_paulgraham_essay.py
scripts/data/synthetic/json/download_qa_dataset.sh
```

如果 RULER 仓库结构变化，脚本只会打印 warning，不会因为找不到这些 helper 脚本而直接崩溃。此时需要手动查看 RULER 仓库 README。

## Git 注意事项

下载的数据集和生成的负载不应提交到 Git。仓库 `.gitignore` 已忽略常见数据目录和大文件格式：

```text
experiment/
datasets/
data/
*.arrow
*.parquet
```

已有被 Git 跟踪的实验脚本不会因为 `.gitignore` 自动消失；`.gitignore` 只影响新的未跟踪文件。
