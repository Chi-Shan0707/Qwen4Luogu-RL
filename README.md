# LuoguQwen-RL — TinyLoRA Experiment

<div align="center">

**强化学习训练超参数压缩模型：Qwen2.5-Coder on Luogu OJ**

[中文版本](#中文版本) | [English Version](#english-version)

如果这个项目对你有帮助，或者你觉得有点意思，请点击右上角的 Star 支持一下！这对我很重要，万分感谢PwP！<br>
If you find this project useful or interesting, please give it a Star! 🌟 Your support is my greatest motivation.<br>
</div>

---

# 中文版本

## LuoguQwen-RL — TinyLoRA 实验

本仓库是原「LuoguQwen LoRA 微调」，一个[基于 SFT的项目](https://github.com/Chi-Shan0707/Qwen4Luogu-SFT)的进化版：

> 什么，你问我为什么要挑选 Qwen2.5-1.5B-Instruct 进行微调？<br>
> —— 那当然是因为它参数量小啦。<br>
>
> 什么，你继续问我为什么不挑选 Qwen2.5-Coder-1.5B-Instruct 进行微调？<br>
> ~~我如果在这阿里进行过代码训练上的模型进行微调，哪能看得出我微调的效果？~~<br>
> ~~好吧，其实是我问千问有什么参数量小的模型，它推荐了这个，然后我一时间忘记继续去搜集信息，直接开搞惹，结果训练到一半才在 ModelScope 上刷到 Qwen2.5-Coder-1.5B-Instruct。PWP~~<br>
> ~~第一遍实在太差了，反正还要再训练一遍，还是弄 Qwen2.5-Coder-1.5B-Instruct 吧~~<br>
> 这个也太差劲了，上 7B 吧 PwP<br>
> *不对，为什么疯狂报 mismatch 啊啊？从 1.5B→7B 我啥都没改啊？*<br>
> *疯狂 debug，疯狂研究格式……*<br>
> 算了，格式弄成所谓的标准型吧。<br>
> 7B 根本跑不动啊，只能 3B。<br>
> ~~啊训练完了，参数根本上传不动啊？啊，huggingface 也上传不动啊 PwP~~<br>

然后，6号晚上，~~天助我也~~，我看到了TinyLoRA的论文，所以我就开始了这项尝试（或者可以说“复现”）：
- 基座：Qwen2.5-Coder-3B-Instruct，4bit 量化以挤爆最后一点显存；
- 训练：不用 SFT，用 RL（GRPO）；
- 参数：全模型只保留 **16 个可训练标量参数**；
- 任务：用「编译+运行 C++ 代码」的方式在洛谷题目上搞代码强化学习。
<br>

目前，这个`train_rl.py`是可以运行且训练的，但是能成功运行+通过样例测试的，十不存一（并没有夸张）。<br>
原因可能有:
- 提示词写的不好，下一步需要明确【是否要推理路径】等细节，并开展Prompt Engineering
- token数量截取的太少，目前是1024，但是这个也会带来成本
- GRPO时生成答案数量太少
- luogu题目太难
- RL的reward写的不够好
- 3B模型比较差<br>

<br>

`train_rl.py`中支持修改
- 更换模型，如采用Qwen2.5-7B
- GRPO的config
- reward
- ...

---

## 目录

- [项目概述](#项目概述)
- [论文复现](#论文复现)
- [核心特点](#核心特点)
- [快速开始](#快速开始)
- [数据准备与格式](#数据准备与格式)
- [训练流程（RL / GRPO）](#训练流程rl--grpo)
- [TinyLoRA Tiling 技术细节](#tinylora-tiling-技术细节)
- [奖励函数：编译运行 C++ 代码](#奖励函数编译运行-c-代码)
- [资源消耗与注意事项](#资源消耗与注意事项)
- [开源与许可证](#开源与许可证)
- [引用](#引用)


---

## 项目概述

LuoguQwen-RL 的目标是：

> 在显存受限（3B 模型 + 4bit 量化）且参数极致压缩（仅 16 个参数）的前提下，
> 通过强化学习让 Qwen2.5-Coder 在洛谷竞赛题上学会「能过样例」的 C++ 代码生成。

本仓库并不是凭空设计的，而是一个**TinyLoRA 论文方向的复现与变体实验**：

- `theory/README.md` 中给出了 TinyLoRA / GRPO 的理论与工程细节梳理；
- 本项目在此基础上，将 TinyLoRA 的思想从数学推理（如 GSM8K）迁移到**代码生成 + 编译执行奖励**场景；
- 论文中经典设置是 7B 模型 + 13 个参数，本仓库使用 3B Coder 模型 + 16 个参数，保持「极低秩 + 全局共享」这一精神内核。

核心脚本：

- `train_rl.py`：
  - 加载 4bit 量化的 `Qwen2.5-Coder-3B-Instruct`；
  - 将指定 Linear 层替换为自定义 `TinyLoRALinear`，并通过共享向量 `global_v` 实现 TinyLoRA Tiling；
  - 使用 TRL 的 `GRPOTrainer` 进行代码强化学习；
  - 奖励来自本地 `g++` 编译 + 测试用例执行通过率。
- `convert_dataset.py`：
  - 从本地洛谷题目数据（Markdown 风格）中用正则抽取 `prompt`（题面）与 `test_cases`（输入输出样例）；
  - 过滤掉包含中文的样例，转存为 JSONL，供 RL 训练使用。
- `download_dataset.py`：
  - 从 Hugging Face 下载 DPO 格式的洛谷数据集并保存到 `./local_luogu_dpo`（供 `convert_dataset.py` 使用）。
- `verify_pipeline.py`：
  - 用于验证模型加载、生成、代码提取与编译运行的端到端流水线（示例：加载模型并尝试用给定样例对生成代码进行编译运行评测）。

目录结构（节选）：

- `train_rl.py`：主训练脚本（TinyLoRA + GRPO）。
- `download_dataset.py`：从 Hugging Face 下载 DPO 格式数据并保存到 `./local_luogu_dpo`。
- `verify_pipeline.py`：验证 model->generate->extract->compile 流程的脚本。
- `convert_dataset.py`：将本地洛谷 DPO 数据转为 RL JSONL 格式。
- `local_luogu_dpo/`：从原 DPO 数据集转存的本地数据（`load_from_disk` 产物）。
- `local_luogu_rl/luogu_rl_data.jsonl`：RL 训练数据（`convert_dataset.py` 输出）。
- `models/Qwen2.5-Coder-3B-Instruct/`：基座模型目录（可通过 ModelScope 自动下载）。
- `output/`：RL 训练输出目录（包括最终的 `tiny_lora_v.pt`，内含 `global_v` 向量及重建所需的元信息）。

---
## 论文复现

[cite_start]本项目是论文 **"Learning to Reason in 13 Parameters" (Morris et al., 2026)** 的非官方复现与工程适配 [cite: 2]。

### 1. 核心理论：TinyLoRA
原论文提出了一种极端的参数高效微调方法 **TinyLoRA**，旨在打破 LoRA 的秩（Rank）限制。
- [cite_start]**痛点**：传统 LoRA 即使 Rank=1，其参数量仍与模型宽度 $d$ 成正比（$O(d \times r)$），对于 7B 模型约为数百万参数 [cite: 17, 158]。
- [cite_start]**创新**：TinyLoRA 利用 SVD 冻结原权重的特征方向 ($U, V$)，仅学习一个极小的向量 $v$。通过在不同层之间共享这个向量（**Tiling**），可将全网可训练参数压缩至个位数 [cite: 7, 175, 181]。
- **公式**：
  $$W' = W + U \Sigma (\sum_{i=1}^{u} v_i P_i) V^\top$$
  [cite_start]其中 $U, \Sigma, V$ 来自原权重的 SVD 分解（冻结），$P$ 是固定随机投影，$v$ 是唯一的可训练参数 [cite: 173, 174]。

### 2. 为什么必须是 RL？
[cite_start]论文的核心发现是：**在如此极端的参数限制下（<100 参数），SFT（监督微调）几乎完全失效，只有 RL（强化学习）能奏效** [cite: 10, 65]。
- [cite_start]**SFT 的局限**：SFT 强迫模型记忆参考答案的格式和风格（"Noise"），这需要较大的容量 [cite: 147, 148]。
- [cite_start]**RL 的优势**：RL 仅关注最终结果的对错（"Signal"），允许模型忽略无关细节。TinyLoRA 正是利用这一点，在仅有 13 个参数的情况下，通过 GRPO 算法在 GSM8K 上达到了 91% 的准确率 [cite: 64, 149]。

### 3. 本项目的“魔改”适配
我们遵循论文的精神内核，但针对**代码生成任务**和**消费级显卡**进行了适配：

| 特性 | 原论文设置 (Paper) | 本项目适配 (Ours) |
| :--- | :--- | :--- |
| **任务领域** | [cite_start]数学推理 (GSM8K, MATH) [cite: 8] | **代码竞赛 (Luogu OJ)** |
| **基座模型** | [cite_start]Qwen2.5-7B / Llama-3 [cite: 64] | **Qwen2.5-Coder-3B-Instruct** |
| **参数量** | 13 参数 ($u=13$) | **16 参数 ($u=16$)** |
| **精度处理** | [cite_start]BF16 / FP32 [cite: 8] | **4-bit 量化 (NF4) + 动态反量化 SVD** |
| **奖励机制** | 答案匹配 (Exact Match) | **g++ 编译 + 测试用例运行 (RLVR)** |
| **显存优化** | 需高显存 (A100/H100) | **适配单卡消费级 GPU (16GB+)** |

> **关键工程挑战**：原论文未涉及 4-bit 量化模型。本项目额外实现了在初始化阶段对 4-bit 权重进行 `dequantize` 解包，在 CPU 上完成 FP32 精度的 SVD 分解，再转回 BF16 注册为 Buffer 的流程，从而在低显存环境下实现了 TinyLoRA 初始化。

## 核心特点

- **极致参数压缩**：
  - 整个模型的可训练参数只有一个向量 `global_v ∈ R^{16}`；
  - 全网所有被替换的 Linear 层都共享这 16 个标量；
  - 你可以通过运行 `train_rl.py` 或 `verify_pipeline.py` 来查看模型参数信息（总参数量 / 可训练参数量 / 压缩率）。

- **TinyLoRA Tiling**：
  - 对原始 Linear 权重（包括 4bit 量化权重）做 SVD 分解，得到固定的骨架 `U, S, Vh`；
  - 再通过随机矩阵簇 `P ∈ R^{u×r×r}` 与共享向量 `v ∈ R^u` 重构一个低秩增量；
  - 只训练 `v`，实现论文中的 Tiling / 全参数共享。

- **真实代码环境奖励**：
  - 把模型生成的 C++ 代码写入临时文件；
  - 使用系统 `g++` 编译；
  - 三档离散 reward：编译失败=0，编译成功但样例错误=0.5，通过样例=1.0；
  - 代码不通过编译 / 超时 / 运行错误 -> reward 直接趋近于 0。

- **显存友好**：
  - 基座为 3B Coder 模型，结合 bitsandbytes 4bit 量化 + BF16 计算；
  - 在单卡有限显存环境下也能跑完整的 RL loop（当然，会比较慢）。

---

## 快速开始

### 1. 环境准备

建议使用 Linux + Python 3.10 及以上版本，并确保已安装 `g++` 编译器。

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

> 提示：`requirements.txt` 中已包含 `torch`、`transformers`、`datasets`、`trl`、`peft`、`bitsandbytes`、`modelscope` 等依赖。

### 2. 下载基座模型

`train_rl.py` 会在本地不存在模型时，自动通过 ModelScope 下载：

- 模型 ID：`qwen/Qwen2.5-Coder-3B-Instruct`
- 默认本地路径：`./models/Qwen2.5-Coder-3B-Instruct`

你也可以显式调用：

```python
from modelscope.hub.snapshot_download import snapshot_download

snapshot_download(
    repo_id="qwen/Qwen2.5-Coder-3B-Instruct",
    local_dir="./models/Qwen2.5-Coder-3B-Instruct",
)
```

### 3. 准备洛谷数据（DPO → RL）

假设你已经有一个从 Hugging Face / ModelScope 下载的洛谷 DPO 数据集，并通过 `datasets` 的 `load_from_disk` 保存到了本地 `./local_luogu_dpo/` 目录（目录下含 `state.json` 等文件）。

运行：

```bash
python convert_dataset.py
```

`convert_dataset.py` 会：

- 从 `./local_luogu_dpo` 中读取 `train` split；
- 提取 `item["conversations"][0]["value"]` 作为题目描述；
- 用正则在题面中匹配「样例输入 / 输出」代码块；
- 丢弃包含中文字符的样例（只保留纯数字 / 英文 / 符号的样例）；
- 将结果写入 `./local_luogu_rl/luogu_rl_data.jsonl`；
- 同时把提取失败的题面写入 `./local_luogu_rl/failed_extraction.jsonl` 以便人工排查。

### 4. 可选：验证流水线与数据下载

在真正训练前，可执行以下脚本进行检查与准备：

- 下载 DPO 数据集（如果你还没下载）：

```bash
python download_dataset.py
```

- 验证端到端流水线（模型加载、生成、代码提取与编译运行的示例）：

```bash
python verify_pipeline.py
```

`verify_pipeline.py` 会加载 tokenizer 和模型（若本地不存在则使用远端 ID），对预设 JSON 示例生成代码、提取并尝试编译运行样例，从而帮助你验证环境是否完整（例如是否安装 `g++`、模型与 tokenizer 配置是否正确等）。

说明 TinyLoRA 注入与参数冻结逻辑是正常的。

### 5. 启动 RL 训练

基础用法（使用默认u=16，训练全部数据）：

```bash
python train_rl.py
```

自定义 TinyLoRA 参数数量（u 值）：

```bash
python train_rl.py 32     # 使用 u=32（32 个可训练参数）
python train_rl.py 8      # 使用 u=8（8 个可训练参数）
```

限制训练样本数量：

```bash
python train_rl.py 16 100      # u=16，仅训练前 100 个样本
python train_rl.py 32 50       # u=32，仅训练前 50 个样本
python train_rl.py 16          # u=16，训练全部样本（第二个参数可省略）
```

> **参数说明**：
> - **第一个参数** `u`：TinyLoRA 中共享向量 `global_v` 的维度，即可训练参数的总数。若不提供，则默认使用 `u=16`。
> - **第二个参数** `MAX_SAMPLES`：最多训练的样本数量。若不提供，则使用全部数据集。这个参数在快速实验、调试超参数或显存不足时非常有用。 目前是shuffle后取数据集（原数据集每道题目会出现2次），故如此。（目前种子是全局的TINYLORA_SEED。）

`train_rl.py` 将会：

1. 确保基座模型已准备好（必要时自动下载）；
2. 以 4bit 量化方式加载 `Qwen2.5-Coder-3B-Instruct`；
3. 根据命令行参数创建 u 维的共享向量；
4. 注入 TinyLoRA Tiling（全局共享 `global_v`）；
5. 从 `./local_luogu_rl/luogu_rl_data.jsonl` 读取 RL 数据；
6. 若指定了 `MAX_SAMPLES`，则仅选取前 N 个样本进行训练；
7. 使用 `GRPOTrainer` 进行强化学习；
8. 训练完成后，将训练结果保存为 `output/tiny_lora_v.pt`。

**保存内容**：`tiny_lora_v.pt` 是一个 dict，包含还原模型所需的全部信息：

```python
{
    "global_v": tensor([...]),     # 训练好的 v 向量，shape=(u,)
    "u_value": 32,                 # v 的维度
    "rank": 2,                     # TinyLoRA 的 rank
    "seed": 42,                    # P 矩阵的随机种子（用于复现）
    "model_id": "qwen/Qwen2.5-Coder-3B-Instruct",  # 基座模型 ID
    "total_replaced_layers": 252,  # 替换的层数
}
```

> **还原方式**：加载基座模型 → 用相同 `seed` 固定随机种子 → 用相同 `u_value` 和 `rank` 执行 `apply_tiny_lora` → 将 `global_v` 加载回 `global_params.global_v`。种子相同保证 P 矩阵完全一致，SVD 是确定性运算所以 U/S/Vh 也一致。

如果你想自定义输出目录，可以修改 `train_rl.py` 顶部的：

```python
OUTPUT_DIR = "./output/luoguqwencoder-lora"
```

---

## 数据准备与格式

### 上游数据：洛谷题目（DPO 版）

- **数据来源**：Hugging Face 数据集 `Misaka114514/luogu_dpo`（https://huggingface.co/datasets/Misaka114514/luogu_dpo）。*在此谢过！*

原始数据形态大致为：

- 字段 `conversations`：一个对话列表；
- 通常 `conversations[0]["value"]` 是题目描述（Markdown 风格）；
- 题目中包含类似：

```markdown
**输入：**
```text
... 样例输入 ...
```

**输出：**
```text
... 样例输出 ...
```
```

`convert_dataset.py` 会从这里抽取测试用例。

### RL 训练数据：JSONL 格式

`convert_dataset.py` 生成的 `luogu_rl_data.jsonl` 中，每一行是一条 JSON，对应一题：

```json
{
  "prompt": "<完整题目描述，通常是 Markdown 文本>",
  "test_cases": [
    {"input": "<样例输入 1>", "output": "<样例输出 1>"},
    {"input": "<样例输入 2>", "output": "<样例输出 2>"}
  ]
}
```

在 `train_rl.py` 中通过：

```python
from datasets import load_dataset

rl_dataset = load_dataset(
    "json",
    data_files="./local_luogu_rl/luogu_rl_data.jsonl",
    split="train",
)
```

直接作为 `GRPOTrainer` 的 `train_dataset` 使用。

---

## 训练流程（RL / GRPO）

核心训练逻辑位于 `train_rl.py`：

1. **模型加载与量化**
   - 使用 `BitsAndBytesConfig`：
     - `load_in_4bit=True`
     - `bnb_4bit_quant_type="nf4"`
     - `bnb_4bit_use_double_quant=True`
     - `bnb_4bit_compute_dtype=torch.float16`
   - 通过 `device_map="auto"` 将模型自动切分到可用 GPU。

2. **TinyLoRA 注入与参数冻结**
   - 创建全局共享向量（维度由命令行参数 `u` 决定，默认16）：
     - `global_v = nn.Parameter(torch.zeros(U_VALUE))`
   - 通过 `apply_tiny_lora(model, global_v)`：
     - 遍历模型子模块；
     - 找到名字以 `q_proj / k_proj / v_proj / o_proj / gate_proj / up_proj / down_proj` 结尾的 `nn.Linear`；
     - 替换为 `TinyLoRALinear`；
   - 随后：
     - 仅保留 `global_v` 的 `requires_grad=True`；
     - 其他所有参数全部 `requires_grad=False`。

3. **GRPO 配置**

`train_rl.py` 中使用的示例超参数：

- `num_train_epochs=1`
- `per_device_train_batch_size=1`
- `gradient_accumulation_steps=8`
- `learning_rate=1e-5`
- `num_generations=4`（Group Size G，每个样本采样 4 个答案）
- `max_completion_length=512`
- `bf16=True`

你可以根据显存与训练时间需求调整上面的参数。

4. **训练循环**

GRPO 的整体流程简要为：

- 对于每个样本 `prompt`：
  1. 采样多个 `completions`（C++ 代码）；
  2. 调用 `code_reward_func` 对每个 completion 编译 + 运行，得到 reward；
  3. 使用 GRPO 算法根据 reward 更新策略（这里就是更新 16 维的 `global_v`）。

---

## TinyLoRA Tiling 技术细节

自定义层 `TinyLoRALinear` 的核心思想：

1. 对原始权重矩阵 `W ∈ R^{out×in}` 做 SVD：

   $$W = U S V^H$$

   - 实现中先将 4bit 权重反量化为 `W_real`，再在 CPU 上做 `torch.linalg.svd`；
   - 只取前 `rank=2` 个奇异值及对应的列 / 行，得到精简版 `U, S, V^H`；
   - 这些张量通过 `register_buffer` 注册为 Buffer，不参与训练。

2. 定义全局共享参数：

   - `v ∈ R^u`，其中 `u=16`；
   - 随机初始化一组固定矩阵簇 `P ∈ R^{u×r×r}`；
   - 构造：

     $$R = \sum_{i=1}^{u} v_i P_i \in R^{r×r}$$

3. 构造增量权重：

   - $$\Delta W = U S R V^H$$
   - 实际前向中计算：

     $$y = x W^T + x (\Delta W)^T$$

4. Tiling（跨层共享）

   - 模型中所有目标 `nn.Linear` 层都共享同一个 `v`；
   - 整个模型只有这一组 16 维参数在更新。

你可以通过 `verify_pipeline.py` 或直接观察 `train_rl.py` 的启动日志来确认 TinyLoRA 注入是否正确并检查可训练参数量。

---

## 奖励函数：编译运行 C++ 代码

奖励函数实现位于 `train_rl.py` 中的 `code_reward_func` 与 `compile_and_run`：

1. **从模型输出中提取代码**
   - 优先匹配形如：

     ```markdown
          ```cpp
          // C++ 代码
          ```
     ```

   - 若没有显式代码块，则回退为只要包含 `#include` 的裸代码段；
   - 若仍无法识别，则直接给 0 分。

2. **编译阶段**
   - 将代码写入临时目录中的 `solution.cpp`；
   - 通过正则删除代码中的 `freopen(...)` 等文件重定向语句，改用标准输入输出；
   - 使用：

     ```bash
     g++ solution.cpp -o solution -O2
     ```

   - 编译失败 / 超时 -> 本次样本 reward = 0。

3. **运行阶段**
   - 对每个测试用例：
     - 将 `case["input"]` 作为 stdin；
     - 捕获 stdout，与 `case["output"]` 进行字符串级比对（`strip()` 后）；
   - 运行有超时保护（例如 2 秒），防止死循环卡死训练。

4. **打分规则**

   奖励函数采用三档评分制：

   - **编译失败** 或 **代码格式无效**：`reward = 0`
     - 包括编译错误、编译超时、无法提取代码块等情况；
   
   - **编译成功但测试用例失败**：`reward = 0.5`
     - 代码能通过 g++ 编译，但运行后不能通过全部样例测试（可能通过部分或全部失败）；
   
   - **编译成功且通过所有测试用例**：`reward = 1.0`
     - 代码既能编译成功，也能在所有提供的样例上产生正确输出。

   **核心强化信号**：
   - 这种设计鼓励模型先学会生成「能编译的代码」（0 → 0.5 的进步），
   - 然后在编译基础上进一步优化逻辑以通过测试用例（0.5 → 1.0 的进步）。
   - 相比连续打分，离散 reward 提供了更清晰的学习阶段划分。

> 这意味着模型不仅要「看起来像 C++」，还要真的能通过样例输入输出，
> 强化信号来自真实的编译器与运行环境，而非静态打分。

---

## 资源消耗与注意事项

- **显存**：
  - 3B 模型 + 4bit 量化 + BF16 计算，单卡 16GB 显存可以尝试（但余量不算大）；
  - RL + 编译运行会显著增加时间消耗，训练速度会比传统 LoRA SFT 慢很多。

- **操作系统**：
  - 推荐 Linux 环境（当前脚本在 Linux 下开发与测试）；
  - 需要可用的 `g++`，并且能够在临时目录下创建与执行可执行文件。

- **安全**：
  - 强烈不建议对不受信任的数据集运行此奖励函数；
  - 本项目的假设是「数据集来源可信」且仅用于研究环境。

---

## 开源与许可证

- 本仓库脚本默认采用 MIT 许可证（见 `LICENSE`）。
- 基座模型 `Qwen2.5-Coder-3B-Instruct` 由第三方（Qwen 团队）提供，请遵守其原始许可证；
- 本仓库不分发完整基座模型权重，只提供：
  - TinyLoRA / RL 相关代码；
  - 数据处理脚本；
  - 可选的 TinyLoRA 参数文件（`tiny_lora_v.pt`，内含训练好的 `global_v` 向量及重建模型所需的元信息：u 值、rank、随机种子等）。

---

## 引用



```bibtex
@article{morris2026learning,
  title={Learning to Reason in 13 Parameters},
  author={Morris, John X and Mireshghallah, Niloofar and Ibrahim, Mark and Mahloujifar, Saeed},
  journal={arXiv preprint arXiv:2602.04118},
  year={2026}
}
```

---

## English Version

### LuoguQwen-RL — TinyLoRA Experiment

LuoguQwen-RL is an evolution of the original [LuoguQwen SFT project](https://github.com/Chi-Shan0707/Qwen4Luogu-SFT).

The goal of LuoguQwen-RL is:
> Under the constraints of limited VRAM (3B model + 4bit quantization) and extreme parameter compression (only 16 parameters), train Qwen2.5-Coder through Reinforcement Learning (RL) to generate C++ code that passes sample tests on Luogu competitive programming problems.

This repository is an **unofficial reproduction and adaptation of the TinyLoRA paper**:
- `theory/README.md` provides theoretical insights into TinyLoRA / GRPO.
- We extend TinyLoRA from mathematical reasoning (GSM8K) to **code generation + compile-and-run rewards**.
- While the paper uses 7B models with 13 parameters, we use a 3B Coder model with 16 parameters, maintaining the "extreme low-rank + global sharing" core philosophy.

**Core Scripts:**
- `train_rl.py`: Main training script using 4-bit `Qwen2.5-Coder-3B-Instruct`, TinyLoRA Tiling, and `GRPOTrainer` with `g++` rewards.
- `convert_dataset.py`: Extracts `prompt` and `test_cases` from local Luogu DPO data (Markdown style) and filters non-ASCII samples.
- `download_dataset.py`: Downloads Luogu DPO dataset from Hugging Face.
- `verify_pipeline.py`: Validates the end-to-end flow of model loading, generation, extraction, and compilation.

### Paper Reproduction

This project is based on the paper **"Learning to Reason in 13 Parameters" (Morris et al., 2026)**.

#### 1. Core Theory: TinyLoRA
TinyLoRA is an extreme parameter-efficient fine-tuning method that breaks the rank limits of traditional LoRA ($O(d \times r)$). By freezing the original weight's characteristic directions ($U, V$) via SVD and learning only a tiny shared vector $v$ (**Tiling**), it compresses trainable parameters to single digits.

#### 2. Why Reinforcement Learning?
At such extreme parameter scales (<100 parameters), Supervised Fine-Tuning (SFT) often fails because it forces the model to memorize noise (formatting/styles). RL instead focuses on the "Signal" (correctness), allowing the model to ignore irrelevant details and succeed even with minimal capacity.

#### 3. Adaptation Table

| Feature | Paper Setting | Our Adaptation |
| :--- | :--- | :--- |
| **Domain** | Math Reasoning (GSM8K, MATH) | **Code Competitions (Luogu OJ)** |
| **Base Model** | Qwen2.5-7B / Llama-3 | **Qwen2.5-Coder-3B-Instruct** |
| **Parameters** | 13 parameters ($u=13$) | **16 parameters ($u=16$)** |
| **Precision** | BF16 / FP32 | **4-bit (NF4) + Dynamic Dequant SVD** |
| **Reward** | Exact Match | **g++ Compile + Test Case Execution** |
| **Optimization**| High-end GPUs (A100/H100) | **Consumer GPUs (16GB+ VRAM)** |

### Core Features

- **Extreme Parameter Compression**: The entire model's trainable parameters consist of a single vector `global_v ∈ R^{16}` shared across all replaced linear layers.
- **TinyLoRA Tiling**: Freezes the base skeleton (`U, S, Vh`) from SVD and reconstructs low-rank increments via a shared vector `v`.
- **Real-world Code Reward**:
  - Compiles generated code with system `g++`.
  - Discrete rewards: `0` for failure, `0.5` for compilation success, `1.0` for passing all test cases.
- **VRAM Friendly**: Optimized for 16GB+ single-GPU setups using 4-bit quantization and BF16 computation.

### Quick Start

#### 1. Environment
Requires Linux, Python 3.10+, and `g++`.
```bash
pip install -r requirements.txt
```

#### 2. Model Download
`train_rl.py` auto-downloads `qwen/Qwen2.5-Coder-3B-Instruct` to `./models/` if missing.

#### 3. Data Preparation
```bash
python download_dataset.py  # Download DPO data
python convert_dataset.py   # Convert to RL (JSONL) format
```

#### 4. Verification
```bash
python verify_pipeline.py    # Verify the model->generate->compile loop
```

#### 5. Start RL Training
```bash
python train_rl.py [u] [MAX_SAMPLES]
```
- **`u`**: Shared vector dimension (default 16).
- **`MAX_SAMPLES`**: Max number of samples to train (default all).

Training saves `output/tiny_lora_v.pt` containing `global_v` and reconstruction metadata (seed, rank, model_id).

### Data Preparation and Format

- **Source**: DPO-formatted Luogu data — Hugging Face dataset `Misaka114514/luogu_dpo` (https://huggingface.co/datasets/Misaka114514/luogu_dpo).*Thank you, Misaka114514!*
- **RL Format**: Each line in `luogu_rl_data.jsonl` contains:
  - `prompt`: Problem statement in Markdown.
  - `test_cases`: List of dictionary pairs with `input` and `output`.

### Training Process (RL / GRPO)

1. **Quantization**: Loads base model in 4-bit (NF4) with double quantization.
2. **Injection**: Replaces `q_proj`, `k_proj`, etc., with `TinyLoRALinear`.
3. **GRPO**: Uses Group Relative Policy Optimization. Each prompt samples multiple completions (G=4), evaluated by the `code_reward_func`.

### TinyLoRA Tiling Technical Details

- **SVD Integration**: 4-bit weights are dequantized to FP32 on CPU for SVD decomposition. The top components are stored as frozen buffers.
- **Increment Construction**: $\Delta W = U S (\sum_{i=1}^{u} v_i P_i) V^H$, where $P$ is a fixed random projection cluster.
- **Global Sharing**: Every injected layer references the same `global_v`.

### Reward Function: Compile and Run C++ Code

1. **Extraction**: Regex matching for code blocks or standard `#include` snippets.
2. **Compilation**: Strips `freopen` to use standard I/O; runs `g++ -O2`.
3. **Scoring Logic**:
   - `0`: Compilation error or invalid format.
   - `0.5`: Successfully compiled but failed tests (partial or full).
   - `1.0`: Successfully passed all sample cases.
   This provides a clear gradient: Learn to compile first, then learn to solve.

### Resource Consumption and Notes

- **VRAM**: 16GB is the baseline recommendation.
- **Performance**: Slower than SFT due to overhead from sampling and external compiler calls.
- **Safety**: Reward function executes compiled binaries; only use with trusted datasets.

### License and Citation

- Code: MIT License.
- Base Model: Qwen License.

If you find this project useful, please consider citing the original paper:

```bibtex
@article{morris2026learning,
  title={Learning to Reason in 13 Parameters},
  author={Morris, John X and Mireshghallah, Niloofar and Ibrahim, Mark and Mahloujifar, Saeed},
  journal={arXiv preprint arXiv:2602.04118},
  year={2026}
}
```