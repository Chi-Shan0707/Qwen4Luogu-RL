# RL Training Data — Luogu Dataset Format

<div align="center">

**洛谷题目 RL 训练数据集说明**

[中文版本](#中文版本) | [English Version](#english-version)

</div>

---

# 中文版本

## 洛谷 RL 训练数据集

本目录包含通过 `convert_dataset.py` 从洛谷 DPO 数据集转换而来的强化学习训练数据，共有两个重要文件：

1. **`luogu_rl_data.jsonl`**：成功提取的题目及测试用例
2. **`failed_extraction.jsonl`**：提取失败的题目及失败原因

### 1. luogu_rl_data.jsonl — 成功的训练数据

#### 文件格式

每一行是一个 JSON 对象，对应一道竞赛题目及其测试用例。

#### 字段说明

| 字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| `prompt` | string | 完整的题目描述，包括题意、输入输出格式、样例说明等。由 `convert_dataset.py` 在题面前加入指令前缀。 |
| `test_cases` | array | 测试用例数组，每个元素是包含 `input` 和 `output` 的对象。 |
| `test_cases[i].input` | string | 第 i 个测试用例的输入数据（标准输入）。 |
| `test_cases[i].output` | string | 第 i 个测试用例的预期输出（标准输出）。 |

#### 示例

以下是一条成功提取的训练数据示例：

```json
{
  "prompt": "你将得到一个编程竞赛题目。请逐步推理解决方案，然后用C或C++提供完整实现。请勿包含任何调试信息或额外输出。将最终解决方案放在单个代码块中：\n```cpp\n<your code here>\n```\n\n题目内容:\n\n# P1002 [NOIP 2002 普及组] 过河卒\n\n## 题目描述\n\n棋盘上 A 点有一个过河卒，需要走到目标 B 点。卒行走的规则：可以向下、或者向右。...",
  "test_cases": [
    {
      "input": "6 6 3 3",
      "output": "6"
    }
  ]
}
```

#### 处理流程

`convert_dataset.py` 的转换过程：

1. **加载原始 DPO 数据**：从 `../local_luogu_dpo/train` split 读取所有题目
2. **提取题目描述**：获取 `conversations[0]["value"]` 字段，即题目的 Markdown 格式文本
3. **添加指令前缀**：在题面前加入 C++实现相关的指令
4. **正则匹配测试用例**：
   - 匹配形如 `**输入：** ` 和 `**输出：** ` 的结构化代码块
   - 检查匹配结果是否包含中文字符
5. **过滤纯英文/数字样例**：仅保留不含中文字符的测试用例
   - 这样做的目的是保持测试用例的一致性，避免需要特殊处理的非ASCII字符
6. **保存成功结果**：写入 `luogu_rl_data.jsonl`

#### 数据统计

- 总行数：成功提取的题目数量
- 可直接用于 `GRPOTrainer` 的 `train_dataset`
- 使用 Hugging Face `datasets` 库加载：
  ```python
  from datasets import load_dataset
  rl_dataset = load_dataset("json", data_files="./local_luogu_rl/luogu_rl_data.jsonl", split="train")
  ```

---

### 2. failed_extraction.jsonl — 提取失败的题目

#### 文件格式

每一行是一个 JSON 对象，记录提取失败的题目及失败原因。

#### 字段说明

| 字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| `prompt` | string | 原始题目描述（未经处理）。 |
| `reason` | string | 提取失败的原因，通常为以下几种：<br>- `"No valid samples found or contained Chinese"`：未找到有效的测试用例，或所有样例都包含中文字符<br>- 其他异常或正则匹配失败的详细信息 |

#### 示例

以下是一条提取失败的题目示例：

```json
{
  "prompt": "你将得到一个编程竞赛题目。...\n\n# P1008 [NOIP 1998 普及组] 三连击\n\n## 题意描述\n将 1, 2, ..., 9 共 9 个数分成 3 组，分别组成 3 个三位数，...\n\n## 输入格式\n无\n\n## 输出格式\n若干行，每行 3 个数字。...",
  "reason": "No valid samples found or contained Chinese"
}
```

#### 失败原因分析

- **"No valid samples found or contained Chinese"**
  - 最常见的失败原因
  - 说明题目的样例要么根本不存在，要么样例中包含了中文字符（如中文输入输出说明）
  - 建议：人工检查这些题目，判断是否需要手动添加测试用例或者放弃使用

#### 故障排查

如果发现某些题目不应该失败，可以：
1. 在 `failed_extraction.jsonl` 中找到该题目
2. 查看 `reason` 字段了解失败原因
3. 修改 `convert_dataset.py` 中的正则表达式逻辑（通常在提取测试用例部分）
4. 重新运行 `python convert_dataset.py` 转换数据

---

# English Version

## Luogu RL Training Dataset

This directory contains Reinforcement Learning training data converted from the Luogu DPO dataset using `convert_dataset.py`. It contains two important files:

1. **`luogu_rl_data.jsonl`**: Successfully extracted problems and test cases
2. **`failed_extraction.jsonl`**: Failed extraction problems and failure reasons

### 1. luogu_rl_data.jsonl — Successfully Extracted Training Data

#### File Format

Each line is a JSON object corresponding to one competitive programming problem with its test cases.

#### Field Explanation

| Field | Type | Description |
| :--- | :--- | :--- |
| `prompt` | string | Complete problem description, including problem statement, input/output format, examples, etc. `convert_dataset.py` prepends an instruction prefix describing C++ implementation requirements. |
| `test_cases` | array | Array of test cases, each containing `input` and `output` objects. |
| `test_cases[i].input` | string | Input data for the i-th test case (stdin). |
| `test_cases[i].output` | string | Expected output for the i-th test case (stdout). |

#### Example

Here is an example of a successfully extracted training data entry:

```json
{
  "prompt": "You will receive a competitive programming problem. Please reason through the solution step by step, then provide a complete implementation in C or C++. Do not include any debugging information or extra output. Put the final solution in a single code block:\n```cpp\n<your code here>\n```\n\nProblem:\n\n# P1002 [NOIP 2002 Beginner] Crossing River with a Soldier\n\n## Problem Description\n\nThere is a soldier at point A on a chessboard who needs to reach point B. The soldier can move down or right...",
  "test_cases": [
    {
      "input": "6 6 3 3",
      "output": "6"
    }
  ]
}
```

#### Conversion Process

The `convert_dataset.py` conversion pipeline:

1. **Load Original DPO Data**: Read all problems from `../local_luogu_dpo/train` split
2. **Extract Problem Description**: Get the `conversations[0]["value"]` field (Markdown-formatted problem text)
3. **Add Instruction Prefix**: Prepend C++ implementation instructions to the problem statement
4. **Extract Test Cases via Regex**:
   - Match structured code blocks following patterns like `**Input:** ` and `**Output:** `
   - Check if matched results contain Chinese characters
5. **Filter by ASCII-only Samples**: Keep only test cases without Chinese characters
   - This ensures consistency and avoids special handling of non-ASCII characters
6. **Save Success Results**: Write to `luogu_rl_data.jsonl`

#### Data Statistics

- Total Lines: Number of successfully extracted problems
- Direct usage with `GRPOTrainer`'s `train_dataset`
- Load using Hugging Face `datasets`:
  ```python
  from datasets import load_dataset
  rl_dataset = load_dataset("json", data_files="./local_luogu_rl/luogu_rl_data.jsonl", split="train")
  ```

---

### 2. failed_extraction.jsonl — Failed Extraction Problems

#### File Format

Each line is a JSON object recording a failed extraction problem and its failure reason.

#### Field Explanation

| Field | Type | Description |
| :--- | :--- | :--- |
| `prompt` | string | Original problem description (unprocessed). |
| `reason` | string | Reason for extraction failure, typically:<br>- `"No valid samples found or contained Chinese"`：No valid test cases found, or all samples contain Chinese characters<br>- Other exception or regex matching failure details |

#### Example

Here is an example of a failed extraction entry:

```json
{
  "prompt": "You will receive a competitive programming problem. ...\n\n# P1008 [NOIP 1998 Beginner] Triple Hit\n\n## Problem Description\nDivide 1, 2, ..., 9 into 3 groups...\n\n## Input Format\nNone\n\n## Output Format\nMultiple lines, 3 numbers per line...",
  "reason": "No valid samples found or contained Chinese"
}
```

#### Failure Reason Analysis

- **"No valid samples found or contained Chinese"**
  - Most common failure reason
  - Indicates either no test cases exist for the problem, or all examples contain Chinese characters (e.g., Chinese input/output descriptions)
  - Recommendation: Manually inspect these problems and decide whether to add test cases manually or skip them

#### Troubleshooting

If certain problems should not have failed:
1. Find the problem in `failed_extraction.jsonl`
2. Check the `reason` field to understand the failure
3. Modify the regex expression logic in `convert_dataset.py` (typically in the test case extraction section)
4. Re-run `python convert_dataset.py` to convert the data
