import re
import json
from datasets import load_from_disk
from tqdm import tqdm

# ========== 配置 ==========
# 你的本地数据集路径（就是包含 state.json 的那个文件夹）
DATASET_PATH = "./local_luogu_dpo" 
# 输出文件路径
OUTPUT_FILE = "./local_luogu_rl/luogu_rl_data.jsonl"
def contains_chinese(text):
    """检查文本是否包含中文字符"""
    return bool(re.search(r'[\u4e00-\u9fa5]', text))


def extract_samples(description):
    """
    鲁棒性增强版 + 数据清洗：
    1. 定位样例区域。
    2. 正则提取。
    3. 【新增】过滤掉包含中文的脏数据。
    """
    samples = []
    text = description.replace("\r\n", "\n")

    # === 步骤 1: 定位区域 (保持不变) ===
    header_pattern = r"(?:#+\s*(?:样例|Sample))"
    headers = list(re.finditer(header_pattern, text))
    search_text = text
    if headers:
        start_pos = headers[0].start()
        search_text = text[start_pos:]

    # === 步骤 2: 正则提取 (保持不变) ===
    input_pattern = r"(?:\*\*输入：\*\*|Input:)\s*```.*?\n([\s\S]*?)```"
    output_pattern = r"(?:\*\*输出：\*\*|Output:)\s*```.*?\n([\s\S]*?)```"
    inputs = re.findall(input_pattern, search_text)
    outputs = re.findall(output_pattern, search_text)
    
    # === 步骤 3: 配对与验证 (【修改点：增加中文检查】) ===
    if inputs and outputs and len(inputs) == len(outputs):
        for inp, outp in zip(inputs, outputs):
            clean_in = inp.strip()
            clean_out = outp.strip()
            
            # 只有非空 且 不含中文 才是有效样例
            if clean_in and clean_out:
                if contains_chinese(clean_in) or contains_chinese(clean_out):
                    continue # 跳过脏数据
                
                samples.append({
                    "input": clean_in,
                    "output": clean_out
                })
    
    # === 步骤 4: 兜底策略 (【修改点：增加中文检查】) ===
    if not samples:
        fallback_pattern = r"(?:输入样例|Sample Input).*?```.*?\n([\s\S]*?)```.*?(:?输出样例|Sample Output).*?```.*?\n([\s\S]*?)```"
        matches = re.findall(fallback_pattern, text, re.DOTALL)
        for inp, outp in matches:
            clean_in = inp.strip()
            clean_out = outp.strip()
            
            if clean_in and clean_out:
                if contains_chinese(clean_in) or contains_chinese(clean_out):
                    continue
                    
                samples.append({
                    "input": clean_in,
                    "output": clean_out
                })

    return samples if len(samples) > 0 else None

# ========== 主程序 ==========
# 新增：失败案例输出路径
FAILED_FILE = "./local_luogu_rl/failed_extraction.jsonl"

print(f"正在加载数据集: {DATASET_PATH}")
dataset = load_from_disk(DATASET_PATH)

valid_count = 0
total_count = 0
failed_items = [] # 用于存储提取失败的案例

print("开始提取...")
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    # 只要 train split
    for item in tqdm(dataset['train']): 
        total_count += 1
        
        # 1. 获取题目
        if 'conversations' in item:
            problem_text = item['conversations'][0]['value']
        else:
            continue
            
        # 2. 提取样例
        samples = extract_samples(problem_text)
        
        # 3. 如果成功，写入数据文件
        if samples:
            data_object = {
                "prompt": problem_text,
                "test_cases": samples,
            }
            f.write(json.dumps(data_object, ensure_ascii=False) + "\n")
            valid_count += 1
        else:
            # 【新增】如果失败，记录到失败列表
            failed_items.append({
                "prompt": problem_text,
                "reason": "No valid samples found or contained Chinese"
            })

# ========== 结束统计与输出失败案例 ==========
print(f"\n提取完成！")
print(f"总数据条数: {total_count}")
print(f"成功提取并保存: {valid_count}")
print(f"提取失败: {len(failed_items)}")
print(f"提取率: {valid_count/total_count*100:.2f}%")
print(f"成功数据已保存至: {OUTPUT_FILE}")

# 将失败案例写入单独的文件，方便人工检查
if failed_items:
    print(f"正在保存 {len(failed_items)} 条失败案例到: {FAILED_FILE}")
    with open(FAILED_FILE, "w", encoding="utf-8") as f_fail:
        for item in failed_items:
            f_fail.write(json.dumps(item, ensure_ascii=False) + "\n")
    print("你可以检查 failed_extraction.jsonl 看看是什么奇怪的题目格式导致了提取失败。")