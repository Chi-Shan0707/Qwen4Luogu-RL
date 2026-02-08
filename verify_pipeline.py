import torch
import os
import re
import json
import subprocess
import tempfile
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ==================== 配置区域 ====================
MS_MODEL_ID = "qwen/Qwen2.5-Coder-3B-Instruct"
LOCAL_MODEL_DIR = "./models/Qwen2.5-Coder-3B-Instruct"

# 【关键】直接使用你提供的 JSON 数据结构进行测试
# 这里使用了 P1029 [NOIP 2001 普及组] 作为测试题
TEST_DATA_JSON = {
    "prompt": "你将得到一个编程竞赛题目。请逐步推理解决方案，然后用C或C++提供完整的实现。请勿包含任何调试信息或额外输出。将最终解决方案放在单个代码块中：\n```cpp\n<your code here>\n```\n\n\n题目内容:\n\n# P1029 [NOIP 2001 普及组] 最大公约数和最小公倍数问题\n\n\n## 题目描述\n\n输入两个正整数 $x_0, y_0$，求出满足下列条件的 $P, Q$ 的个数：\n1. $P,Q$ 是正整数。\n2. 要求 $P, Q$ 以 $x_0$ 为最大公约数，以 $y_0$ 为最小公倍数。\n试求：满足条件的所有可能的 $P, Q$ 的个数。\n\n## 输入格式\n\n一行两个正整数 $x_0, y_0$。\n\n## 输出格式\n\n一行一个数，表示求出满足条件的 $P, Q$ 的个数。\n\n## 说明/提示\n\n$P,Q$ 有 $4$ 种：\n1. $3, 60$。\n2. $15, 12$。\n3. $12, 15$。\n4. $60, 3$。\n对于 $100\\%$ 的数据，$2 \\le x_0, y_0 \\le {10}^5$。\n**【题目来源】**\nNOIP 2001 普及组第二题\n\n## 样例\n\n### 样例 1\n\n**输入：**\n```\n3 60\n```\n\n**输出：**\n```\n4\n```\n\n",
    "test_cases": [{"input": "3 60", "output": "4"}]
}
# =================================================

def print_step(title):
    print(f"\n{'='*10} {title} {'='*10}")

def extract_code(completion):
    """从回复中提取代码，逻辑同 train_rl.py"""
    # 优先匹配代码块
    match = re.search(r"```(?:cpp|c\+\+)?\n(.*?)```", completion, re.DOTALL)
    if match:
        return match.group(1), "Code Block"
    # 兜底匹配 #include
    elif "#include" in completion:
        return completion, "Raw Text"
    else:
        return None, "Failed"

def compile_and_run(code, test_cases):
    """编译并运行，逻辑同 train_rl.py"""
    # 移除 freopen，防止卡死
    code = re.sub(r'freopen\s*\(.*?\);', '', code, flags=re.IGNORECASE)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        src_file = os.path.join(temp_dir, "solution.cpp")
        exe_file = os.path.join(temp_dir, "solution")
        
        # 写入
        with open(src_file, 'w', encoding='utf-8') as f:
            f.write(code)
            
        print(f"   -> 正在编译临时文件...")
        # 编译
        try:
            res = subprocess.run(
                ['g++', src_file, '-o', exe_file, '-O2'],
                capture_output=True, text=True, timeout=5
            )
            if res.returncode != 0:
                return 0.0, f"编译失败:\n{res.stderr}"
        except Exception as e:
            return 0.0, f"编译异常: {e}"

        # 运行测试用例
        passed = 0
        total = len(test_cases)
        for i, case in enumerate(test_cases):
            input_data = case['input']
            expected_output = case['output'].strip()
            
            try:
                res = subprocess.run(
                    [exe_file],
                    input=input_data,
                    capture_output=True,
                    text=True,
                    timeout=2 # 2秒超时
                )
                actual_output = res.stdout.strip()
                
                if actual_output == expected_output:
                    print(f"   -> Case {i+1}: ✅ 通过 (输入: '{input_data.strip()}' | 预期: '{expected_output}' | 实际: '{actual_output}')")
                    passed += 1
                else:
                    print(f"   -> Case {i+1}: ❌ 失败 (输入: '{input_data.strip()}' | 预期: '{expected_output}' | 实际: '{actual_output}')")
            except subprocess.TimeoutExpired:
                print(f"   -> Case {i+1}: ⚠️ 运行超时 (Timeout)")
            except Exception as e:
                print(f"   -> Case {i+1}: ⚠️ 运行错误 {e}")
        
        return passed / total, "Success"

def main():
    print_step("STEP 1: 加载模型与Tokenizer")
    
    # 检查 g++
    try:
        subprocess.run(['g++', '--version'], capture_output=True)
        print("✅ 检测到 g++ 编译器")
    except:
        print("❌ 未检测到 g++，请先安装 (sudo apt install g++)")
        return

    # 加载 Tokenizer
    model_path = LOCAL_MODEL_DIR if os.path.exists(LOCAL_MODEL_DIR) else MS_MODEL_ID
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型 (4-bit)
    print(f"正在加载模型: {model_path} (4-bit)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    print("✅ 模型加载完成")

    # ------------------------------------------------------------------
    print_step("STEP 2: 验证 Chat Template (JSON -> Qwen Prompt)")
    
    # 模拟 train_rl.py 中的数据处理逻辑
    raw_prompt = TEST_DATA_JSON['prompt']
    messages = [
        {"role": "system", "content": "你是一个智能编程助手。推理部分内容控制在128token以内。代码要严格按照传统c++编写。"},
        {"role": "user", "content": raw_prompt}
    ]
    
    # 应用模版
    final_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    print("--- 最终输入给模型的 Prompt 开头部分 ---")
    print(final_prompt[:300] + "...\n")
    print("--- 最终输入给模型的 Prompt 结尾部分 ---")
    print("..." + final_prompt[-100:])
    
    # 检查关键标签
    if "<|im_start|>system" in final_prompt and "<|im_start|>assistant" in final_prompt:
        print("\n✅ 模版格式检查通过 (检测到 Qwen ChatML 标签)")
    else:
        print("\n❌ 警告：未检测到 ChatML 标签，请检查 tokenizer_config.json")

    # ------------------------------------------------------------------
    print_step("STEP 3: 执行模型生成")
    
    inputs = tokenizer([final_prompt], return_tensors="pt").to(model.device)
    
    print(f"Prompt token 长度: {inputs.input_ids.shape[1]}")
    print("正在生成 (Max 1024 tokens)...")
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,     
            temperature=0.6,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # 解码
    full_response = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
    
    # 只要生成部分
    if "<|im_start|>assistant" in full_response:
        response_only = full_response.split("<|im_start|>assistant")[-1]
    else:
        response_only = full_response
    
    print("\n--- 模型生成的代码部分 (前1000字符) ---")
    print(response_only[:1000] + "..." if len(response_only)>500 else response_only)

    # ------------------------------------------------------------------
    print_step("STEP 4: 验证代码提取与评测 (基于 JSON test_cases)")
    
    extracted_code, method = extract_code(response_only)
    test_cases = TEST_DATA_JSON['test_cases']
    
    if extracted_code:
        print(f"✅ 成功提取代码 (方式: {method})")
        print(f"正在使用 {len(test_cases)} 个测试用例进行评测...")
        
        # 实际运行评测
        score, msg = compile_and_run(extracted_code, test_cases)
        
        print(f"\n📊 最终得分 (Reward): {score}")
        
        if score == 1.0:
            print("🎉 结论：Pipeline 完美通过！模型成功解出了题目。")
        elif score > 0.0:
            print("⚠️ 结论：Pipeline 通畅，代码可运行，但部分用例未通过 (这是 RL 训练需要解决的问题)。")
        else:
            print(f"⚠️ 结论：代码编译失败或运行全错。详细信息: {msg}")
            print("注意：对于未微调的 3B 模型，第一次做对中等难度的数论题(P1029)是有挑战的。只要编译过程没报错，Pipeline 就是好的。")
    else:
        print("❌ 代码提取失败！模型可能没生成代码块。")

if __name__ == "__main__":
    main()