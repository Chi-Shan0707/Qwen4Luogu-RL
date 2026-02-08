import torch
import torch.nn as nn
import os
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import prepare_model_for_kbit_training
from trl import GRPOTrainer, GRPOConfig
from modelscope.hub.snapshot_download import snapshot_download
import bitsandbytes as bnb


# ========== 模型配置 ==========
MS_MODEL_ID = "qwen/Qwen2.5-Coder-3B-Instruct"
LOCAL_MODEL_DIR = "./models/Qwen2.5-Coder-3B-Instruct"
OUTPUT_DIR = "./output/luoguqwencoder-lora"

#  Qwen2.5-Coder-3B-Instruct
# ========== 下载模型 ==========
if not os.path.exists(LOCAL_MODEL_DIR):
    print(f"从ModelScope下载模型 {MS_MODEL_ID} 到 {LOCAL_MODEL_DIR}...")
    snapshot_download(
        repo_id=MS_MODEL_ID,
        local_dir=LOCAL_MODEL_DIR,
    )
    print("模型下载完成！")
else:
    print(f"本地已存在模型，直接加载：{LOCAL_MODEL_DIR}")

# ========== 加载 tokenizer ==========
tokenizer = AutoTokenizer.from_pretrained(
    LOCAL_MODEL_DIR,
    trust_remote_code=True,
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


# ========== 加载模型（4bit 量化）==========
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL_DIR,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    # torch_dtype=torch.bfloat16,
    dtype=torch.bfloat16,
)
model.config.use_cache = False

# 准备模型进行 k-bit 训练
model = prepare_model_for_kbit_training(model)



# ========== 定义 TinyLoRA 层 ==========

# 获取模型第一层的设备 (通常是 cuda:0)
device = model.model.layers[0].self_attn.q_proj.weight.device
print(f"模型主设备: {device}")

# 【修复错误1】创建一个 wrapper module 来正确注册 global_v
class TinyLoRAGlobalParams(nn.Module):
    """专门用于注册全局共享向量的容器"""
    def __init__(self, u_dim=16, device='cpu', dtype=torch.bfloat16):
        super().__init__()
        # 这样注册才会被 model.named_parameters() 识别
        self.global_v = nn.Parameter(torch.zeros(u_dim, device=device, dtype=dtype))
    
    def forward(self):
        # 容器模块不需要实际的前向逻辑
        return self.global_v

# 创建全局参数容器
global_params = TinyLoRAGlobalParams(u_dim=16, device=device, dtype=torch.bfloat16)


class TinyLoRALinear(nn.Module):
    def __init__(self, original_layer, rank = 2, u = 16, global_params_ref=None):
    # R= v_1 P_1 + v_2 P_2 + ... + v_u P_u
    # v都是scalar
    # P都是rank x rank的矩阵
    # global_params_ref: 指向包含 global_v 的容器模块

        super().__init__()
        # 必先继承父类的初始化函数，才能使用 nn.Module 的功能（例如注册参数和缓冲区）。
        
        #  super().__init__() 是什么？
        # 这是 Python 面向对象编程（OOP）的标准写法。
        # 含义：调用父类（Parent Class）的初始化函数。
        # 在这里的作用：你的类 TinyLoRALinear 继承自 nn.Module（PyTorch 的神经网络基类）。执行 super().__init__() 是为了让 PyTorch 的机制生效，比如：
        # 注册你定义的 self.v 为可训练参数。
        # 注册 self.U, self.S 等为 Buffer（不训练的参数）。


        print(f"original_layer.device: {original_layer.weight.device}, dtype: {original_layer.weight.dtype}")

        original_device = original_layer.weight.device # 记录原device


        self.base_layer = original_layer
        
        # 【关键修复】保存对 global_params 容器的引用，而不是直接持有参数
        # 这样 v 只被 model.tiny_lora_params 持有一次，避免重复注册问题
        if global_params_ref is None:
            raise RuntimeError("必须传入 global_params_ref！")
        self.global_params_ref = global_params_ref

        W = original_layer.weight.data.float()
        if hasattr(original_layer.weight, "quant_state"):
            # 【修复错误2】4-bit 情况：显式传入 quant_type
            W_real = bnb.functional.dequantize_4bit(
                original_layer.weight.data, 
                original_layer.weight.quant_state,
                quant_type="nf4"  # 与 BitsAndBytesConfig 中的配置一致
            )
        else:
            # 非量化情况
            W_real = original_layer.weight.data


        W_real_on_cpu = W_real.float().cpu()

        U, S ,Vh = torch.linalg.svd( W_real_on_cpu ,full_matrices=False)

        # SVD 分解 W 矩阵
        # W = U S Vh 
        # Vh是 V的Hermitian transposed，共轭转置
        # 冻结 U, S, V (LoRA-XS 的骨架)

        

        # 将结果转回 BFloat16 并移回 GPU
        # 截断并注册(即固定住)
        # 建议转回 bf16 省显存
        # 
        # 这一步也是为了让 TinyLoRA 的参数和主模型精度保持一致
        
        target_dtype = torch.bfloat16

        self.register_buffer('U', U[:, :rank].to(original_device).to(target_dtype)) 
        self.register_buffer('S', torch.diag(S[:rank]).to(original_device).to(target_dtype))
        self.register_buffer('Vh', Vh[:rank, :].to(original_device).to(target_dtype))
        
        # 固定随机矩阵 P  (For TinyLoRA)
        self.register_buffer('P', torch.randn(u, rank, rank, device=original_device, dtype=target_dtype))

    def forward(self, x):
        # 【关键修复】动态从容器中获取 global_v，而不是作为自己的属性
        # 这样确保 v 只被 model.tiny_lora_params 注册一次
        v = self.global_params_ref.global_v
        
        # 计算 TinyLoRA 的增量矩阵 R = sum_i(v_i * P_i)
        # 注意：不能用 'u, urr -> rr'，因为 einsum 输出中同一下标不能重复
        # 必须用不同字母区分两个 rank 维度
        R = torch.einsum('u, uij -> ij', v, self.P)
        # 重组增量权重
        delta_W = self.U @ self.S @ R @ self.Vh
        # 前向传播：x * (W + delta_W)^T
        return self.base_layer(x) + x @ delta_W.t()


def apply_tiny_lora(model, global_params_ref):
    """
    遍历模型，将所有目标 Linear 层替换为 TinyLoRALinear，
    并传入对 global_params 容器的引用，实现论文中的 Tiling (全参数共享)。
    """
    # Qwen/Llama 的目标模块名称通常包含这些
    target_suffixes = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    # 计数器
    replaced_count = 0
    
    # 递归函数：遍历子模块
    for name, child in model.named_children():
        # 如果是目标 Linear 层
        if isinstance(child, (nn.Linear, bnb.nn.Linear4bit)) and any(name.endswith(s) for s in target_suffixes):
            # 1. 创建 TinyLoRA 层，传入 global_params 容器的引用
            new_layer = TinyLoRALinear(child, rank=2, u=16, global_params_ref=global_params_ref)
            
            # 2. 替换掉原模块 (Monkey Patch)
            setattr(model, name, new_layer)
            replaced_count += 1
            print(f"已替换: {name} -> TinyLoRA (Shared)")
            
        else:
            # 继续递归遍历子模块 (例如 model.layers.0.self_attn...)
            replaced_count += apply_tiny_lora(child, global_params_ref)
            
    return replaced_count

# ========== 执行替换 ==========
print("正在应用 TinyLoRA Tiling (参数共享)...")

# 【关键修复】先将 global_params 注册为模型的子模块
# 这样在层替换时，TinyLoRALinear 就能通过引用访问到已注册的 global_v
model.tiny_lora_params = global_params
print(f"✅ 已将 global_params 注册到模型")

# 然后再进行层替换，传入 global_params 容器本身
total_replaced = apply_tiny_lora(model, global_params)
print(f"替换完成！共替换了 {total_replaced} 个模块。")

# ========== 关键步骤：冻结除 v 以外的所有参数 ==========
print("正在冻结模型参数...")

# 【更优雅的方案】直接通过对象引用操作，不依赖字符串匹配
# 1. 第一步：全局冻结所有参数
model.requires_grad_(False)

# 2. 第二步：精准解冻 global_v
# 直接通过对象引用操作，绝对稳健
global_params.global_v.requires_grad = True
print(f"✅ 可训练参数: global_v, shape={global_params.global_v.shape}")

# 验证可训练参数
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in model.parameters())
print(f"\n总参数量: {all_params:,}")
print(f"可训练参数量: {trainable_params}")
if trainable_params != 16:
    raise RuntimeError(f"警告：可训练参数数量为 {trainable_params}，预期为 16！")

import re
import subprocess
import tempfile
import os

def compile_and_run(code, test_cases):
    """
    编译并运行代码，返回通过率 (0.0 ~ 1.0)
    """
    code = re.sub(r'freopen\s*\(.*?\);', '', code, flags=re.IGNORECASE)
    # 1. 创建临时目录 (用完即删，防止垃圾文件堆积)
    with tempfile.TemporaryDirectory() as temp_dir:
        src_file = os.path.join(temp_dir, "solution.cpp")
        exe_file = os.path.join(temp_dir, "solution")
        
        # 2. 写入 C++ 代码
        with open(src_file, 'w', encoding='utf-8') as f:
            f.write(code)
            
        # 3. 编译 (加上 -O2 优化，且不链接多余库)
        # timeout=5 防止编译器卡死
        try:
            compile_result = subprocess.run(
                ['g++', src_file, '-o', exe_file, '-O2'],
                capture_output=True, text=True, timeout=5
            )
            if compile_result.returncode != 0:
                return 0.0 # 编译失败
        except subprocess.TimeoutExpired:
            return 0.0 # 编译超时

        # 4. 运行测试用例
        passed_count = 0
        total_cases = len(test_cases)
        
        if total_cases == 0:
            return 0.0

        for case in test_cases:
            input_data = case['input']
            expected_output = case['output'].strip()
            
            try:
                # 关键：使用 input=input_data 模拟 freopen/cin
                # timeout=2 秒，防止死循环 (非常重要！！！)
                run_result = subprocess.run(
                    [exe_file],
                    input=input_data,
                    capture_output=True,
                    text=True,
                    timeout=2 
                )
                
                # 获取模型输出并去首尾空格
                actual_output = run_result.stdout.strip()
                
                # 简单比对 (也可以根据需要改成浮点数比对等)
                if actual_output == expected_output:
                    passed_count += 1
                    
            except subprocess.TimeoutExpired:
                pass # 运行超时算错
            except Exception:
                pass # 运行时错误(RE)算错

        return passed_count / total_cases

def code_reward_func(completions, test_cases, **kwargs):
    """
    GRPO 要求的 Reward Function 格式
    completions: list[str], 模型生成的多个回复
    test_cases: list[list[dict]], 对应的测试用例（注意 GRPO 传进来的是 batch）
    """
    rewards = []
    
    # 遍历每一条生成的回复
    for completion, cases in zip(completions, test_cases):
        # 1. 提取代码块
        # 匹配 ```cpp ... ``` 或 ``` ... ```
        match = re.search(r"```(?:cpp|c\+\+)?\n(.*?)```", completion, re.DOTALL)
        
        if not match:
            # 如果没提取到，尝试找一下是否有裸代码（包含 #include）
            if "#include" in completion:
                code = completion
            else:
                rewards.append(0.0) # 格式完全不对
                continue
        else:
            code = match.group(1)

        # 2. 评测
        score = compile_and_run(code, cases)
        rewards.append(score)
        
    return rewards
    

# 【修复错误4 - 最终方案】绕过 Trainer 对纯量化模型的检查
# Trainer 的检查逻辑 (transformers/trainer.py):
#   _is_quantized_and_base_model = model.is_quantized AND NOT model._hf_peft_config_loaded
#   if _is_quantized_and_base_model and not isinstance(model, PeftModel): raise ValueError
#
# 我们的 TinyLoRA 是合法的 adapter（只训练 16 个参数），但不是标准 PeftModel。
# 设置 _hf_peft_config_loaded = True 让第一道检查直接为 False，不会走到 isinstance 判断。
# 这不影响实际计算——权重已经在内存中量化，TinyLoRA 层正确处理了反量化。
model._hf_peft_config_loaded = True

print("✅ 已设置 _hf_peft_config_loaded=True，绕过 Trainer 的量化模型检查")



# ========== 加载数据集 ==========


# 当你使用 load_dataset("json", data_files="....jsonl") 时，
# Hugging Face 会默认把你提供的这个文件归类为 train 分区（这是它的默认行为）。

# 注意：data_files 指向你 convert_dataset.py 生成的具体文件路径
# split="train" 很重要！因为 load_dataset 默认返回 DatasetDict，
# 而 Trainer 需要的是 Dataset 对象，指定 split="train" 直接拿到数据。
rl_dataset = load_dataset(
    "json", 
    data_files="./local_luogu_rl/luogu_rl_data.jsonl", 
    # 确认这里的路径和你 convert_dataset.py 里的 OUTPUT_FILE 一致
    split="train"
)

# 2. (可选) 打印一条数据验证一下
print(f"数据加载成功！样本数量: {len(rl_dataset)}")
print(f"样例数据: {rl_dataset[0]}")
# ========== 配置并启动 GRPO 训练 ==========
# 配置 GRPO
training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=1, # 单卡显存不够就设为 1
    gradient_accumulation_steps=8, # 累积梯度来模拟大 Batch
    learning_rate=1e-5,            # RL 学习率通常要小
    num_generations=4,             # Group Size (G): 每次采样 4 个答案
    max_completion_length=512,     # 生成的最大长度
    logging_steps=1,
    bf16=True,                     # 开启 BF16 加速
)

# 初始化训练器
trainer = GRPOTrainer(
    model=model,
    reward_funcs=code_reward_func, # 你的判题函数
    args=training_args,
    train_dataset=rl_dataset,   # 处理好的数据
    processing_class=tokenizer,    # Tokenizer
)

# 开始训练！
print("🚀 开始 TinyLoRA-RL 训练...")
trainer.train()

# 保存 LoRA (只保存那个 v 向量)
# 注意：peft 的 save_pretrained 可能不认你的自定义层
# 手动保存 global_v
torch.save(global_params.global_v, f"{OUTPUT_DIR}/tiny_lora_v.pt")
print("训练完成，参数已保存！")