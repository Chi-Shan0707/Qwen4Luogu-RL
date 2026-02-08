import torch
import torch.nn as nn
import os
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from trl import GRPOTrainer, GRPOConfig
from modelscope.hub.snapshot_download import snapshot_download
import optimum
import bitsandbytes as bnb
from peft import prepare_model_for_kbit_training


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

# 直接在 GPU 上创建 global_v，并设为 bfloat16
global_v = nn.Parameter(torch.zeros(16, device=device, dtype=torch.bfloat16))
global_v.requires_grad = True


class TinyLoRALinear(nn.Module):
    def __init__(self, original_layer, rank = 2, u = 16, shared_v =None):
    # R= v_1 P_1 + v_2 P_2 + ... + v_u P_u
    # v都是scalar
    # P都是rank x rank的矩阵

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

        W = original_layer.weight.data.float()
        if hasattr(original_layer.weight, "quant_state"):
            # 4-bit 情况
            W_real = bnb.functional.dequantize_4bit(
                original_layer.weight.data, 
                original_layer.weight.quant_state
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
        
        # 唯一的可训练参数 v (如果传入 shared_v 则实现参数共享)

        if shared_v is not None:
            # 严查设备是否一致
            if shared_v.device != original_device:
                raise RuntimeError(
                    f"设备不匹配！shared_v 在 {shared_v.device}, "
                    f"但当前层在 {original_device}。\n"
                    "在单卡训练中，请确保 global_v 和模型都在同一张卡上。"
                )
            
            # 直接引用！不要 clone，不要 nn.Parameter
            self.v = shared_v 
        else:
            self.v = nn.Parameter(torch.zeros(u, device=original_device, dtype=target_dtype))

    def forward(self, x):
        # 计算 TinyLoRA 的增量矩阵 R
        R = torch.einsum('u, urr -> rr', self.v, self.P)
        # 重组增量权重
        delta_W = self.U @ self.S @ R @ self.Vh
        # 前向传播：x * (W + delta_W)^T
        return self.base_layer(x) + x @ delta_W.t()


def apply_tiny_lora(model, shared_v):
    """
    遍历模型，将所有目标 Linear 层替换为 TinyLoRALinear，
    并强制使用同一个 shared_v，实现论文中的 Tiling (全参数共享)。
    """
    # Qwen/Llama 的目标模块名称通常包含这些
    target_suffixes = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    # 计数器
    replaced_count = 0
    
    # 递归函数：遍历子模块
    for name, child in model.named_children():
        # 如果是目标 Linear 层
        if isinstance(child, nn.Linear) and any(name.endswith(s) for s in target_suffixes):
            # 1. 创建 TinyLoRA 层，传入 global_v
            # 注意：original_layer=child，shared_v=shared_v
            new_layer = TinyLoRALinear(child, rank=2, u=16, shared_v=shared_v)
            
            # 2. 替换掉原模块 (Monkey Patch)
            setattr(model, name, new_layer)
            replaced_count += 1
            print(f"已替换: {name} -> TinyLoRA (Shared)")
            
        else:
            # 继续递归遍历子模块 (例如 model.layers.0.self_attn...)
            apply_tiny_lora(child, shared_v)
            
    return replaced_count

# ========== 执行替换 ==========
print("正在应用 TinyLoRA Tiling (参数共享)...")
# global_v 已经在你之前的代码中定义了
total_replaced = apply_tiny_lora(model, global_v)
print(f"替换完成！共替换了 {total_replaced} 个模块。")

# ========== 关键步骤：冻结除 v 以外的所有参数 ==========
print("正在冻结模型参数...")

for name, param in model.named_parameters():
    # 只有 global_v 需要梯度，其他全部冻结
    # 注意：因为我们是把 shared_v 传进去的，id(param) == id(global_v)
    if param is global_v:
        param.requires_grad = True
    else:
        param.requires_grad = False

import re
import subprocess
import tempfile

import subprocess
import tempfile
import re
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
# 你可能需要手动保存 global_v
torch.save(global_v, f"{OUTPUT_DIR}/tiny_lora_v.pt")
print("训练完成，参数已保存！")