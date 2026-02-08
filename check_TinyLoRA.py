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


# ========== 加载数据集 ==========
# dataset = load_dataset("Misaka114514/luogu_dpo")
dataset = load_from_disk("./local_luogu_dataset")
print("已完成数据集加载")

# ========== 定义 TinyLoRA 层 ==========

# 创建一个全局共享的向量 v (实现论文提到的 Tiling)
global_v = nn.Parameter(torch.zeros(16)) 
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
            # 处理多卡/多设备时的共享引用问题
            if shared_v.device != original_device:
                 # 如果设备不一致，创建一个不共享梯度的副本（这是一个妥协，严谨的共享需要 DDP 同步）
                 self.v = nn.Parameter(shared_v.data.to(original_device).clone())
            else:
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

# 验证一下
print("\n========== 可训练参数检查 ==========")
trainable_params = 0
all_params = 0
for _, param in model.named_parameters():
    all_params += param.numel()
    if param.requires_grad:
        trainable_params += param.numel()
        print(f"可训练参数: 形状 {param.shape}, 元素数 {param.numel()}")

print(f"\n总参数量: {all_params}")
print(f"可训练参数量: {trainable_params}")
print(f"参数压缩率: {trainable_params / all_params * 100:.8f}%")

if trainable_params == 16:
    print(">>> 成功！当前仅训练 16 个参数 (TinyLoRA Tiling 生效) <<<")
else:
    print(">>> 警告：可训练参数数量不对，请检查代码 <<<")


