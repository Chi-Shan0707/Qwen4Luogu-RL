import torch
import os
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from modelscope.hub.snapshot_download import snapshot_download
import optimum
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

# ========== LoRA 配置 ==========
lora_config = LoraConfig(
    r=64, # 调参是一个矩阵， 这是矩阵的维度，原16
    lora_alpha=16, #适配器影响强度通常是r的两倍
    lora_dropout=0.05,# 随机丢弃适配器权重 原0.1
    bias="none", 
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    
  
    # modules_to_save=["embed_tokens", "lm_head"],  # 可选，少量额外参数，提升指令对齐
    # 这是一个巨大的陷阱。

    #     问题：embed_tokens 和 lm_head 是词表映射层。Qwen 的词表很大（约 152k）。当你把它们加入 modules_to_save，意味着你实际上在训练数亿个参数（152000 * hidden_size），这完全违背了 LoRA 的初衷。

    #     后果：

    #     显存爆炸/训练极慢：原本 LoRA 只需要训练几兆参数，现在变成了几百兆甚至更多。

    #     灾难性遗忘：在一个小的竞赛数据集（如洛谷）上全量更新词表头，会破坏模型原本通用的语言能力，导致模型变得“傻”或者只会复读训练集。

    #     过拟合：参数量过大，数据量不够。

    #     建议：直接删除这一行。对于代码微调，通常只需要训练 Attention 和 MLP 层的 LoRA 权重即可。
    
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ========== 加载数据集 ==========
# dataset = load_dataset("Misaka114514/luogu_dpo")
dataset = load_from_disk("./local_luogu_dataset")


# ========== SFTConfig：仅包含训练参数（TRL 0.27+ 规范）==========
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1, #一个gpu上同时处理1批数据
    gradient_accumulation_steps=16,# 每16步才更新一次参数
    # batch， 一次处理的样本
    # batchsize 16*1=16个
    num_train_epochs=2, # 完整遍历数据集2次
    learning_rate=2e-4, # 学习率，即参数更新的步长
    # 分析：1e-5 对于 LoRA 来说偏保守（全量微调才用这么小）。LoRA 通常可以容忍更大的学习率。
    # 建议：尝试 2e-4 或 3e-4。
    weight_decay=0.01,
    lr_scheduler_type="cosine", #: 学习率调度器类型。设置为 "cosine"，表示使用余弦退火调度器：学习率从初始值逐渐下降到 0，形成余弦曲线。这有助于平稳收敛，避免后期震荡。
    warmup_steps=100,
    fp16=False,
    bf16=True,
    logging_steps=50,# 每50步记录一次日志
    save_steps=50, # 每500步保存一次模型检查点
    save_total_limit=10, #保存的检查点总数限制。设置为 2，只保留最新的 2 个检查点，防止磁盘空间占用过多。
    report_to="none",
    gradient_checkpointing=True, # 是否启用梯度检查点。设置为 True，通过重计算中间激活节省显存（以时间换空间），适合大模型训练。
    remove_unused_columns=False,
    dataloader_pin_memory=False,
    neftune_noise_alpha=0.5,# NEFTune 噪声参数。若设置为 5.0，用于在训练中添加噪声以改善生成质量（减少生成重复）。值通常在 0-10 之间，0 表示禁用。
# 分析：NEFTune 通过在 Embedding 层加噪声来避免过拟合。但是 5.0 是一个非常激进的值（通常用 5.0 是为了增强对话的多样性）。对于编程和数学这种要求精确逻辑的任务，过大的噪声会干扰模型学习精确的语法和逻辑路径。

# 建议：去掉这一行，或者将其降低到 1.0 甚至 0。
)

# ========== SFTTrainer：只负责训练，不做格式化（TRL 0.27+ 规范）==========
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset["train"],
    processing_class=tokenizer,

)

# ========== 训练 ==========
model.gradient_checkpointing_enable()
trainer.train()

# ========== 保存 ==========
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"训练完成，LoRA权重已保存到：{OUTPUT_DIR}")
