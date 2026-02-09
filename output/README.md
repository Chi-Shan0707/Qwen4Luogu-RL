# tiny_lora_v.pt — TinyLoRA 参数文件 / Artifact

简明说明（中文）

- 文件位置：`./output/luoguqwencoder-lora/tiny_lora_v.pt`
- 内容（dict）：
  - `global_v`：训练好的共享向量（torch.Tensor），例如：`shape=torch.Size([32])`。
  - `u_value`：`u` 的数值（如 32）。
  - `rank`：TinyLoRA 的 rank（如 2）。
  - `seed`：用于生成固定 P 矩阵的随机种子（如 42）。
  - `model_id`：基座模型 ID（例如 `qwen/Qwen2.5-Coder-3B-Instruct`）。
  - `total_replaced_layers`：被替换的层数（便于记录）。

快速恢复步骤（中文简洁示例）：

```python
import torch
from transformers import AutoModelForCausalLM
# from transformers import BitsAndBytesConfig  # 如训练脚本所示，使用相同的 4bit 加载配置

sd = torch.load("./output/luoguqwencoder-lora/tiny_lora_v.pt", map_location="cpu")
# metadata
u = sd["u_value"]
seed = sd["seed"]
v = sd["global_v"]  # torch.Tensor
model_id = sd.get("model_id", "qwen/Qwen2.5-Coder-3B-Instruct")

# 1) 加载基座模型（须与训练时的量化/device_map 配置一致）
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

# 2) 创建 global params 容器并加载 v（使用 train_rl 中的 TinyLoRAGlobalParams）
#    确保使用相同 u、相同随机种子（sd['seed']）
from train_rl import TinyLoRAGlobalParams, apply_tiny_lora

device = model.model.layers[0].self_attn.q_proj.weight.device
global_params = TinyLoRAGlobalParams(u_dim=u, device=device, dtype=torch.bfloat16)
with torch.no_grad():
    global_params.global_v.copy_(v.to(global_params.global_v.dtype).to(device))

# 3) 固定随机种子并注入 TinyLoRA
torch.manual_seed(seed)
apply_tiny_lora(model, global_params)

# 4) 验证：
print("Loaded global_v shape:", global_params.global_v.shape)
```

---

Brief (English)

- Path: `./output/luoguqwencoder-lora/tiny_lora_v.pt`
- What it contains:
  - `global_v` (torch.Tensor, e.g. shape=[32])
  - `u_value` (int, e.g. 32)
  - `rank` (int, e.g. 2)
  - `seed` (int, e.g. 42)
  - `model_id` (str)
  - `total_replaced_layers` (int)

Quick restore snippet (concise):

```python
import torch
from transformers import AutoModelForCausalLM

sd = torch.load("./output/luoguqwencoder-lora/tiny_lora_v.pt", map_location="cpu")
model_id = sd.get("model_id")
u = sd["u_value"]
seed = sd["seed"]
v = sd["global_v"]

# load base model (use same 4bit / device_map settings as training)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

# use helpers from train_rl to create global params and inject TinyLoRA
from train_rl import TinyLoRAGlobalParams, apply_tiny_lora

device = model.model.layers[0].self_attn.q_proj.weight.device
global_params = TinyLoRAGlobalParams(u_dim=u, device=device, dtype=torch.bfloat16)
with torch.no_grad():
    global_params.global_v.copy_(v.to(global_params.global_v.dtype).to(device))

torch.manual_seed(seed)
apply_tiny_lora(model, global_params)

print("global_v shape:", global_params.global_v.shape)
```

Notes

- 恢复时务必：使用与训练一致的 `u_value`、`rank` 与随机 `seed`，并使用相同的基座模型与量化加载配置（4bit/NF4 等），以保证 `P` 矩阵与 SVD 分解一致，实现可复现的增量重建。