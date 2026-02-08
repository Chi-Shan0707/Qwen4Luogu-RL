from datasets import load_dataset

# 下载数据集到本地
dataset = load_dataset("Misaka114514/luogu_dpo")
dataset.save_to_disk("./local_luogu_dpo")
