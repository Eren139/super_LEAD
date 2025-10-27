import json
import torch
from tqdm import tqdm  # 导入进度条库
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载大模型和tokenizer
model_name_or_path = ""  # 替换为你的模型路径
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16,
                                             trust_remote_code=True).cuda()
model.eval()

# 读取处理后的JSON文件
with open('', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 使用进度条处理每个条目
for key in tqdm(data.keys(), desc="处理进度"):
    # 获取origin_prompt的值
    origin_prompt = data[key]['origin_prompt']

    # 调用模型生成回复
    response, history = model.chat(tokenizer, origin_prompt, history=[])

    # 将模型输出设置为prediction的值
    data[key]['prediction'] = response

# 保存修改后的数据到新文件
with open('', 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)