import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

# 设置CUDA设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 本地模型配置
model_dir = ""  # 请将此路径替换为您的本地模型路径

# 加载分词器和模型
tokenizer = LlamaTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = LlamaForCausalLM.from_pretrained(model_dir, device_map="auto",
                                             torch_dtype=torch.float16,
                                             trust_remote_code=True)

# 定义生成响应的函数
def generate_response(prompt: str) -> str:
    inputs = tokenizer(f'</s>Human:{prompt} </s>Assistant: ', return_tensors='pt')
    inputs = inputs.to('cuda')
    pred = model.generate(**inputs, max_new_tokens=512, repetition_penalty=1.2)
    response = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
    return response.split("Assistant: ")[1]

# 读取CSV文件
csv_file_path = ""  # 请替换为您的CSV文件路径
df = pd.read_csv(csv_file_path)

# 添加第三列用于存储生成的响应
if 'response' not in df.columns:
    df['response'] = ''

# 遍历CSV中的每个问题并生成响应
for index, row in tqdm(df.iterrows(), total=len(df)):
    question = row[0]  # 假设问题在第一列
    response = generate_response(question)
    df.at[index, 'response'] = response  # 将响应写入第三列

    # 保存中间结果到CSV文件
    if index % 10 == 0:  # 每处理10行保存一次
        df.to_csv(csv_file_path, index=False)
        print(f"已保存到文件，当前处理行数: {index + 1}")

# 最后一次保存结果到CSV文件
df.to_csv(csv_file_path, index=False)
print("所有行处理完成，结果已保存到文件。")
