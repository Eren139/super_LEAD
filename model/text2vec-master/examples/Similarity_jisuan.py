import json
from text2vec import Similarity
from tqdm import tqdm

# 加载预训练模型
sim_model = Similarity("/root/autodl-tmp/model/text2vec-base-multilingual")

# 读取JSON文件
with open('/root/autodl-tmp/data/ifd_sim_dir+/ifd_20per.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 遍历JSON对象，计算语义相似度
for item in tqdm(data, desc="Processing", unit="item"):
    sentence1 = item['instruction'] + " " + item['input']
    sentence2 = item['output']

    # 计算相似度分数
    scores = sim_model.get_scores([sentence1], [sentence2])
    similarity_score = float(scores[0][0])  # 将float32转换为Python的float

    # 添加相似度分数到JSON对象中
    item['Similarity'] = similarity_score

# 将结果写回到JSON文件
with open('/root/autodl-tmp/data/ifd_sim_dir+/ifd_20per_sim.json', 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

print("相似度计算完成，并已写入到新的JSON文件中。")
