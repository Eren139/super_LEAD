import json

def normalize_score(score, min_score, max_score):
    return (score - min_score) / (max_score - min_score)

with open('', 'r', encoding='utf-8') as f:
    data = json.load(f)

min_score = float('inf')
max_score = float('-inf')

for obj in data:
    score = obj.get('reward_score', 0)
    min_score = min(min_score, score)
    max_score = max(max_score, score)

print(f"最小的 score 值: {min_score}")
print(f"最大的 score 值: {max_score}")

for obj in data:
    score = obj.get('reward_score', 0)
    ifd_ppl = obj.get('ifd_ppl', 0)

    normalized_score = normalize_score(score, min_score, max_score)

    obj['reward_score'] = normalized_score

    obj['LED'] = normalized_score * ifd_ppl

sorted_data = sorted(data, key=lambda x: x['LETS'], reverse=True)

with open('', 'w', encoding='utf-8') as f:
    json.dump(sorted_data, f, indent=4, ensure_ascii=False)

print(f"文件中的对象数量：{len(sorted_data)}")
