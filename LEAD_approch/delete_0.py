import json

with open("", "r", encoding="utf-8") as f:
    data = json.load(f)

filtered_data = [obj for obj in data if obj.get("score", 0) > 0]

sorted_data = sorted(filtered_data, key=lambda x: x["score"], reverse=True)

with open("", "w", encoding="utf-8") as f:
    json.dump(sorted_data, f, ensure_ascii=False, indent=4)

print(f"新 JSON 文件的对象数量: {len(sorted_data)}")
