import json

def remove_keys_from_json(input_file, output_file, keys_to_remove):
    with open(input_file, 'r') as f:
        data = json.load(f)

    for obj in data:
        for key in keys_to_remove:
            if key in obj:
                del obj[key]

    object_count = len(data)

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"Final file contains {object_count} objects.")

# 文件路径
input_file = ''
output_file = ''
keys_to_remove = [""]

# 调用函数
remove_keys_from_json(input_file, output_file, keys_to_remove)
