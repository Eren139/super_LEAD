import json
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

model = AutoModel.from_pretrained(
    "",
    device_map="cuda",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("", trust_remote_code=True)


def compute_score(model, tokenizer, instruction, output):
    instruction = str(instruction)
    output = str(output)

    chat = [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": output}
    ]
    return model.get_score(tokenizer, chat)



input_file = ''
output_file = ''

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

for obj in tqdm(data, desc="Processing dialogues"):
    instruction = obj.get("instruction", "")
    output = obj.get("output", "")

    score = compute_score(model, tokenizer, instruction, output)

    obj["score"] = score

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f"Scores computed and saved to {output_file}")
