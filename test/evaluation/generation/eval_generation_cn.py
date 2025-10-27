from transformers import LlamaTokenizer, LlamaForCausalLM, set_seed
import torch
import argparse
import json
import os
from tqdm import tqdm

# 定义不同的提示模板
PROMPT_DICT_ALPACA = {
    "prompt_input": (
        "以下是描述任务的指令，并提供了进一步的上下文输入。请生成合适的回应完成请求。\n\n"
        "### 指令:\n{instruction}\n\n### 输入:\n{input}\n\n### 回应:"
    ),
    "prompt_no_input": (
        "以下是描述任务的指令。请生成合适的回应完成请求。\n\n"
        "### 指令:\n{instruction}\n\n### 回应:"
    ),
}
PROMPT_DICT_WIZARDLM = {
    "prompt_input": (
        "下面是一个任务的描述和输入。请给出一个恰当的回答。\n\n"
        "### 任务描述:\n{instruction}\n\n### 输入:\n{input}\n\n### 回答:"
    ),
    "prompt_no_input": (
        "下面是一个任务的描述。请给出一个恰当的回答。\n\n"
        "### 任务描述:\n{instruction}\n\n### 回答:"
    ),
}


# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="要使用的数据集名称。",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default='koala',
        help="使用的提示模板类型: alpaca, koala。",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=None,
        help="生成时使用的束搜索数量。",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="预训练模型的路径或模型标识符。",
        required=False,
    )
    parser.add_argument("--seed", type=int, default=0, help="随机种子，以确保实验的可复现性。")
    parser.add_argument("--max_length", type=int, default=1024, help="生成的最大长度。")
    args = parser.parse_args()
    return args


# 主流程函数
def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载中文模型
    model = LlamaForCausalLM.from_pretrained(args.model_name_or_path, cache_dir="../cache/", torch_dtype=torch.float16)
    tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path, cache_dir="../cache/")

    model.to(device)
    model.eval()

    # 选择提示模板
    if args.prompt == 'alpaca':
        prompt_input, prompt_no_input = PROMPT_DICT_ALPACA["prompt_input"], PROMPT_DICT_ALPACA["prompt_no_input"]
    elif args.prompt == 'koala':
        prompt_input, prompt_no_input = PROMPT_DICT_KOALA["prompt_input"], PROMPT_DICT_KOALA["prompt_no_input"]

    # 根据数据集名称选择数据路径和解析键
    if args.dataset_name == "koala":
        dataset_path = '/root/autodl-tmp/test/evaluation/test_data/koala_test_set.jsonl'
        prompt_key = 'instruction'  # 假设指令在 `instruction` 字段
    else:
        raise ValueError("Unsupported dataset!")

    # 读取并处理数据集
    with open(dataset_path) as f:
        results = []
        dataset = list(f)
        for point in tqdm(dataset):
            point = json.loads(point)
            instruction = point[prompt_key]

            # 使用提示模板
            if 'input' in point and point['input']:
                prompt = prompt_input.format(instruction=instruction, input=point['input'])
            else:
                prompt = prompt_no_input.format(instruction=instruction)

            try:
                inputs = tokenizer(prompt, return_tensors="pt")
                input_ids = inputs.input_ids.to(device)
                generate_ids = model.generate(input_ids, max_length=args.max_length)
                outputs = \
                tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

                # 保存生成的输出
                point['raw_output'] = outputs
                point['response'] = outputs.split("回答:")[1]  # 根据生成的格式提取中文回答
                results.append(point)
            except ValueError as e:
                print(f"Skipping point due to error: {e}")

    # 保存生成的结果
    output_dir = os.path.join(args.model_name_or_path, 'test_inference')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    saved_name = args.dataset_name + "_" + str(args.max_length) + ".json"
    with open(os.path.join(output_dir, saved_name), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
