from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import torch
import argparse
import json
import os
from tqdm import tqdm

PROMPT_DICT_ALPACA = {
    "prompt_input": (
        "下面是一条描述任务的指令，并配有提供进一步上下文的输入。请生成一个合适的响应来完成该请求。\n\n"
        "### 指令:\n{instruction}\n\n### 输入:\n{input}\n\n### 回答:"
    ),
    "prompt_no_input": (
        "{instruction}"
    ),
}
PROMPT_DICT_WIZARDLM = {
    "prompt_input": (
        "{instruction}\n{input}\n\n### 回答:"
    ),
    "prompt_no_input": (
        "{instruction}\n\n### 回答:"
    ),
}
PROMPT_DICT_VICUNA = {
    "prompt_input": (
        "一名好奇的用户与人工智能助手的对话。助手会给出有帮助的、详细且礼貌的回答。用户: {instruction}\n输入:\n{input} 助手:"
    ),
    "prompt_no_input": (
        "一名好奇的用户与人工智能助手的对话。助手会给出有帮助的、详细且礼貌的回答。用户: {instruction} 助手:"
    ),
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="要使用的数据集名称 (通过 datasets 库).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default='alpaca',
        help="alpaca, wiz, vicuna.",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=None,
        help=(
            "用于评估的 beam 数量。此参数将传递给 ``model.generate``，"
            "并在 ``evaluate`` 和 ``predict`` 中使用。"
        ),
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="预训练模型的路径或来自 huggingface.co/models 的模型标识符。",
        required=False,
    )
    parser.add_argument("--seed", type=int, default=0, help="用于可重复训练的种子。")
    parser.add_argument("--max_length", type=int, default=1024)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16, trust_remote_code=True).cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    model.to(device)
    model.eval()

    if args.prompt == 'alpaca':
        prompt_input, prompt_no_input = PROMPT_DICT_ALPACA["prompt_input"], PROMPT_DICT_ALPACA["prompt_no_input"]
    elif args.prompt == 'wiz':
        prompt_input, prompt_no_input = PROMPT_DICT_WIZARDLM["prompt_input"], PROMPT_DICT_WIZARDLM["prompt_no_input"]
    elif args.prompt == 'vicuna':
        prompt_input, prompt_no_input = PROMPT_DICT_VICUNA["prompt_input"], PROMPT_DICT_VICUNA["prompt_no_input"]

    if args.dataset_name == "laiw":
        dataset_path = '/root/autodl-tmp/test/evaluation/test_data/laiw.jsonl'
        prompt_key = 'prompt'
    elif args.dataset_name == "disc":
        dataset_path = '/root/autodl-tmp/test/evaluation/test_data/disc_qa.jsonl'
        prompt_key = 'prompt'
    elif args.dataset_name == "lawbench":
        dataset_path = '/root/autodl-tmp/test/evaluation/test_data/lawbench_test.jsonl'
        prompt_key = 'prompt'
    elif args.dataset_name == "self_built":
        dataset_path = '/root/autodl-tmp/test/evaluation/test_data/Self_built.jsonl'
        prompt_key = 'prompt'

    with open(dataset_path) as f:
        results = []
        dataset = list(f)
        for point in tqdm(dataset):
            point = json.loads(point)
            instruction = point[prompt_key]
            if args.dataset_name == "sinstruct":
                instances = point['instances']
                assert len(instances) == 1
                if instances[0]['input']:
                    prompt = prompt_input.format_map({"instruction": instruction, 'input': instances[0]['input']})
                else:
                    prompt = prompt_no_input.format_map({"instruction": instruction})
            else:
                prompt = prompt_no_input.format_map({"instruction": instruction})
            # 模型回答包含prompt的情况，可以打印一下raw_output看看格式
            try:
                # 使用 internlm 的 chat 方法来生成响应
                response, history = model.chat(tokenizer, prompt, history=[])

                # 保存完整的 raw_output，包括 prompt 和模型回复
                point['raw_output'] = f"下面是一条描述任务的指令。请生成一个合适的响应来完成该请求。\n\n### 指令:\n{instruction}\n\n### 回答:{response}"

                # 只提取模型的回复
                point['response'] = response  # 直接使用模型的回复，不再拆分

                results.append(point)
            except ValueError as e:
                print(f"Skipping point due to error: {e}")

    output_dir = os.path.join(args.model_name_or_path, 'test_inference')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    saved_name = args.dataset_name + "_" + str(args.max_length) + ".json"
    with open(os.path.join(output_dir, saved_name), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
