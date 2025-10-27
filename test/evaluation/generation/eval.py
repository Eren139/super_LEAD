import argparse
import json
import os
import time
import numpy as np
import sys
from openai import OpenAI
from tqdm import tqdm
import logging
from typing import List, Dict, Any
import tiktoken

gpt_encoder = tiktoken.get_encoding("cl100k_base")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
client = OpenAI(
    base_url="https://xiaoai.plus/v1",
    api_key="sk-KcSNoz6uv6WqeIQ4Z6OgkjgK5S4mIVg4EjdYW65d38zbOmYB"
)

def dispatch_openai_requests(
        messages_list: List[List[Dict[str, Any]]],
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        wait_time: int = 5  # 增加等待时间参数
) -> List[str]:
    """Dispatches requests to OpenAI API synchronously.
    """
    responses = []
    for x in messages_list:
        success = False
        while not success:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=x,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                )
                responses.append(response)
                success = True
            except Exception as e:
                error_message = str(e)
                if "429" in str(e):
                    logger.info("Too many requests. Retrying after a delay.")
                    time.sleep(wait_time)
                elif "400" in error_message:
                    logger.info("Bad Request. Skipping to the next request.")
                    success = True  # Skip to the next request
                else:
                    raise e
        time.sleep(wait_time)
    return responses

def parse_score(review):
    try:
        score_pair = review.split("\n")[0]
        score_pair = score_pair.replace(",", " ")
        sp = score_pair.split(" ")
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            raise Exception("Invalid score pair.")
    except Exception as e:
        logger.error(
            f"{e}\nContent: {review}\n" "You must manually fix the score pair."
        )
        return [-1, -1]

def gen_prompt(ques, ans1, ans2):

    sys_prompt = "You are a helpful and precise assistant for checking the quality of the answer."
    prompt_template = "[Question]\n{question}\n\n[The Start of Assistant 1's Answer]\n{answer_1}\n\n[The End of Assistant 1's Answer]\n\n[The Start of Assistant 2's Answer]\n{answer_2}\n\n[The End of Assistant 2's Answer]\n\n[System]\n{criteria}\n\n"
    criteria = "We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.\nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment."
    prompt = prompt_template.format(
        question=ques, answer_1=ans1, answer_2=ans2, criteria=criteria
    )
    return sys_prompt, prompt


def get_json_list(file_path):
    file_path = os.path.expanduser(file_path)
    with open(file_path, "r") as f:
        json_list = []
        for line in f:
            json_list.append(json.loads(line))
        return json_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, default='sk-KcSNoz6uv6WqeIQ4Z6OgkjgK5S4mIVg4EjdYW65d38zbOmYB')
    parser.add_argument("--api_model", type=str, default='gpt-3.5-turbo')
    parser.add_argument("--api_base", type=str, default='https://xiaoai.plus/v1')
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Batch size to call OpenAI GPT",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1024,
        help="maximum number of tokens produced in the output",
    )
    parser.add_argument(
        "--wait_time",
        type=int,
        default=5,
        help="Time to wait between requests"
    )
    args = parser.parse_args()
    if args.api_base != '':
        client.api_base = args.api_base
    client.api_key = args.api_key

    # 文件路径列表
    wraped_files = [
        r'D:\code\Superfiltering-main\test\scripts\logs\lei_tuhe5%2525-VS-en_ifd\koala_wrap.json',
        r'D:\code\Superfiltering-main\test\scripts\logs\lei_tuhe5%2525-VS-en_ifd\lima_wrap.json',
        r'D:\code\Superfiltering-main\test\scripts\logs\lei_tuhe5%2525-VS-en_ifd\sinstruct_wrap.json',
        r'D:\code\Superfiltering-main\test\scripts\logs\lei_tuhe5%2525-VS-en_ifd\vicuna_wrap.json',
        r'D:\code\Superfiltering-main\test\scripts\logs\lei_tuhe5%2525-VS-en_ifd\wizardlm_wrap.json'
    ]

    # 遍历文件路径
    for wraped_file in wraped_files:
        print('Begin:', wraped_file)

        wraped_info = json.load(open(wraped_file, encoding='utf-8'))
        meta_info = wraped_info['Meta_Info']
        dataset_name = meta_info['dataset_name']
        qa_jsons = wraped_info['data']

        if (dataset_name == "vicuna"):
            prompt_key = 'text'
        elif (dataset_name == "koala"):
            prompt_key = 'prompt'
        elif (dataset_name == "sinstruct"):
            prompt_key = 'instruction'
        elif (dataset_name == "wizardlm"):
            prompt_key = 'Instruction'
        elif (dataset_name == "lima"):
            prompt_key = 'conversations'

        total_len = len(qa_jsons)
        question_idx_list = list(range(total_len))

        predictions_all = []
        for reverse in range(2):  # reverse or not
            message_list = []
            token_len_list = []

            for i in question_idx_list:

                instruction = qa_jsons[i][prompt_key]
                ques = instruction

                if reverse:  # reverse = 1, secondly
                    ans1 = qa_jsons[i]['Answer2']
                    ans2 = qa_jsons[i]['Answer1']
                else:  # reverse = 0, firstly
                    ans1 = qa_jsons[i]['Answer1']
                    ans2 = qa_jsons[i]['Answer2']
                sys_prompt, prompt = gen_prompt(ques, ans1, ans2)

                message = [
                    {"role": "system", "content": sys_prompt},
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ]
                message_list.append(message)
                token_len_list.append(len(gpt_encoder.encode(prompt)))

            predictions = []
            i = 0
            wait_base = 10
            retry = 0
            error = 0
            pbar = tqdm(total=len(message_list))
            batch_size = args.batch_size
            while (i < len(message_list)):
                token_limit_in_current_batch = min(args.max_tokens, 4070 - max(token_len_list[i:i + batch_size]))
                try:
                    batch_predictions = dispatch_openai_requests(
                        messages_list=message_list[i:i + batch_size],
                        model=args.api_model,
                        temperature=0.0,
                        max_tokens=token_limit_in_current_batch,
                        top_p=1.0,
                        wait_time=args.wait_time  # 传递等待时间参数
                    )
                    predictions += batch_predictions
                    retry = 0
                    i += batch_size
                    wait_base = 10
                    pbar.update(batch_size)
                except:
                    retry += 1
                    error += 1
                    print("Batch error: ", i, i + batch_size)
                    print("retry number: ", retry)
                    print("error number: ", error)
                    time.sleep(wait_base)
                    wait_base = wait_base * 2
            pbar.close()
            predictions_all.append(predictions)

        all_scores = []
        for reverse in range(2):
            scores_list = []
            predictions = predictions_all[reverse]
            for idx, prediction in enumerate(predictions):
                review = prediction.choices[0].message.content
                scores = parse_score(review)
                review_key = 'review' if not reverse else 'review_reverse'
                scores_key = 'scores' if not reverse else 'scores_reverse'
                qa_jsons[idx][review_key] = review
                qa_jsons[idx][scores_key] = str(scores)
                scores_list.append(scores)

            all_scores.append(scores_list)
            avg_scores = np.array(scores_list).mean(0)
            avg_key = 'average_scores' if not reverse else 'average_scores_reverse'
            meta_info[avg_key] = str(avg_scores.tolist())

        wraped_info['Meta_Info'] = meta_info
        wraped_info['data'] = qa_jsons

        if 'gpt-4' in args.api_model:
            output_review_file = wraped_file.strip('.json') + '_reviews_gpt4.json'
        elif 'gpt-3.5' in args.api_model:
            output_review_file = wraped_file.strip('.json') + '_reviews_gpt3.5.json'

        with open(f"{output_review_file}", "w", encoding="utf-8") as f:
            json.dump(wraped_info, f, ensure_ascii=False, indent=4)

        print('Finish:', wraped_file)
