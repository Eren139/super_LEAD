import json
import numpy as np
import argparse
from tqdm import tqdm
import math


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_data_path", type=str, default='')
    parser.add_argument("--json_save_path", type=str, default='')
    parser.add_argument("--filter_threash", type=float, default=-1)
    parser.add_argument("--key_name", type=str, default='score', help='score')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print(args)

    with open(args.json_data_path, "r") as f:
        json_data = json.load(f)

    def sort_key(x):
        if math.isnan(x[args.key_name]):
            return (0, 0)
        return (1, x[args.key_name])

    filtered_data = [x for x in json_data if
                     (isinstance(x[args.key_name], (int, float)) and x[args.key_name] >= args.filter_threash)]

    new_data = sorted(filtered_data, key=sort_key, reverse=True)

    print(len(new_data))

    with open(args.json_save_path, 'w', encoding='utf-8') as file:
        json.dump(new_data, file, indent=4, ensure_ascii=False)

    print(f'Done: Data Selection: {args.json_data_path}')
    print(f'Number of filtered objects saved: {len(new_data)}')


if __name__ == '__main__':
    main()
