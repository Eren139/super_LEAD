import json
import argparse
import math

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_data_path", type=str, default='')
    parser.add_argument("--json_save_path", type=str, default='')
    parser.add_argument("--sample_rate", type=float, default=0.2)
    parser.add_argument("--filter_threash", type=float, default=1)
    parser.add_argument("--key_name", type=str, default='ifd_ppl',help='ifd_ppl')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print(args)

    with open(args.json_data_path, "r") as f:
        json_data = json.load(f)
    sample_num = int(52002*args.sample_rate)

    def sort_key(x):
        if math.isnan(x[args.key_name]):
            return (0, 0) 
        return (1, x[args.key_name])

    filtered_data = [x for x in json_data if (isinstance(x[args.key_name], (int, float)) and x[args.key_name] < args.filter_threash)]
    new_data = sorted(filtered_data, key=sort_key, reverse=True)

    new_data = new_data[:sample_num]
    print(len(new_data))
    with open(args.json_save_path, 'w') as file:
        json.dump(new_data, file, indent=4)
    
    print('Done: Data Selection:',args.json_data_path)


if __name__ == '__main__':
    main()