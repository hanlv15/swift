"""将数据分成 训练数据 和 测试数据 """

import random
import numpy as np
import jsonlines
from sklearn.model_selection import train_test_split
import argparse
import os

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--test_size", type=float, default=0.2)
args = parser.parse_args()

DEFAULT_SEED = 42
def set_seed(seed=DEFAULT_SEED):
    np.random.seed(seed)
    random.seed(seed)

# data_dir = "/home/hanlv/workspace/code/research/infodemic/LLM/LoRA/swift_data/"
search_engine = "brave"
model_name = "mixtral"
data_dir = f"./with_{model_name}_info/{search_engine}/"
version = "1"

dict_list = []
with jsonlines.open(data_dir + f"data{version}.jsonl", mode="r") as file_jsonl:
    for item in file_jsonl.iter():
        dict_list.append(item)
    set_seed()
    train_list, test_list = train_test_split(dict_list, test_size=args.test_size, shuffle=True)


split_type = f"{int(10 - args.test_size * 10)}:{int(args.test_size * 10)}"

data_path = data_dir + f"train_test_split/{split_type}/"

if not os.path.exists(data_path):
    os.mkdir(data_path)

with jsonlines.open(
    data_path + f"train_data{version}.jsonl", 
    mode="w"
) as file_jsonl:
    for line in train_list:
        file_jsonl.write(line)


with jsonlines.open(data_path + f"test_data{version}.jsonl", mode="w") as file_jsonl:
    for line in test_list:
        file_jsonl.write(line)

