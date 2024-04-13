"""将训练数据分出 不同的比例"""

import random
import numpy as np
import jsonlines
from sklearn.model_selection import train_test_split
import argparse
import os

# parser = argparse.ArgumentParser(description='manual to this script')
# parser.add_argument("--size", type=float, default=0.1)
# args = parser.parse_args()

DEFAULT_SEED = 42
def set_seed(seed=DEFAULT_SEED):
    np.random.seed(seed)
    random.seed(seed)

# data_dir = "/home/hanlv/workspace/code/research/infodemic/LLM/LoRA/swift_data/"
search_engine = "brave"
model_name = "mixtral"
data_dir = f"./with_{model_name}_info/{search_engine}/"
version = "1"

data_path = data_dir + f"train_test_split/8:2/"
train_data = []
with jsonlines.open(
    data_path + f"train_data{version}.jsonl", 
    mode="r"
) as file_jsonl:
    for item in file_jsonl.iter():
        train_data.append(item)

set_seed()

for size in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    sample_size = int(size * len(train_data))
    new_train_data = random.sample(train_data, sample_size)
        
    with jsonlines.open(data_path + f"subtrain_data{version}/train_data_{size}.jsonl", mode="w") as file_jsonl:
        for line in new_train_data:
            file_jsonl.write(line)

