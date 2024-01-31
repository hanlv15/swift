import json
import prompt_rag
import argparse
import subprocess

search_engine = "brave"
K = 10
sort = False


def get_gpu_count():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=gpu_name', '--format=csv'])
    gpu_list = output.decode().strip().split('\n')[1:]
    return len(gpu_list)

num_gpus = get_gpu_count()

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--part", type=int)
args = parser.parse_args()
part = args.part

with open(f"/root/workspace/data/dataset/research/misinformation_dataset/COVMIS-main/data/train_{search_engine}_search.json", "r") as f:
    data_search = json.load(f)

n = num_gpus # gpu个数=将数据划分的个数
assert f"part需要在1~{n}之间", 1 <= part <= n

_len = len(data_search) // n
if part == n:
    data_search = data_search[_len * (n-1):]
else:
    data_search = data_search[_len * (n-1): _len * n]

prompt_rag.update_train_search_llm(
    f"SOLAR-10.7B-Instruct-v1.0-{part}", "solar", 8001,
    search_engine, data_search, part,
    K=K, sort=sort
)




