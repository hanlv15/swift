ckpt_list = [
    "/home/hanlv/workspace/code/research/infodemic/LLM/swift/examples/pytorch/llm/output/liar2/Llama-3-8B-Instruct/with_llama3_info/brave/data1.2-split=8:1:1-ratio=1.0/dora-r=8/lr=1.1e-4-20240723-17:05:29/checkpoint-611",
    "/home/hanlv/workspace/code/research/infodemic/LLM/swift/examples/pytorch/llm/output/liar2/Llama-3-8B-Instruct/with_llama3_info/brave/data1.2-split=8:1:1-ratio=1.0/dora-r=8/lr=9e-5-20240723-17:05:31/checkpoint-611",
    
    # "/home/hanlv/workspace/code/research/infodemic/LLM/swift/examples/pytorch/llm/output/liar2/Llama-3-8B-Instruct/with_llama3_info/brave/data1.2-split=8:1:1-ratio=1.0/dora-r=8/lr=1.6e-4-20240724-07:04:51/checkpoint-611",
    # "/home/hanlv/workspace/code/research/infodemic/LLM/swift/examples/pytorch/llm/output/liar2/Llama-3-8B-Instruct/with_llama3_info/brave/data1.2-split=8:1:1-ratio=1.0/dora-r=8/lr=1.7e-4-20240724-01:12:19/checkpoint-611",
]

path = "/home/hanlv/workspace/code/research/infodemic/LLM/swift/examples/pytorch/llm/output/covmis/Llama-3-8B-Instruct/with_llama3_info/brave/data1-split=8:2-ratio=1.0/dora-r=8"
# path = "/home/hanlv/workspace/code/research/infodemic/LLM/swift/examples/pytorch/llm/output/liar2/Llama-3-8B-Instruct/with_llama3_info/brave/data1-split=8:1:1-ratio=1.0-epochs=1/dora-r=8"
path = ""


data_dir = ""
# data_dir = "/home/hanlv/workspace/code/research/infodemic/LLM/swift/examples/pytorch/llm/my_data/liar2/with_llama3_info/brave/train_valid_split/8:1:1/test_data1.jsonl"

data_type = "valid" # test, valid

import subprocess
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# 要运行的Python文件的路径
if len(ckpt_list) == 0 and len(path) == 0:
    raise Exception()
# if len(ckpt_list) > 0 and len(path) > 0:
#     raise Exception()

if len(path) > 0:
    for file in os.listdir(path):
        ckpt_dir = os.path.join(path, file, "checkpoint-609")
        if os.path.exists(ckpt_dir):
            ckpt_list.append(ckpt_dir)

# 使用subprocess运行Python文件
for ckpt in ckpt_list:
    subprocess.run(["python", "infer_tuned.py", 
                    "--ckpt_dir", ckpt,
                    "--data_dir", data_dir,
                    "--data_type", data_type,
])

