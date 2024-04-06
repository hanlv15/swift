ckpt_list = [
    # "/home/hanlv/workspace/code/research/infodemic/LLM/swift/examples/pytorch/llm/output/merlinite-7b/with_mixtral_info/brave/data1-split=8:2-ratio=1.0/lora-r=3/lr=1.7e-4-20240401-18:01:10/checkpoint-608",
    # "/home/hanlv/workspace/code/research/infodemic/LLM/swift/examples/pytorch/llm/output/merlinite-7b/with_mixtral_info/brave/data1-split=8:2-ratio=1.0/lora-r=3/lr=1.8e-4-20240401-20:29:36/checkpoint-608"
]

path = "/home/hanlv/workspace/code/research/infodemic/LLM/swift/examples/pytorch/llm/output/openchat_3.5/with_mixtral_info/brave/data2-split=8:2-ratio=1.0/lora-r=3"

# path = ""

import subprocess
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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
    subprocess.run(["python", "infer_tuned.py", "--ckpt", ckpt])

