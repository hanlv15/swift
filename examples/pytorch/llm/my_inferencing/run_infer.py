ckpt_list = [
    
]

path = "/home/hanlv/workspace/code/research/infodemic/LLM/swift/examples/pytorch/llm/output/openchat_3.5/with_llama3_info/brave/data1-split=8:2-ratio=1.0/vera-r=256"
# path = "/home/hanlv/workspace/code/research/infodemic/LLM/swift/examples/pytorch/llm/output/Mistral-7B-Instruct-v0.3/with_llama3_info/brave/data1-split=8:2-ratio=1.0/lora_plus-r=8"
# path = "/home/hanlv/workspace/code/research/infodemic/LLM/swift/examples/pytorch/llm/output/Mistral-7B-Instruct-v0.3/with_llama3_info/brave/data1-split=8:2-ratio=1.0/dora-r=8"

# path = ""

import subprocess
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1' # 0, 2

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

