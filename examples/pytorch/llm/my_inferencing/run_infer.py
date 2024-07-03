ckpt_list = [
    # "/home/hanlv/workspace/code/research/infodemic/LLM/swift/examples/pytorch/llm/output/Llama-3-8B-Instruct/with_llama3_info/brave/data1-split=8:2-ratio=1.0/pissa-r=8/lr=8e-5-20240703-17:56:39/checkpoint-609",
    # "/home/hanlv/workspace/code/research/infodemic/LLM/swift/examples/pytorch/llm/output/Llama-3-8B-Instruct/with_llama3_info/brave/data1-split=8:2-ratio=1.0/pissa-r=8/lr=1.2e-4-20240703-17:56:43/checkpoint-609",
    "/home/hanlv/workspace/code/research/infodemic/LLM/swift/examples/pytorch/llm/output/Llama-3-8B-Instruct/with_llama3_info/brave/data1-split=8:2-ratio=1.0/pissa-r=8/lr=1.3e-4-20240703-17:56:46/checkpoint-609",
]
# path = "/home/hanlv/workspace/code/research/infodemic/LLM/swift/examples/pytorch/llm/output/Phi-3-medium-4k-instruct/with_llama3_info/brave/data1-split=8:2-ratio=1.0/lora-r=8"
# path = "/home/hanlv/workspace/code/research/infodemic/LLM/swift/examples/pytorch/llm/output/Llama-3-8B-Instruct/with_llama3_info/brave/data1-split=8:2-ratio=1.0/pissa-r=8"
path = "/home/hanlv/workspace/code/research/infodemic/LLM/swift/examples/pytorch/llm/output/Llama-3-8B-Instruct/with_llama3_info/brave/data1-split=8:2-ratio=1.0/vera-r=256"

path = ""

import subprocess
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

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

