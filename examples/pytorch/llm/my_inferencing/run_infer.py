ckpt_list = [
    # "/home/hanlv/workspace/code/research/infodemic/LLM/swift/examples/pytorch/llm/output/Llama-3-8B-Instruct/with_llama3_info/brave/data1-split=8:2-ratio=1.0/vera-r=256/lr=4.6e-2-20240623-01:48:03/checkpoint-609",
    # "/home/hanlv/workspace/code/research/infodemic/LLM/swift/examples/pytorch/llm/output/Llama-3-8B-Instruct/with_llama3_info/brave/data1-split=8:2-ratio=1.0/vera-r=256/lr=4.7e-2-20240623-03:45:41/checkpoint-609",
    # "/home/hanlv/workspace/code/research/infodemic/LLM/swift/examples/pytorch/llm/output/Llama-3-8B-Instruct/with_llama3_info/brave/data1-split=8:2-ratio=1.0/rslora-r=8/lr=1.8e-4-20240623-05:43:01/checkpoint-609",
    # "/home/hanlv/workspace/code/research/infodemic/LLM/swift/examples/pytorch/llm/output/Llama-3-8B-Instruct/with_llama3_info/brave/data1-split=8:2-ratio=1.0/rslora-r=8/lr=1.9e-4-20240623-08:09:44/checkpoint-609",

    # "/home/hanlv/workspace/code/research/infodemic/LLM/swift/examples/pytorch/llm/output/Llama-3-8B-Instruct/with_llama3_info/brave/data1-split=8:2-ratio=1.0/vera-r=256/lr=4.8e-2-20240623-01:48:10/checkpoint-609",
    # "/home/hanlv/workspace/code/research/infodemic/LLM/swift/examples/pytorch/llm/output/Llama-3-8B-Instruct/with_llama3_info/brave/data1-split=8:2-ratio=1.0/vera-r=256/lr=4.9e-2-20240623-03:44:50/checkpoint-609",
    # "/home/hanlv/workspace/code/research/infodemic/LLM/swift/examples/pytorch/llm/output/Llama-3-8B-Instruct/with_llama3_info/brave/data1-split=8:2-ratio=1.0/rslora-r=8/lr=1.6e-4-20240623-05:41:25/checkpoint-609",
    # "/home/hanlv/workspace/code/research/infodemic/LLM/swift/examples/pytorch/llm/output/Llama-3-8B-Instruct/with_llama3_info/brave/data1-split=8:2-ratio=1.0/rslora-r=8/lr=2.1e-4-20240623-08:07:37/checkpoint-609",

    "/home/hanlv/workspace/code/research/infodemic/LLM/swift/examples/pytorch/llm/output/Llama-3-8B-Instruct/with_llama3_info/brave/data1-split=8:2-ratio=1.0/vera-r=256/lr=2.5e-2-20240623-01:48:18/checkpoint-609",
    "/home/hanlv/workspace/code/research/infodemic/LLM/swift/examples/pytorch/llm/output/Llama-3-8B-Instruct/with_llama3_info/brave/data1-split=8:2-ratio=1.0/vera-r=256/lr=5.3e-2-20240623-03:45:28/checkpoint-609",
    "/home/hanlv/workspace/code/research/infodemic/LLM/swift/examples/pytorch/llm/output/Llama-3-8B-Instruct/with_llama3_info/brave/data1-split=8:2-ratio=1.0/rslora-r=8/lr=4e-5-20240623-05:42:51/checkpoint-609",
    "/home/hanlv/workspace/code/research/infodemic/LLM/swift/examples/pytorch/llm/output/Llama-3-8B-Instruct/with_llama3_info/brave/data1-split=8:2-ratio=1.0/rslora-r=8/lr=3e-5-20240623-08:10:13/checkpoint-609"
]

path = "/home/hanlv/workspace/code/research/infodemic/LLM/swift/examples/pytorch/llm/output/Llama-3-8B-Instruct/with_llama3_info/brave/data1-split=8:2-ratio=1.0/rslora-r=8"

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

