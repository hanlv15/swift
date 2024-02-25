ckpt_list = [
    # "/home/hanlv/workspace/code/research/infodemic/LLM/swift/examples/pytorch/llm/output/Mistral-7B-Instruct-v0.2/with_solar_info/brave/data1-split=8:2-ratio=1.0/adalora/lr=1.5e-3-20240225-13:37:27/checkpoint-609",
    # "/home/hanlv/workspace/code/research/infodemic/LLM/swift/examples/pytorch/llm/output/Mistral-7B-Instruct-v0.2/with_solar_info/brave/data1-split=8:2-ratio=1.0/adalora/lr=1.6e-3-20240225-16:34:33/checkpoint-609",
    "/home/hanlv/workspace/code/research/infodemic/LLM/swift/examples/pytorch/llm/output/Mistral-7B-Instruct-v0.2/with_solar_info/brave/data1-split=8:2-ratio=1.0/adalora/lr=1.7e-3-20240225-19:10:24/checkpoint-609",
    "/home/hanlv/workspace/code/research/infodemic/LLM/swift/examples/pytorch/llm/output/Mistral-7B-Instruct-v0.2/with_solar_info/brave/data1-split=8:2-ratio=1.0/adalora/lr=1.8e-3-20240225-21:45:30/checkpoint-609"
]
path = "/home/hanlv/workspace/code/research/infodemic/LLM/swift/examples/pytorch/llm/output/Mistral-7B-Instruct-v0.2/with_solar_info/brave/data1-split=8:2-ratio=1.0/adalora"
path = ""
import subprocess
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'

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

