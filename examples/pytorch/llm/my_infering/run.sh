#!/bin/bash

# 定义参数列表
params_list=(
"/home/hanlv/workspace/code/research/infodemic/LLM/swift/examples/pytorch/llm/output/Mistral-7B-Instruct-v0.2/with_solar_info/brave/data1-split=8:2-ratio=1.0/lora/lr=1.5e-4-20240119-19:54:10/checkpoint-609"
)

# 循环运行同一个 Python 脚本，每次传递不同的参数
for param in "${params_list[@]}"
do
  CUDA_VISIBLE_DEVICES=0 python infer_tuned.py --ckpt "$param"
done
