#!/bin/bash


# 定义参数列表
params_list=("/home/hanlv/workspace/code/research/infodemic/LLM/swift/examples/pytorch/llm/output/openchat_3.5/with_solar_info/brave/data1-split=8:2-ratio=1.0/lr=7e-5-20240118-07:32:59/checkpoint-609" \
"/home/hanlv/workspace/code/research/infodemic/LLM/swift/examples/pytorch/llm/output/openchat_3.5/with_solar_info/brave/data1-split=8:2-ratio=1.0/lr=8e-5-20240118-05:29:34/checkpoint-609" \
"/home/hanlv/workspace/code/research/infodemic/LLM/swift/examples/pytorch/llm/output/openchat_3.5/with_solar_info/brave/data1-split=8:2-ratio=1.0/lr=9e-5-20240118-03:26:10/checkpoint-609"
)

# 循环运行同一个 Python 脚本，每次传递不同的参数
for param in "${params_list[@]}"
do
  CUDA_VISIBLE_DEVICES=1 python infer_tuned.py --ckpt "$param"
done
