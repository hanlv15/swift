#!/bin/bash

# 定义参数列表
params_list=(
"/home/hanlv/workspace/code/research/infodemic/LLM/swift/examples/pytorch/llm/output/DareBeagle-7B-v2/with_solar_info/brave/data1-split=8:2-ratio=1.0/lora/lr=1.4e-4-20240123-10:16:15/checkpoint-609"
)

for param in "${params_list[@]}"
do
  CUDA_VISIBLE_DEVICES=0 python infer_tuned.py --ckpt "$param"
done
