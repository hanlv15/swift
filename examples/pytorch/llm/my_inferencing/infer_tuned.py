# 修改checkpoint路径
import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--ckpt_dir", type=str)
args = parser.parse_args()
ckpt_dir = args.ckpt_dir

import sys
import json
import jsonlines
import os
import subprocess

dirs = [".."]
for _dir in dirs:
    if _dir not in sys.path:
        sys.path.append(_dir)

from swift.llm import (
    get_model_tokenizer, get_template, get_vllm_engine, 
    inference_vllm, VllmGenerationConfig, LoRARequest
)
from swift.tuners import Swift
from custom import CustomModelType, CustomTemplateType
import evaluation

with open(f"{ckpt_dir}/sft_args.json", "r") as f:
    sft_args = json.load(f)

with jsonlines.open(f"{ckpt_dir}/../logging.jsonl", 'r') as f:
    for item in f.iter():
        training_result = item
train_loss = training_result["train_loss"]

def get_engine_config_request(ckpt_dir):
    # 检查checkpoint是否为peft格式
    if os.path.exists(ckpt_dir + '/default'):
        if not os.path.exists(ckpt_dir + '-peft'):
            subprocess.run(["python", "../llm_export.py", "--ckpt", ckpt_dir, "--to_peft_format", "true"])
        ckpt_dir = ckpt_dir + '-peft'


    lora_request = LoRARequest('default-lora', 1, ckpt_dir)

    model_type, template_type = sft_args["model_type"], sft_args["template_type"]
    vllm_engine = get_vllm_engine(
        model_type, 
        tensor_parallel_size=1,
        seed=42,
        enable_lora=True,
        max_loras=1, 
        max_lora_rank=16,
        engine_kwargs={"max_num_seqs": 128}
    )

    template = get_template(template_type, vllm_engine.hf_tokenizer)
    generation_config = VllmGenerationConfig(
        max_new_tokens=512,
        do_sample=False,
        temperature=0,
    )
    return vllm_engine, template, generation_config, lora_request

evaluation.cal_metric_single_llm(
    get_engine_config_request, inference_vllm, 
    sft_args, ckpt_dir, train_loss, save=True, 
)

