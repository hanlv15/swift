import json
import prompt_rag
import torch
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'

dirs = ["../.."]
for _dir in dirs:
    if _dir not in sys.path:
        sys.path.append(_dir)

from swift.llm import (
    ModelType, get_vllm_engine, get_default_template_type,
    get_template, inference_vllm, VllmGenerationConfig
)
from custom import CustomModelType

# model_type = CustomModelType.mixtral_moe_7b_instruct_gptq_int4
model_type = CustomModelType.llama_3_70b_instruct_awq
llm_engine = get_vllm_engine(
    model_type, 
    # torch_dtype=torch.float16,  # 检查正确的数据类型！！！！
    tensor_parallel_size=2,
    max_model_len=4096,
    # gpu_memory_utilization=0.95,
    # model_id_or_path="/home/css/models/Mixtral-8x7B-Instruct-v0.1-GPTQ-int4",
    engine_kwargs = {
        # "enforce_eager": True,
        "max_num_seqs": 64,
        "seed": 42,
    }
)

# template_type = get_default_template_type(model_type)
# template = get_template(template_type, llm_engine.hf_tokenizer)

# generation_config = VllmGenerationConfig(
#     max_new_tokens=2048,
#     temperature=0,
# )

# get_resp_list = lambda request_list : inference_vllm(
#     llm_engine, template, request_list, 
#     generation_config=generation_config, 
#     use_tqdm=True, 
# )

# search_engine = "brave"
# model_name = 'llama3'
# K = 5
# sort = False

# with open(f"/home/hanlv/workspace/data/machine_learning/dataset/research/misinformation_dataset/COVMIS-2024/train_{search_engine}_search.json", "r") as f:
#     data_search = json.load(f)

# # data_search = data_search[:10] + [data_search[9690]]
# prompt_rag.update_train_search_llm(
#     model_name, get_resp_list, search_engine, data_search,
#     K=K, sort=sort
# )
