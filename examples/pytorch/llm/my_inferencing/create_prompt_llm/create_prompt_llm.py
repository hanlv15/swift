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
from custom import CustomModelType, CustomTemplateType

model_type = CustomModelType.mixtral_moe_7b_instruct_gptq_int4
llm_engine = get_vllm_engine(
    model_type, 
    torch_dtype=torch.float16,  # 检查正确的数据类型！！！！
    tensor_parallel_size=2,
    
    engine_kwargs = {
        # "enforce_eager": True,
        "max_num_seqs": 64,
        "seed": 42,
    }
)

template_type = get_default_template_type(model_type)
template = get_template(template_type, llm_engine.hf_tokenizer)

generation_config = VllmGenerationConfig(
    max_new_tokens=4096,
    temperature=0,
)

get_resp_list = lambda request_list : inference_vllm(
    llm_engine, template, request_list, 
    generation_config=generation_config, 
    use_tqdm=True,
)

search_engine = "brave"
model_name = 'mixtral'
K = 5
sort = False

with open(f"/home/hanlv/workspace/data/machine_learning/dataset/research/misinformation_dataset/COVMIS-main/data/train_{search_engine}_search.json", "r") as f:
    data_search = json.load(f)

prompt_rag.update_train_search_llm(
    model_name, get_resp_list, search_engine, data_search,
    K=K, sort=sort
)

