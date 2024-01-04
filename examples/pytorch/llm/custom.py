# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, Optional, Tuple

from datasets import Dataset as HfDataset
from modelscope import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                        MsDataset)
from torch import dtype as Dtype
from transformers.utils.versions import require_version

from swift.llm import (LoRATM, Template, TemplateType, dataset_map,
                       get_dataset, get_model_tokenizer, get_template,
                       print_example, register_dataset, register_model,
                       register_template)
from swift.utils import get_logger

logger = get_logger()


class CustomModelType:
    tigerbot_7b = 'tigerbot-7b'
    tigerbot_13b = 'tigerbot-13b'
    tigerbot_13b_chat = 'tigerbot-13b-chat'

    orca2_7b = "orca-2-7b"
    openchat_35 = "openchat_3.5"
    neural_chat_7b = "neural-chat-7b-v3"
    solar_10_7b = "solar-10.7b-instruct"
    marcoroni_7b = "marcoroni-7B-v3"

class CustomTemplateType:
    tigerbot = 'tigerbot'

    orca2 = "orca-2"
    openchat_35 = "openchat_3.5"
    neural = "neural"
    solar="solar"
    marcoroni="marcoroni"

class CustomDatasetName:
    stsb_en = 'stsb-en'


@register_model(CustomModelType.tigerbot_7b,
                'TigerResearch/tigerbot-7b-base-v3', LoRATM.llama2,
                TemplateType.default_generation)
@register_model(CustomModelType.tigerbot_13b,
                'TigerResearch/tigerbot-13b-base-v2', LoRATM.llama2,
                TemplateType.default_generation)
@register_model(CustomModelType.tigerbot_13b_chat,
                'TigerResearch/tigerbot-13b-chat-v4', LoRATM.llama2,
                CustomTemplateType.tigerbot)
def get_tigerbot_model_tokenizer(model_dir: str,
                                 torch_dtype: Dtype,
                                 model_kwargs: Dict[str, Any],
                                 load_model: bool = True,
                                 **kwargs):
    use_flash_attn = kwargs.pop('use_flash_attn', False)
    if use_flash_attn:
        require_version('transformers>=4.34')
        logger.info('Setting use_flash_attention_2: True')
        model_kwargs['use_flash_attention_2'] = True
    model_config = AutoConfig.from_pretrained(
        model_dir, trust_remote_code=True)
    model_config.pretraining_tp = 1
    model_config.torch_dtype = torch_dtype
    logger.info(f'model_config: {model_config}')
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=True)
    model = None
    if load_model:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            config=model_config,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            **model_kwargs)
    return model, tokenizer

@register_model(CustomModelType.openchat_35,
                '/home/css/models/openchat-3.5-1210', LoRATM.llama2,
                CustomTemplateType.openchat_35)
def get_openchat35_model_tokenizer(model_dir: str,
                                 torch_dtype: Dtype,
                                 model_kwargs: Dict[str, Any],
                                 load_model: bool = True,
                                 **kwargs):
    model_config = AutoConfig.from_pretrained(model_dir)
    # model_config.pretraining_tp = 1
    model_config.torch_dtype = torch_dtype
    logger.info(f'model_config: {model_config}')
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = None
    if load_model:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            config=model_config,
            torch_dtype=torch_dtype,
            **model_kwargs)
    return model, tokenizer

@register_model(CustomModelType.orca2_7b,
                '/home/css/models/Orca-2-7b', LoRATM.llama2,
                CustomTemplateType.orca2)
def get_orca2_model_tokenizer(model_dir: str,
                                 torch_dtype: Dtype,
                                 model_kwargs: Dict[str, Any],
                                 load_model: bool = True,
                                 **kwargs):
    model_config = AutoConfig.from_pretrained(model_dir)
    # model_config.pretraining_tp = 1
    model_config.torch_dtype = torch_dtype
    logger.info(f'model_config: {model_config}')
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    model = None
    if load_model:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            config=model_config,
            torch_dtype=torch_dtype,
            **model_kwargs)
    return model, tokenizer

@register_model(CustomModelType.neural_chat_7b,
                '/home/css/models/neural-chat-7b-v3-3-Slerp', LoRATM.llama2,
                CustomTemplateType.neural)
def get_solar_model_tokenizer(model_dir: str,
                                 torch_dtype: Dtype,
                                 model_kwargs: Dict[str, Any],
                                 load_model: bool = True,
                                 **kwargs):
    model_config = AutoConfig.from_pretrained(model_dir)
    model_config.torch_dtype = torch_dtype
    logger.info(f'model_config: {model_config}')
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = None
    if load_model:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            config=model_config,
            torch_dtype=torch_dtype,
            **model_kwargs)
    return model, tokenizer

@register_model(CustomModelType.solar_10_7b,
                '/home/css/models/SOLAR-10.7B-Instruct-v1.0', LoRATM.llama2,
                CustomTemplateType.solar)
def get_solar_model_tokenizer(model_dir: str,
                                 torch_dtype: Dtype,
                                 model_kwargs: Dict[str, Any],
                                 load_model: bool = True,
                                 **kwargs):
    model_config = AutoConfig.from_pretrained(model_dir)
    model_config.torch_dtype = torch_dtype
    logger.info(f'model_config: {model_config}')
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = None
    if load_model:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            config=model_config,
            torch_dtype=torch_dtype,
            **model_kwargs)
    return model, tokenizer

@register_model(CustomModelType.marcoroni_7b,
                '/home/css/models/Marcoroni-7B-v3', LoRATM.llama2,
                CustomTemplateType.marcoroni)
def get_solar_model_tokenizer(model_dir: str,
                                 torch_dtype: Dtype,
                                 model_kwargs: Dict[str, Any],
                                 load_model: bool = True,
                                 **kwargs):
    model_config = AutoConfig.from_pretrained(model_dir)
    model_config.torch_dtype = torch_dtype
    logger.info(f'model_config: {model_config}')
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = None
    if load_model:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            config=model_config,
            torch_dtype=torch_dtype,
            **model_kwargs)
    return model, tokenizer

# Ref: https://github.com/TigerResearch/TigerBot/blob/main/infer.py
register_template(
    CustomTemplateType.tigerbot,
    Template(['{{SYSTEM}}'],
             ['\n\n### Instruction:\n{{QUERY}}\n\n### Response:\n'], [],
             [['eos_token_id']], ''))

register_template(
    CustomTemplateType.openchat_35,
    Template(
        [],
        ['GPT4 Correct User: {{QUERY}}<|end_of_turn|>GPT4 Correct Assistant:'],
        ['<|end_of_turn|>'], ['<|end_of_turn|>'], None, ['{{SYSTEM}}<|end_of_turn|>']))

# 不支持多轮对话
register_template(
    CustomTemplateType.orca2,
    Template(
        [], ['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'],
        ['<|im_end|>\n'], [None], None,
        ['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n']))


register_template(
    CustomTemplateType.neural,
    Template( 
        [],
        ['### User:\n{{QUERY}}\n### Assistant:\n'],
        ['\n'], ['</s>'], 'You are a helpful assistant.', ['### System:\n{{SYSTEM}}\n']))

register_template(
    CustomTemplateType.solar,
    Template( 
        [],
        ['### User:\n{{QUERY}}\n\n### Assistant:\n'],
        ['\n\n'], ['</s>'], None, ['### System:\n{{SYSTEM}}\n\n']))

register_template(
    CustomTemplateType.marcoroni,
    Template( 
        [],
        ['### Instruction:\n\n{{QUERY}}\n\n### Response:\n'],
        ['\n\n'], ['</s>'], None, ['### System:\n\n{{SYSTEM}}\n\n']))


def _preprocess_stsb(dataset: HfDataset) -> HfDataset:
    prompt = """Task: Based on the given two sentences, provide a similarity score between 0.0 and 5.0.
Sentence 1: {text1}
Sentence 2: {text2}
Similarity score: """
    query = []
    response = []
    for d in dataset:
        query.append(prompt.format(text1=d['text1'], text2=d['text2']))
        response.append(f"{d['label']:.1f}")
    return HfDataset.from_dict({'query': query, 'response': response})


@register_dataset(
    CustomDatasetName.stsb_en, 'huangjintao/stsb', task='text-generation')
def get_stsb_dataset(dataset_id_or_path: str,
                     **kwargs) -> Tuple[HfDataset, Optional[HfDataset]]:
    dataset_dict = MsDataset.load(dataset_id_or_path)
    train_dataset = dataset_dict['train'].to_hf_dataset()
    val_dataset = dataset_dict['validation'].to_hf_dataset()
    return tuple(
        _preprocess_stsb(dataset) for dataset in [train_dataset, val_dataset])


if __name__ == '__main__':
    # The Shell script can view `examples/pytorch/llm/scripts/custom`.
    # test dataset
    train_dataset, val_dataset = get_dataset([CustomDatasetName.stsb_en],
                                             check_dataset_strategy='warning')
    print(f'train_dataset: {train_dataset}')
    print(f'val_dataset: {val_dataset}')
    # test model base
    model, tokenizer = get_model_tokenizer(
        CustomModelType.tigerbot_13b, use_flash_attn=False)
    # test model chat
    model, tokenizer = get_model_tokenizer(
        CustomModelType.tigerbot_13b_chat, use_flash_attn=False)
    # test template
    template = get_template(CustomTemplateType.tigerbot, tokenizer)
    train_dataset = dataset_map(train_dataset, template.encode)
    print_example(train_dataset[0], tokenizer)