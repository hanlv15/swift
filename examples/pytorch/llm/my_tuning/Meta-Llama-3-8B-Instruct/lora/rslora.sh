#!/bin/bash

# 检查是否提供了足够的参数
if [ "$#" -ne 8 ]; then
    echo "错误：需要提供8个参数"
    echo "用法: bash $0 <test_size> <train_ratio> <sft_type> <lora_rank> <learning_rate> <with_or_without_info> <data_version>"
    exit 1
fi

test_size=$1
train_ratio=$2
sft_type=$3
lora_rank=$4
learning_rate=$5 # 1e-4
with_or_without_info=$6
data_version=$7
device=$8

num_epochs=1

split_type=$(echo "10 - $test_size * 10" | bc | awk '{print int($1)}'):$(echo "$test_size * 10" | bc | awk '{print int($1)}')

custom_train_dataset_path=my_data/$with_or_without_info/train_test_split/$split_type/subtrain_data$data_version/train_data_$train_ratio.jsonl
custom_val_dataset_path=my_data/$with_or_without_info/train_test_split/$split_type/test_data$data_version.jsonl

output_name="lr=$learning_rate-$(date +"%Y%m%d-%H:%M:%S")"
if [ "$train_ratio" = "1" ] || [ -z "$train_ratio" ]; then
    train_ratio=1.0
    custom_train_dataset_path=my_data/$with_or_without_info/train_test_split/$split_type/train_data$data_version.jsonl
fi

lora_alpha=$(expr $lora_rank \* 4)

max_length=8192

NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" \
PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=$device \
python llm_sft.py \
    --model_type meta-llama-3-8B-instruct \
    --model_id_or_path /home/css/models/Meta-Llama-3-8B-Instruct \
    --check_model_is_latest false \
    --sft_type $sft_type \
    --tuner_backend peft \
    --template_type _llama3 \
    --dtype AUTO \
    --add_output_dir_suffix false \
    --output_dir output/Llama-3-8B-Instruct/$with_or_without_info/data$data_version-split=$split_type-ratio=$train_ratio/rslora-r=$lora_rank/"$output_name" \
    --dataset $custom_train_dataset_path#-1 \
    --dataset_test_ratio 0 \
    --num_train_epochs $num_epochs \
    --max_length $max_length \
    --max_new_tokens 512 \
    --check_dataset_strategy warning \
    --lora_rank $lora_rank \
    --lora_alpha $lora_alpha \
    --lora_dropout_p 0.05 \
    --lora_target_modules ALL \
    --use_rslora true \
    --lora_dtype AUTO \
    --gradient_checkpointing true \
    --batch_size 1 \
    --weight_decay 0.1 \
    --learning_rate $learning_rate \
    --gradient_accumulation_steps 16 \
    --max_grad_norm 0.5 \
    --warmup_ratio 0.03 \
    --save_total_limit 1 \
    --logging_steps 10 \
    --use_flash_attn false \
    --full_determinism true \
    --do_sample false

