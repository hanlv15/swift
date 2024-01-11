#!/bin/bash

# 检查是否提供了足够的参数
if [ "$#" -ne 4 ]; then
    echo "错误：需要提供4个参数"
    echo "用法: bash $0 <test_size> <train_ratio> <learning_rate> <data_version>"
    exit 1
fi

test_size=$1
train_ratio=$2
learning_rate=$3 # 1e-4
data_version=$4
with_or_without_info=with_solar_info

num_epochs=1

split_type=$(echo "10 - $test_size * 10" | bc | awk '{print int($1)}'):$(echo "$test_size * 10" | bc | awk '{print int($1)}')

custom_train_dataset_path=my_data/$with_or_without_info/train_test_split/$split_type/subtrain_data$data_version/train_data_$train_ratio.jsonl
custom_val_dataset_path=my_data/$with_or_without_info/train_test_split/$split_type/test_data$data_version.jsonl

output_name="split_type=$split_type-train_ratio=$train_ratio-$(date +"%Y%m%d-%H:%M:%S")"
if [ "$train_ratio" = "1" ] || [ -z "$train_ratio" ]; then
    train_ratio=1.0
    custom_train_dataset_path=my_data/$with_or_without_info/train_test_split/$split_type/train_data$data_version.jsonl
fi

nproc_per_node=2
# eval_times=15
gradient_accumulation_steps=$(expr 16 / $nproc_per_node)
# num_train_data=$(echo "scale=0; 12192 * (1 - $test_size) * $train_ratio / 1" | bc)
# total_batch_size=$(expr $gradient_accumulation_steps \* $nproc_per_node)
# eval_steps=$(expr $num_train_data \* num_epochs / $total_batch_size / $eval_times)


max_length=8192

PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=1,2 \
torchrun \
    --nproc_per_node=$nproc_per_node \
    --master_port 29505 \
    llm_sft.py \
    --model_type marcoroni-7B-v3 \
    --model_cache_dir /home/css/models/Marcoroni-7B-v3 \
    --check_model_is_latest false \
    --model_revision master \
    --sft_type lora \
    --tuner_backend peft \
    --template_type marcoroni \
    --dtype fp16 \
    --add_output_dir_suffix false \
    --output_dir output/Marcoroni-7B-v3/$with_or_without_info/data$data_version/lr=$learning_rate/"$output_name" \
    --ddp_backend nccl \
    --custom_train_dataset_path $custom_train_dataset_path \
    --dataset_test_ratio 0 \
    --train_dataset_sample -1 \
    --val_dataset_sample -1 \
    --num_train_epochs $num_epochs \
    --max_length $max_length \
    --max_new_tokens $max_length \
    --check_dataset_strategy warning \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout_p 0.05 \
    --lora_target_modules ALL \
    --lora_dtype AUTO \
    --gradient_checkpointing true \
    --batch_size 1 \
    --weight_decay 0.01 \
    --learning_rate $learning_rate \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --max_grad_norm 0.5 \
    --warmup_ratio 0.03 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --use_flash_attn false
