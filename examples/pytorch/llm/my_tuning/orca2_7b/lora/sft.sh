#!/bin/bash

# 检查是否提供了足够的参数
if [ "$#" -ne 3 ]; then
    echo "错误：需要提供三个参数"
    echo "用法: bash $0 <test_size> <train_ratio> <with_or_without_info>"
    exit 1
fi

test_size=$1
train_ratio=$2
with_or_without_info=$3

split_type=$(echo "10 - $test_size * 10" | bc | awk '{print int($1)}'):$(echo "$test_size * 10" | bc | awk '{print int($1)}')

custom_train_dataset_path=my_data/$with_or_without_info/train_test_split/$split_type/subtrain_data/train_data_$train_ratio.jsonl
if [ "$train_ratio" = "1" ] || [ -z "$train_ratio" ]; then
    train_ratio=1.0
    custom_train_dataset_path=my_data/$with_or_without_info/train_test_split/$split_type/train_data.jsonl
fi
output_name="$(date +"%Y%m%d-%H:%M:%S")-split_type=$split_type-train_ratio=$train_ratio"

nproc_per_node=2
# eval_times=15
gradient_accumulation_steps=$(expr 16 / $nproc_per_node)
# num_train_data=$(echo "scale=0; 12192 * (1 - $test_size) * $train_ratio / 1" | bc)
# total_batch_size=$(expr $gradient_accumulation_steps \* $nproc_per_node)
# eval_steps=$(expr $num_train_data / $total_batch_size / $eval_times)


max_length=4096

PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=1,2 \
torchrun \
    --nproc_per_node=$nproc_per_node \
    --master_port 29500 \
    llm_sft.py \
    --model_type orca-2-7b \
    --model_cache_dir /home/css/models/Orca-2-7b \
    --check_model_is_latest false \
    --model_revision master \
    --sft_type lora \
    --tuner_backend peft \
    --template_type orca-2 \
    --dtype AUTO \
    --add_output_dir_suffix false \
    --output_dir output/orca-2-7b/without_info/"$output_name" \
    --ddp_backend nccl \
    --custom_train_dataset_path my_data/without_info/train_test_split/$split_type/subtrain_data/train_data_$train_ratio.jsonl \
    --dataset_test_ratio 0 \
    --train_dataset_sample -1 \
    --val_dataset_sample -1 \
    --num_train_epochs 1 \
    --max_length $max_length \
    --max_new_tokens $max_length \
    --check_dataset_strategy warning \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout_p 0.05 \
    --lora_target_modules ALL \
    --gradient_checkpointing true \
    --batch_size 1 \
    --weight_decay 0.01 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --max_grad_norm 0.5 \
    --warmup_ratio 0.03 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --use_flash_attn false
