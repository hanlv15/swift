#!/bin/bash

# 检查是否提供了足够的参数
if [ "$#" -ne 6 ]; then
    echo "错误：需要提供6个参数"
    echo "用法: bash $0 <test_size> <train_ratio> <galore_rank> <learning_rate> <with_or_without_info> <data_version>"
    exit 1
fi

test_size=$1
train_ratio=$2
galore_rank=$3
learning_rate=$4 # 1e-5
with_or_without_info=$5
data_version=$6

num_epochs=1

split_type=$(echo "10 - $test_size * 10" | bc | awk '{print int($1)}'):$(echo "$test_size * 10" | bc | awk '{print int($1)}')

custom_train_dataset_path=my_data/$with_or_without_info/train_test_split/$split_type/subtrain_data$data_version/train_data_$train_ratio.jsonl
custom_val_dataset_path=my_data/$with_or_without_info/train_test_split/$split_type/test_data$data_version.jsonl

output_name="lr=$learning_rate-$(date +"%Y%m%d-%H:%M:%S")"
if [ "$train_ratio" = "1" ] || [ -z "$train_ratio" ]; then
    train_ratio=1.0
    custom_train_dataset_path=my_data/$with_or_without_info/train_test_split/$split_type/train_data$data_version.jsonl
fi

nproc_per_node=1
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
    --model_type openchat_3.5 \
    --model_id_or_path /home/css/models/openchat-3.5-0106 \
    --check_model_is_latest false \
    --sft_type full \
    --template_type openchat_3.5 \
    --dtype AUTO \
    --add_output_dir_suffix false \
    --output_dir output/openchat_3.5/$with_or_without_info/data$data_version-split=$split_type-ratio=$train_ratio/galore-r=$galore_rank/"$output_name" \
    --ddp_backend nccl \
    --custom_train_dataset_path $custom_train_dataset_path \
    --use_galore true \
    --galore_rank $galore_rank \
    --galore_update_proj_gap 400 \
    --dataset_test_ratio 0 \
    --val_dataset_sample -1 \
    --train_dataset_sample -1 \
    --eval_steps 1000 \
    --num_train_epochs 1 \
    --max_length $max_length \
    --max_new_tokens $max_length \
    --check_dataset_strategy warning \
    --gradient_checkpointing true \
    --batch_size 1 \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --learning_rate $learning_rate \
    --save_only_model true \
    --max_grad_norm 0.5 \
    --warmup_ratio 0.05 \
    --save_total_limit 1 \
    --logging_steps 5 \
    --use_flash_attn false \
    --do_sample false