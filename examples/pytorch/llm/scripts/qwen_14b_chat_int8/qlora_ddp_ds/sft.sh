# Experimental environment: 2 * A10
# 2 * 20GB GPU memory
nproc_per_node=3

PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0,1,2 \
torchrun \
    --nproc_per_node=$nproc_per_node \
    --master_port 29500 \
    llm_sft.py \
    --model_type qwen-14b-chat-int8\
    --model_cache_dir /home/css/models/Qwen-14B-Chat-Int8 \
    --model_revision master \
    --sft_type lora \
    --tuner_backend peft \
    --template_type AUTO \
    --dtype fp16 \
    --output_dir output \
    --ddp_backend nccl \
    --custom_train_dataset_path /home/hanlv/workspace/code/research/infodemic/LLM/LoRA/Qwen/data.jsonl \
    --train_dataset_sample -1 \
    --num_train_epochs 1 \
    --max_length 2048 \
    --check_dataset_strategy warning \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout_p 0.05 \
    --lora_target_modules ALL \
    --gradient_checkpointing false \
    --batch_size 1 \
    --weight_decay 0.1 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
    --max_grad_norm 0.5 \
    --warmup_ratio 0.03 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --use_flash_attn false \
    --deepspeed default-zero2 \
