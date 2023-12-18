# Experimental environment: A10
# If you want to merge LoRA weight and save it, you need to set `--merge_lora_and_save true`.
PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0 \
python llm_infer.py \
    --ckpt_dir output/orca-2-7b/without_info/"2023-12-16 12:08:51 split_type=8:2 train_ratio=0.1"/checkpoint-60 \
    --load_args_from_ckpt_dir true \
    --eval_human false \
    --max_length 4096 \
    --use_flash_attn false \
    --max_new_tokens 4096 \
    --do_sample true \
    --merge_lora_and_save false

