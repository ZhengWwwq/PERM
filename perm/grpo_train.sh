export WANDB_API_KEY=WANDB_API_KEY

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file deepspeed_zero3.yaml grpo.py \
    --train_data_path DATASET_PATH \
    --num_train_samples 3000 \
    --output_dir OUTPUT_DIR \
    --model_name PATH_TO_BASE_MODEL \
    --reward_model_type oai \
    --reward_generation_kwargs '{"model_name": "gpt-4o-mini", "api_key": "API_KEY", "base_url": "BASE_URL"}' \
    --calculate_score_mode harmonic \
    --learning_rate 1e-6 \
    --beta 0.01 \
    --epochs 1 \
    --peft_r 16 \
    --peft_alpha 32 \
    --peft_dropout 0.1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --warmup_ratio 0.1 \
    --save_steps 200 \
    --save_total_limit 5 \
    --max_completion_length 1024 \
    --use_length_penalty \
    --length_penalty_limit 768 \
    --length_penalty_coef 0.001 \
    --use_standby_reward \
    --standby_reward_coef 0.5