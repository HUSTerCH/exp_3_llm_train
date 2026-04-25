#!/bin/bash

SAVE_DIR="../out"
FROM_WEIGHT="minimind_sft_e2"
DATA_PATH="/hy-tmp/data/hust_handbook_lora.jsonl"
LORA_NAME="minimind_hust_lora"

BATCH_SIZE=64
ACCUMULATION=1              # 无需累积，实际 batch = 64
EPOCHS=6
LEARNING_RATE=2e-4
MAX_SEQ_LEN=512

python train_lora.py \
    --save_dir $SAVE_DIR \
    --lora_name $LORA_NAME \
    --data_path $DATA_PATH \
    --from_weight $FROM_WEIGHT \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --accumulation_steps $ACCUMULATION \
    --learning_rate $LEARNING_RATE \
    --max_seq_len $MAX_SEQ_LEN \
    --num_hidden_layers 8 \
    --device "cuda:0" \
    --dtype "bfloat16" \
    --num_workers 8 \
    --use_compile 0 \
    --log_interval 10 \
    --save_interval 100 \
    --lora_rank 8 \
    --lora_alpha 16