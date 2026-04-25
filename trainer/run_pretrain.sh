#!/bin/bash

# 基础配置
SAVE_DIR="../out"
DATA_PATH="/hy-tmp/data/pretrain_t2t_mini.jsonl"
SAVE_WEIGHT="minimind_pretrain"

BATCH_SIZE=128            # 单步 Batch Size
GRAD_ACC=4                # 梯度累积，等效总 Batch Size = 512
LEARNING_RATE=5e-4
MAX_SEQ_LEN=512
EPOCHS=3

# 设备与加速配置
DEVICE="cuda:0"
DTYPE="bfloat16"
NUM_WORKERS=8
USE_COMPILE=1

python train_pretrain.py \
    --save_dir $SAVE_DIR \
    --save_weight $SAVE_WEIGHT \
    --data_path $DATA_PATH \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --accumulation_steps $GRAD_ACC \
    --learning_rate $LEARNING_RATE \
    --max_seq_len $MAX_SEQ_LEN \
    --device $DEVICE \
    --dtype $DTYPE \
    --num_workers $NUM_WORKERS \
    --use_compile $USE_COMPILE \
    --log_interval 50 \
    --save_interval 1000 \
    --num_hidden_layers 8
