#!/bin/bash

# --- 第一次运行：跑第 1 个 Epoch ---
python train_full_sft.py \
    --save_dir "../out" \
    --save_weight "minimind_sft_e1" \
    --data_path "/hy-tmp/data/sft_t2t_mini.jsonl" \
    --from_weight "minimind_pretrain" \
    --epochs 1 \
    --batch_size 128 \
    --accumulation_steps 4 \
    --learning_rate 2e-5 \
    --max_seq_len 512 \
    --num_hidden_layers 8 \
    --use_compile 1 \
    --log_interval 100 \
    --save_interval 10000


# --- 第二次运行：基于 e1 跑第 2 个 Epoch ---
# 注意：修改 from_weight 指向 e1 的权重，并将保存名改为 e2
python train_full_sft.py \
    --save_dir "../out" \
    --save_weight "minimind_sft_e2" \
    --data_path "/hy-tmp/data/sft_t2t_mini.jsonl" \
    --from_weight "minimind_sft_e1" \
    --epochs 1 \
    --batch_size 128 \
    --accumulation_steps 4 \
    --learning_rate 1e-5 \
    --max_seq_len 512 \
    --num_hidden_layers 8 \
    --use_compile 1 \
    --log_interval 100 \
    --save_interval 10000