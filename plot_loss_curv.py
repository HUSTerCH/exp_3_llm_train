import re
from pathlib import Path

import matplotlib.pyplot as plt


def parse_log(log_file: Path):
    steps = []
    losses = []
    with open(log_file, 'r', encoding="utf-8") as f:
        lines = f.readlines()
    current_epoch = 1
    for line in lines:
        match = re.search(r'Epoch:\[(\d+)/(\d+)\]\((\d+)/(\d+)\),\s+loss:\s+(\d+\.\d+)', line)
        if match:
            # current_epoch = int(match.group(1))
            total_epoch = int(match.group(2))
            current_step = int(match.group(3))
            total_steps = int(match.group(4))
            loss = float(match.group(5))
            if len(steps) > 0 and current_step == steps[0]:
                current_epoch += 1
            steps.append((current_epoch - 1) * total_steps + current_step)
            losses.append(loss)
    return steps, losses


def plot_loss_curve(steps, losses, title: str):
    """绘制损失曲线"""
    plt.figure(figsize=(12, 6))
    plt.plot(steps, losses, linewidth=0.8, color='blue', alpha=0.7)
    plt.xlabel('Training Step (line index)')
    plt.ylabel('Loss')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)
    # 可选：纵坐标对数变换，便于观察后期下降
    # plt.yscale('log')
    plt.tight_layout()
    plt.savefig(f'loss_curve_{title}.png', dpi=150)
    # plt.show()


pretrain_steps, pretrain_losses = parse_log(Path("trainer/train_pretrain.log"))
plot_loss_curve(pretrain_steps, pretrain_losses, "Pretrain Loss Curve")

sft_steps, sft_losses = parse_log(Path("trainer/train_sft.log"))
plot_loss_curve(sft_steps, sft_losses, "SFT Loss Curve")

lora_steps,lora_losses = parse_log(Path("trainer/train_hust_lora.log"))
plot_loss_curve(lora_steps, lora_losses, "LORA Loss Curve(Epochs=6)")

lora_steps,lora_losses = parse_log(Path("trainer/train_hust_lora_ep_10.log"))
plot_loss_curve(lora_steps, lora_losses, "LORA Loss Curve(Epochs=10)")
