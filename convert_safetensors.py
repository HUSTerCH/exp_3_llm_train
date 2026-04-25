import torch
from safetensors.torch import save_file
import os

def convert_pth_to_safetensors(pth_path, out_path):
    # 1. 加载 pth 权重
    # 如果你在本地 Mac 运行，map_location='cpu' 是必须的
    print(f"[*] 正在加载权重: {pth_path}")
    state_dict = torch.load(pth_path, map_location='cpu')

    # 2. 如果权重中包含 optimizer 或 scaler 状态（针对 checkpoint 文件）
    # 我们只需要提取模型本身的 state_dict
    if 'model' in state_dict:
        state_dict = state_dict['model']

    # 3. 规范化键名 (可选)
    # 某些分布式训练权重会带 'module.' 前缀，safetensors 通常需要清洗它
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("_orig_mod.", "") # 移除 torch.compile 带来的前缀
        name = name.replace("module.", "")  # 移除 DDP 带来的前缀
        new_state_dict[name] = v

    # 4. 执行转换保存
    print(f"[*] 正在写入 safetensors 格式: {out_path}")
    save_file(new_state_dict, out_path)
    print("[+] 转换完成！")

if __name__ == "__main__":
    # 根据你的文件名修改路径
    pth_file = "/root/minimind/out/my_minimind_sft_test/minimind_sft_e1_512.pth"
    output_file = "/root/minimind/out/my_minimind_sft_test/minimind_sft_e1_512.safetensors"
    
    if os.path.exists(pth_file):
        convert_pth_to_safetensors(pth_file, output_file)
    else:
        print(f"[!] 找不到文件: {pth_file}")