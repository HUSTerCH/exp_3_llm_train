# import torch
# from torch import nn
# import torch.nn.functional as F

# class LoraCombinedLinear(nn.Module):
#     def __init__(self, original_linear, rank, alpha, device):
#         super().__init__()
#         self.original_linear = original_linear  # 保存原始 Linear 层
#         self.scaling = alpha / rank
        
#         # 定义 LoRA 的 A 和 B 矩阵
#         self.lora_A = nn.Linear(original_linear.in_features, rank, bias=False).to(device)
#         self.lora_B = nn.Linear(rank, original_linear.out_features, bias=False).to(device)
        
#         # 初始化
#         nn.init.normal_(self.lora_A.weight, std=0.02)
#         nn.init.zeros_(self.lora_B.weight)
        
#         # 冻结原始权重
#         self.original_linear.weight.requires_grad = False
#         if self.original_linear.bias is not None:
#             self.original_linear.bias.requires_grad = False

#     def forward(self, x):
#         # 原始路径 + LoRA 路径
#         return self.original_linear(x) + self.scaling * self.lora_B(self.lora_A(x))

# def apply_lora(model, rank=64, alpha=128):
#     # 1. 确定需要替换的模块路径
#     target_names = []
#     for name, module in model.named_modules():
#         if isinstance(module, nn.Linear) and "lm_head" not in name:
#             target_names.append(name)
    
#     # 2. 逐个执行替换
#     for name in target_names:
#         # 获取父节点和目标属性名
#         parts = name.split('.')
#         target_attr = parts[-1]
#         parent = model
#         for part in parts[:-1]:
#             parent = getattr(parent, part)
        
#         # 获取原始层并创建包装层
#         original_module = getattr(parent, target_attr)
#         new_module = LoraCombinedLinear(original_module, rank, alpha, model.device)
        
#         # 替换！
#         setattr(parent, target_attr, new_module)

#     # 统计 LoRA 参数量
#     lora_params = sum(p.numel() for n, p in model.named_parameters() if 'lora_' in n)
#     print(f"[*] 注入完成，LoRA 参数量: {lora_params/1e6:.3f}M")

# # 注意：为了适配保存和加载逻辑，save_lora 和 load_lora 也需要相应微调
# def save_lora(model, path):
#     state_dict = {}
#     for name, module in model.named_modules():
#         if isinstance(module, LoraCombinedLinear):
#             # 去掉 module. 前缀（如果有）
#             clean_name = name[7:] if name.startswith("module.") else name
#             state_dict[f"{clean_name}.lora_A.weight"] = module.lora_A.weight.data.cpu().half()
#             state_dict[f"{clean_name}.lora_B.weight"] = module.lora_B.weight.data.cpu().half()
#     torch.save(state_dict, path)

# def load_lora(model, path,adapter_weight=0.5):
#     st = torch.load(path, map_location=model.device)
#     for name, module in model.named_modules():
#         if isinstance(module, LoraCombinedLinear):
#             clean_name = name[7:] if name.startswith("module.") else name
#             module.lora_A.weight.data.copy_(st[f"{clean_name}.lora_A.weight"])
#             module.lora_B.weight.data.copy_(st[f"{clean_name}.lora_B.weight"])
#             # 同时更新该层的 scaling
#             module.scaling *= adapter_weight


import torch
from torch import nn
import torch.nn.functional as F

class LoraCombinedLinear(nn.Module):
    def __init__(self, original_linear, rank, alpha, device):
        super().__init__()
        self.original_linear = original_linear
        self.scaling = alpha / rank
        self.lora_A = nn.Linear(original_linear.in_features, rank, bias=False).to(device)
        self.lora_B = nn.Linear(rank, original_linear.out_features, bias=False).to(device)
        nn.init.normal_(self.lora_A.weight, std=0.02)
        nn.init.zeros_(self.lora_B.weight)
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False

    def forward(self, x):
        return self.original_linear(x) + self.scaling * self.lora_B(self.lora_A(x))

def apply_lora(model, rank=8, alpha=16):
    target_names = []
    # 这里的过滤逻辑已经包含了常见的 Attention 命名，并排除了 MLP
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            name_lower = name.lower()
            if any(key in name_lower for key in ['wq', 'wk', 'wv', 'wo', 'q_proj', 'k_proj', 'v_proj', 'o_proj']):
                if "mlp" not in name_lower and "lm_head" not in name_lower:
                    target_names.append(name)
    
    if not target_names:
        # 如果还是 0，打印所有层名自查
        for n, _ in model.named_modules(): print(f"层名自查: {n}")
        raise ValueError("未能识别到任何 Attention 线性层！")

    for name in target_names:
        parts = name.split('.')
        target_attr = parts[-1]
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        
        original_module = getattr(parent, target_attr)
        new_module = LoraCombinedLinear(original_module, rank, alpha, model.device)
        setattr(parent, target_attr, new_module)

    lora_params = sum(p.numel() for n, p in model.named_parameters() if 'lora_' in n)
    print(f"[*] 注入完成（仅Attention），注入层数: {len(target_names)}，LoRA 参数量: {lora_params/1e6:.3f}M")

def save_lora(model, path):
    state_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, LoraCombinedLinear):
            clean_name = name[7:] if name.startswith("module.") else name
            state_dict[f"{clean_name}.lora_A.weight"] = module.lora_A.weight.data.cpu().half()
            state_dict[f"{clean_name}.lora_B.weight"] = module.lora_B.weight.data.cpu().half()
    torch.save(state_dict, path)

def load_lora(model, path, adapter_weight=1.0):
    st = torch.load(path, map_location=model.device)
    for name, module in model.named_modules():
        if isinstance(module, LoraCombinedLinear):
            clean_name = name[7:] if name.startswith("module.") else name
            module.lora_A.weight.data.copy_(st[f"{clean_name}.lora_A.weight"])
            module.lora_B.weight.data.copy_(st[f"{clean_name}.lora_B.weight"])
            module.scaling *= adapter_weight