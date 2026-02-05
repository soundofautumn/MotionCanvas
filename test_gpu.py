#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试 GPU 和项目基本功能"""

import torch
import sys

print("=" * 50)
print("PyTorch GPU 测试")
print("=" * 50)
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"GPU 数量: {torch.cuda.device_count()}")
    print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
    print(f"当前设备: cuda:{torch.cuda.current_device()}")
    
    # 测试 GPU 计算
    print("\n测试 GPU 计算...")
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print(f"[OK] GPU 计算测试成功，结果张量设备: {z.device}")
else:
    print("[ERROR] CUDA 不可用，请检查 GPU 驱动和 PyTorch 安装")

print("\n" + "=" * 50)
print("项目导入测试")
print("=" * 50)
try:
    from diffsynth import ModelManager
    print("[OK] diffsynth 模块导入成功")
except Exception as e:
    print(f"[ERROR] diffsynth 模块导入失败: {e}")
    sys.exit(1)

try:
    from diffsynth import SDImagePipeline, SDXLImagePipeline
    print("[OK] Pipeline 类导入成功")
except Exception as e:
    print(f"[ERROR] Pipeline 类导入失败: {e}")

print("\n" + "=" * 50)
print("测试完成！")
print("=" * 50)

