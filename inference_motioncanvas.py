#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MotionCanvas 推理脚本
使用训练好的 Checkpoint 生成视频
"""

import os
import sys
import argparse
import torch
from PIL import Image

# 确保 diffsynth 可导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diffsynth import ModelManager, save_video
from diffsynth.pipelines.wan_video_motioncanvas import WanVideoPipeline_motioncanvas


def load_checkpoint_weights(pipe, checkpoint_path, device="cpu"):
    """加载训练好的 checkpoint 到 pipeline"""
    print(f">>> 加载 checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    # # 提取 DiT 和 bbox_zeroconv 的 state_dict
    # dit_sd = {}
    # bbox_sd = {}
    # for k, v in ckpt.items():
    #     if k.startswith('module.'):          # 兼容 DeepSpeed 格式
    #         k = k[7:]
    #     if k.startswith('pipe.'):
    #         k = k[5:]
    #     if k.startswith('dit.') or k.startswith('denoising_model.'):
    #         new_k = k.replace('denoising_model.', '').replace('dit.', '')
    #         dit_sd[new_k] = v
    #     elif k.startswith('bbox_zeroconv.'):
    #         new_k = k.replace('bbox_zeroconv.', '')
    #         bbox_sd[new_k] = v
    #     # 其他键不需要，直接丢弃

    # # 释放原 checkpoint 内存
    # del ckpt
    # torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # # 加载到模型
    # if dit_sd:
    #     pipe.dit.load_state_dict(dit_sd, strict=False)
    # if bbox_sd:
    #     pipe.bbox_zeroconv.load_state_dict(bbox_sd, strict=True)
    # return pipe
    
    # 处理不同格式的 checkpoint
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif "module" in ckpt:
        # DeepSpeed 格式
        state_dict = {k: v for k, v in ckpt["module"].items()}
    else:
        state_dict = ckpt

    del ckpt
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # 加载 DiT 权重（MotionCanvas 主要训练的是 DiT）
    dit_state_dict = {}
    bbox_zeroconv_state_dict = {}
    
    for k, v in state_dict.items():
        # 去掉 "pipe." 前缀（如果有）
        if k.startswith("pipe."):
            k = k[5:]
        
        if k.startswith("dit.") or k.startswith("denoising_model."):
            new_k = k.replace("denoising_model.", "").replace("dit.", "")
            dit_state_dict[new_k] = v
        elif k.startswith("bbox_zeroconv."):
            new_k = k.replace("bbox_zeroconv.", "")
            bbox_zeroconv_state_dict[new_k] = v
    
    # 加载权重
    if dit_state_dict:
        missing, unexpected = pipe.dit.load_state_dict(dit_state_dict, strict=False)
        print(f">>> DiT: 加载了 {len(dit_state_dict)} 个参数, missing={len(missing)}, unexpected={len(unexpected)}")
    
    if bbox_zeroconv_state_dict:
        pipe.bbox_zeroconv.load_state_dict(bbox_zeroconv_state_dict, strict=True)
        print(f">>> bbox_zeroconv: 加载了 {len(bbox_zeroconv_state_dict)} 个参数")
    
    return pipe


def main():
    parser = argparse.ArgumentParser(description="MotionCanvas 推理")
    parser.add_argument("--dit_path", type=str, required=True, help="DiT 模型路径")
    parser.add_argument("--vae_path", type=str, required=True, help="VAE 模型路径")
    parser.add_argument("--text_encoder_path", type=str, required=True, help="文本编码器路径")
    parser.add_argument("--image_encoder_path", type=str, default=None, help="图像编码器路径 (I2V 需要)")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="MotionCanvas checkpoint 路径")
    
    parser.add_argument("--input_image", type=str, default=None, help="输入图像路径 (I2V 模式)")
    parser.add_argument("--prompt", type=str, required=True, help="文本提示")
    parser.add_argument("--negative_prompt", type=str, default="", help="负面提示")
    parser.add_argument("--output", type=str, default="output.mp4", help="输出视频路径")
    
    parser.add_argument("--height", type=int, default=480, help="视频高度")
    parser.add_argument("--width", type=int, default=832, help="视频宽度")
    parser.add_argument("--num_frames", type=int, default=49, help="帧数")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="去噪步数")
    parser.add_argument("--cfg_scale", type=float, default=5.0, help="CFG scale")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--fps", type=int, default=15, help="输出帧率")
    
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"], help="数据类型")
    
    args = parser.parse_args()
    
    torch_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    
    # 加载基础模型
    print(">>> 加载基础模型...")
    model_paths = [args.text_encoder_path, args.vae_path, args.dit_path]
    if args.image_encoder_path:
        model_paths.append(args.image_encoder_path)
    
    model_manager = ModelManager(torch_dtype=torch_dtype, device="cpu")
    model_manager.load_models(model_paths)
    
    # 创建 pipeline
    print(">>> 创建 MotionCanvas Pipeline...")
    pipe = WanVideoPipeline_motioncanvas.from_model_manager(model_manager, torch_dtype=torch_dtype, device=args.device)
    
    # 加载 MotionCanvas checkpoint（如果提供）
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        pipe = load_checkpoint_weights(pipe, args.checkpoint_path, device="cpu")
        pipe.bbox_zeroconv = pipe.bbox_zeroconv.to(dtype=torch_dtype, device=args.device)
    
    # 启用显存管理
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    
    # 加载输入图像（I2V 模式）
    input_image = None
    if args.input_image and os.path.exists(args.input_image):
        input_image = Image.open(args.input_image).convert("RGB")
        print(f">>> 输入图像: {args.input_image}")
    
    # 生成视频
    print(f">>> 开始生成视频: {args.width}x{args.height}, {args.num_frames} 帧, {args.num_inference_steps} 步...")
    
    video_frames = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        input_image=input_image,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
        tiled=True,
    )
    
    # 保存视频
    if video_frames and len(video_frames) > 0:
        save_video(video_frames[0], args.output, fps=args.fps, quality=5)
        print(f">>> 视频已保存: {args.output}")
    else:
        print(">>> 生成失败，没有输出帧")


if __name__ == "__main__":
    main()
