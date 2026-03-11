#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MotionCanvas 推理脚本
使用训练好的 Checkpoint 生成视频
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diffsynth import ModelManager, save_video
from diffsynth.pipelines.wan_video_motioncanvas import WanVideoPipeline_motioncanvas

DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, "
    "JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
    "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, messy background, three legs, many people in the background, "
    "walking backwards"
)


def load_checkpoint_weights(pipe, checkpoint_path, device="cpu"):
    """加载训练好的 checkpoint 到 pipeline"""
    print(f">>> 加载 checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif "module" in ckpt:
        state_dict = {k: v for k, v in ckpt["module"].items()}
    else:
        state_dict = ckpt

    del ckpt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    dit_state_dict = {}
    bbox_zeroconv_state_dict = {}

    for k, v in state_dict.items():
        if k.startswith("pipe."):
            k = k[5:]

        if k.startswith("dit.") or k.startswith("denoising_model."):
            new_k = k.replace("denoising_model.", "").replace("dit.", "")
            dit_state_dict[new_k] = v
        elif k.startswith("bbox_zeroconv."):
            new_k = k.replace("bbox_zeroconv.", "")
            bbox_zeroconv_state_dict[new_k] = v

    if dit_state_dict:
        missing, unexpected = pipe.dit.load_state_dict(dit_state_dict, strict=False)
        print(f">>> DiT: 加载了 {len(dit_state_dict)} 个参数, missing={len(missing)}, unexpected={len(unexpected)}")

    if bbox_zeroconv_state_dict:
        pipe.bbox_zeroconv.load_state_dict(bbox_zeroconv_state_dict, strict=True)
        print(f">>> bbox_zeroconv: 加载了 {len(bbox_zeroconv_state_dict)} 个参数")

    return pipe


def build_bbox_mask_from_json(bbox_json_path, num_frames, height, width):
    """
    从 JSON 文件构建 bbox_mask 张量。
    JSON 格式示例:
    {
      "objects": [
        {
          "frames": {
            "0":  [x1, y1, x2, y2],
            "12": [x1, y1, x2, y2],
            ...
          }
        }
      ]
    }
    坐标为归一化 [0,1] 或像素值。
    返回 shape: (1, 3, num_frames, height, width)
    """
    with open(bbox_json_path, "r") as f:
        bbox_data = json.load(f)

    mask = torch.zeros(1, 3, num_frames, height, width)
    objects = bbox_data.get("objects", [])

    for obj in objects:
        frames_dict = obj.get("frames", {})
        for frame_idx_str, bbox in frames_dict.items():
            fi = int(frame_idx_str)
            if fi >= num_frames:
                continue
            x1, y1, x2, y2 = bbox
            if all(0 <= v <= 1.0 for v in [x1, y1, x2, y2]):
                x1, x2 = int(x1 * width), int(x2 * width)
                y1, y2 = int(y1 * height), int(y2 * height)
            else:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            x1, x2 = max(0, x1), min(width, x2)
            y1, y2 = max(0, y1), min(height, y2)
            mask[:, :, fi, y1:y2, x1:x2] = 1.0

    mask = mask * 2.0 - 1.0
    return mask


def main():
    parser = argparse.ArgumentParser(description="MotionCanvas 推理")

    # ---- 模型路径 ----
    parser.add_argument("--dit_path", type=str, required=True, help="DiT 模型路径")
    parser.add_argument("--vae_path", type=str, required=True, help="VAE 模型路径")
    parser.add_argument("--text_encoder_path", type=str, required=True, help="文本编码器路径")
    parser.add_argument("--image_encoder_path", type=str, default=None, help="图像编码器路径 (I2V 需要)")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="MotionCanvas checkpoint 路径")

    # ---- 输入 ----
    parser.add_argument("--input_image", type=str, default=None, help="输入图像路径 (I2V 模式)")
    parser.add_argument("--end_image", type=str, default=None, help="结束帧图像路径 (I2V 首尾帧模式)")
    parser.add_argument("--prompt", type=str, required=True, help="文本提示")
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT, help="负面提示")
    parser.add_argument("--output", type=str, default="output.mp4", help="输出视频路径")

    # ---- 生成参数 ----
    parser.add_argument("--height", type=int, default=480, help="视频高度")
    parser.add_argument("--width", type=int, default=832, help="视频宽度")
    parser.add_argument("--num_frames", type=int, default=49, help="帧数")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="去噪步数")
    parser.add_argument("--cfg_scale", type=float, default=5.0, help="CFG scale")
    parser.add_argument("--sigma_shift", type=float, default=5.0, help="Flow matching sigma shift")
    parser.add_argument("--denoising_strength", type=float, default=1.0, help="去噪强度 (配合 input_video 使用)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--fps", type=int, default=15, help="输出帧率")

    # ---- Tiling (VAE 显存优化) ----
    parser.add_argument("--tiled", action="store_true", default=True, help="启用 VAE tiled 编解码")
    parser.add_argument("--tile_size", type=int, nargs=2, default=[30, 52], help="tile size (h, w)")
    parser.add_argument("--tile_stride", type=int, nargs=2, default=[15, 26], help="tile stride (h, w)")

    # ---- MotionCanvas 参数 ----
    parser.add_argument("--bbox_mask_path", type=str, default=None,
                        help="预计算的 bbox_mask 张量 (.pt) 或 bbox JSON 文件路径")
    parser.add_argument("--track_video_path", type=str, default=None,
                        help="预计算的 track_video 张量 (.pt) 路径")

    # ---- TeaCache 加速 ----
    parser.add_argument("--tea_cache_l1_thresh", type=float, default=None, help="TeaCache L1 阈值")
    parser.add_argument("--tea_cache_model_id", type=str, default="", help="TeaCache 模型 ID")

    # ---- 设备 ----
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"], help="数据类型")

    args = parser.parse_args()

    torch_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    # ---- 加载基础模型 ----
    print(">>> 加载基础模型...")
    model_paths = [args.text_encoder_path, args.vae_path, args.dit_path]
    if args.image_encoder_path:
        model_paths.append(args.image_encoder_path)

    model_manager = ModelManager(torch_dtype=torch_dtype, device="cpu")
    model_manager.load_models(model_paths)

    # ---- 创建 pipeline ----
    print(">>> 创建 MotionCanvas Pipeline...")
    pipe = WanVideoPipeline_motioncanvas.from_model_manager(
        model_manager, torch_dtype=torch_dtype, device=args.device
    )

    # ---- 加载 MotionCanvas checkpoint ----
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        pipe = load_checkpoint_weights(pipe, args.checkpoint_path, device="cpu")
        pipe.bbox_zeroconv = pipe.bbox_zeroconv.to(dtype=torch_dtype, device=args.device)

    # ---- 启用显存管理 ----
    pipe.enable_vram_management(num_persistent_param_in_dit=None)

    # ---- 加载输入图像 (I2V) ----
    input_image = None
    if args.input_image and os.path.exists(args.input_image):
        input_image = Image.open(args.input_image).convert("RGB")
        print(f">>> 输入图像: {args.input_image}")

    end_image = None
    if args.end_image and os.path.exists(args.end_image):
        end_image = Image.open(args.end_image).convert("RGB")
        print(f">>> 结束帧图像: {args.end_image}")

    # ---- 加载 MotionCanvas 控制信号 ----
    bbox_mask = None
    if args.bbox_mask_path and os.path.exists(args.bbox_mask_path):
        if args.bbox_mask_path.endswith(".pt"):
            bbox_mask = torch.load(args.bbox_mask_path, map_location="cpu")
            print(f">>> bbox_mask 张量已加载: {bbox_mask.shape}")
        elif args.bbox_mask_path.endswith(".json"):
            bbox_mask = build_bbox_mask_from_json(
                args.bbox_mask_path, args.num_frames, args.height, args.width
            )
            print(f">>> bbox_mask 从 JSON 构建: {bbox_mask.shape}")
        bbox_mask = bbox_mask.to(dtype=torch_dtype, device=args.device)

    track_video = None
    if args.track_video_path and os.path.exists(args.track_video_path):
        track_video = torch.load(args.track_video_path, map_location="cpu")
        track_video = track_video.to(dtype=torch_dtype, device=args.device)
        print(f">>> track_video 张量已加载: {track_video.shape}")

    # ---- 生成视频 ----
    print(f">>> 开始生成视频: {args.width}x{args.height}, {args.num_frames} 帧, "
          f"{args.num_inference_steps} 步, cfg_scale={args.cfg_scale}, seed={args.seed}")

    video_frames = pipe(
        prompt=[args.prompt],
        negative_prompt=args.negative_prompt,
        input_image=input_image,
        end_image=end_image,
        num_inference_steps=args.num_inference_steps,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        cfg_scale=args.cfg_scale,
        sigma_shift=args.sigma_shift,
        denoising_strength=args.denoising_strength,
        seed=args.seed,
        tiled=args.tiled,
        tile_size=tuple(args.tile_size),
        tile_stride=tuple(args.tile_stride),
        bbox_mask=bbox_mask,
        track_video=track_video,
        tea_cache_l1_thresh=args.tea_cache_l1_thresh,
        tea_cache_model_id=args.tea_cache_model_id,
    )

    # ---- 保存视频 ----
    if video_frames and len(video_frames) > 0:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        save_video(video_frames[0], args.output, fps=args.fps, quality=5)
        print(f">>> 视频已保存: {args.output}")
    else:
        print(">>> 生成失败，没有输出帧")


if __name__ == "__main__":
    main()
