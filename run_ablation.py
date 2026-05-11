#!/usr/bin/env python3
"""
MotionCanvas 批量消融实验脚本。

支持两种模式：
  1) Direct 模式：从配置文件直接读取 bbox/camera/point JSON，无需 LLM。
  2) LLM 模式：为每个实验调用 LLM API 自动生成运动参数。

用法：
  python run_ablation.py --config experiments.json --output_dir ./ablations
  python run_ablation.py --config experiments.yaml --output_dir ./ablations --skip_llm

配置文件格式（JSON 或 YAML）：
  {
    "model": {
      "dit_path": "...",
      "vae_path": "...",
      "text_encoder_path": "...",
      "image_encoder_path": "...",
      "checkpoint_path": "..."
    },
    "defaults": {
      "height": 480, "width": 832, "num_frames": 49,
      "num_inference_steps": 50, "cfg_scale": 5.0,
      "sigma_shift": 5.0, "seed": 42, "fps": 15,
      "negative_prompt": "..."
    },
    "llm": {  // 可选，LLM 模式配置
      "base_url": "https://api.siliconflow.cn/v1",
      "api_key": "sk-xxx",
      "model": "Pro/moonshotai/Kimi-K2.5"
    },
    "experiments": [
      {
        "name": "baseline",
        "image": "images/cat.png",
        "prompt": "a cat walking",
        // LLM 模式：填写 llm_instruction 让 LLM 生成参数
        "llm_instruction": "the cat walks from left to right",
        // Direct 模式：直接填写 bbox/camera/point JSON（优先级高于 LLM）
        "bbox_json": '{"objects": [{"frames": {"0": [0.2,0.3,0.5,0.7], "24": [0.6,0.3,0.9,0.7]}}]}',
        "camera_json": '{"camera": {"keyframes": [{"frame": 0, "zoom": 1.0, "pan": [0,0], "rotation": 0}]}}',
        "point_json": "",
        // 可选单实验覆盖
        "seed": 123,
        "cfg_scale": 7.0
      }
    ]
  }
"""

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diffsynth import ModelManager, save_video
from diffsynth.pipelines.wan_video_motioncanvas import WanVideoPipeline_motioncanvas

# 从 motioncanvas.py 复用关键函数
from apps.gradio.motioncanvas import (
    build_bbox_mask_from_json_str,
    _build_fallback_track_video,
)

DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, "
    "JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
    "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, messy background, three legs, many people in the background, "
    "walking backwards"
)

LLM_MOTION_PROMPT = """You are a motion parameter generator for MotionCanvas, a video generation system.

Given an input image description and a user instruction about desired motion, generate the appropriate bounding box (bbox) keyframes and camera keyframes in JSON format.

**Output format**: Return ONLY a valid JSON object with NO markdown, NO code fences, NO extra text:
{
  "bbox_json": "{\\"objects\\": [{\\"frames\\": {\\"0\\": [x1,y1,x2,y2], \\"24\\": [x1,y1,x2,y2]}}]}",
  "camera_json": "{\\"camera\\": {\\"keyframes\\": [{\\"frame\\": 0, \\"zoom\\": 1.0, \\"pan\\": [0,0], \\"rotation\\": 0}]}}",
  "point_json": ""
}

**Rules**:
- bbox coordinates are normalized [0,1] relative to image dimensions: [x1, y1, x2, y2]
- The video is {num_frames} frames long (0-indexed).
- For camera: zoom=1.0 means no zoom, pan is in pixels (positive=right/down), rotation is in degrees.
- If there is no camera motion, set camera_json with default values (zoom=1.0, pan=[0,0], rotation=0).
- Always output at least one bbox covering the main subject.
- If the subject moves, create keyframes at the start and end frames to define the trajectory.

**Example for "a cat walks from left to right"**:
{{
  "bbox_json": "{{\\"objects\\": [{{\\"frames\\": {{\\"0\\": [0.1, 0.3, 0.4, 0.8], \\"24\\": [0.6, 0.3, 0.9, 0.8]}}}}]}}",
  "camera_json": "{{\\"camera\\": {{\\"keyframes\\": [{{\\"frame\\": 0, \\"zoom\\": 1.0, \\"pan\\": [0, 0], \\"rotation\\": 0}}]}}}}",
  "point_json": ""
}}

**Example for "zoom in on the face"**:
{{
  "bbox_json": "{{\\"objects\\": [{{\\"frames\\": {{\\"0\\": [0.2, 0.1, 0.8, 0.6], \\"24\\": [0.3, 0.2, 0.7, 0.5]}}}}]}}",
  "camera_json": "{{\\"camera\\": {{\\"keyframes\\": [{{\\"frame\\": 0, \\"zoom\\": 1.0, \\"pan\\": [0, 0], \\"rotation\\": 0}}, {{\\"frame\\": 24, \\"zoom\\": 1.5, \\"pan\\": [0, 0], \\"rotation\\": 0}}]}}}}",
  "point_json": ""
}}

Now generate parameters for this request: {instruction}
"""


def load_config(path):
    raw = Path(path).read_text()
    if path.endswith((".yaml", ".yml")):
        import yaml
        return yaml.safe_load(raw)
    return json.loads(raw)


def _load_checkpoint_weights(pipe, checkpoint_path, device="cpu"):
    print(f"  Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt.get("module", ckpt))
    del ckpt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    dit_sd = {}
    bbox_sd = {}
    for k, v in state_dict.items():
        k = k[5:] if k.startswith("pipe.") else k
        if k.startswith("dit.") or k.startswith("denoising_model."):
            dit_sd[k.replace("denoising_model.", "").replace("dit.", "")] = v
        elif k.startswith("bbox_zeroconv."):
            bbox_sd[k.replace("bbox_zeroconv.", "")] = v

    if dit_sd:
        missing, unexpected = pipe.dit.load_state_dict(dit_sd, strict=False)
        print(f"  DiT: loaded {len(dit_sd)} params, missing={len(missing)}, unexpected={len(unexpected)}")
    if bbox_sd:
        pipe.bbox_zeroconv.load_state_dict(bbox_sd, strict=True)
        print(f"  bbox_zeroconv: loaded {len(bbox_sd)} params")
    return pipe


def build_pipeline(cfg):
    model_cfg = cfg["model"]
    torch_dtype = torch.bfloat16
    device = "cuda"

    model_paths = [
        model_cfg["text_encoder_path"],
        model_cfg["vae_path"],
        model_cfg["dit_path"],
    ]
    if model_cfg.get("image_encoder_path"):
        model_paths.append(model_cfg["image_encoder_path"])

    print("Loading model manager...")
    model_manager = ModelManager(torch_dtype=torch_dtype, device="cpu")
    model_manager.load_models(model_paths)

    print("Creating pipeline...")
    pipe = WanVideoPipeline_motioncanvas.from_model_manager(
        model_manager, torch_dtype=torch_dtype, device=device
    )

    ckpt = model_cfg.get("checkpoint_path")
    if ckpt and os.path.exists(ckpt):
        pipe = _load_checkpoint_weights(pipe, ckpt, device="cpu")
        pipe.bbox_zeroconv = pipe.bbox_zeroconv.to(dtype=torch_dtype, device=device)

    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    return pipe


def call_llm_for_motion(instruction, num_frames, llm_cfg):
    import requests

    system_prompt = LLM_MOTION_PROMPT.format(
        num_frames=num_frames, instruction=instruction
    )

    resp = requests.post(
        f"{llm_cfg['base_url'].rstrip('/')}/chat/completions",
        json={
            "model": llm_cfg["model"],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": instruction},
            ],
            "temperature": 0.1,
            "max_tokens": 2048,
        },
        headers={
            "Authorization": f"Bearer {llm_cfg['api_key']}",
            "Content-Type": "application/json",
        },
        timeout=llm_cfg.get("timeout", 120),
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]

    for wrap in ["```json", "```JSON", "```"]:
        if wrap in content:
            content = content.split(wrap, 1)[1]
            content = content.rsplit("```", 1)[0]
            break
    content = content.strip()
    return json.loads(content)


def resolve_experiment_params(exp, defaults, llm_cfg, skip_llm):
    params = copy.deepcopy(defaults)
    params.update({k: exp[k] for k in ("prompt",) if k in exp})
    for key in ["height", "width", "num_frames", "num_inference_steps",
                 "cfg_scale", "sigma_shift", "seed", "fps", "negative_prompt"]:
        if key in exp:
            params[key] = exp[key]

    has_direct = any(exp.get(k) for k in ["bbox_json", "camera_json", "point_json"])
    has_llm = bool(exp.get("llm_instruction")) and llm_cfg and not skip_llm

    if has_direct:
        params["bbox_json"] = exp.get("bbox_json", "")
        params["camera_json"] = exp.get("camera_json", "")
        params["point_json"] = exp.get("point_json", "")
    elif has_llm:
        print(f"  Calling LLM for motion params...")
        result = call_llm_for_motion(
            exp["llm_instruction"], params["num_frames"], llm_cfg
        )
        params["bbox_json"] = result.get("bbox_json", "")
        params["camera_json"] = result.get("camera_json", "")
        params["point_json"] = result.get("point_json", "")
    else:
        params["bbox_json"] = ""
        params["camera_json"] = ""
        params["point_json"] = ""

    return params


def run_experiment(pipe, exp, defaults, llm_cfg, output_dir, skip_llm):
    name = exp["name"]
    exp_dir = Path(output_dir) / name
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    print(f"{'='*60}")

    params = resolve_experiment_params(exp, defaults, llm_cfg, skip_llm)

    with open(exp_dir / "config.json", "w") as f:
        json.dump({"experiment": exp, "resolved_params": {
            k: v for k, v in params.items() if k != "negative_prompt"
        }}, f, indent=2, ensure_ascii=False)

    image_path = exp["image"]
    if not os.path.exists(image_path):
        print(f"  WARNING: image not found: {image_path}, skipping")
        return
    input_image = Image.open(image_path).convert("RGB")
    input_image.save(str(exp_dir / "input.png"))

    pipe_kwargs = {
        "prompt": [params["prompt"]],
        "negative_prompt": params.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT),
        "input_image": input_image,
        "end_image": None,
        "num_inference_steps": int(params["num_inference_steps"]),
        "height": int(params["height"]),
        "width": int(params["width"]),
        "num_frames": int(params["num_frames"]),
        "cfg_scale": params["cfg_scale"],
        "sigma_shift": params["sigma_shift"],
        "seed": int(params["seed"]),
        "tiled": True,
        "tile_size": (30, 52),
        "tile_stride": (15, 26),
    }

    bbox_mask = None
    if params.get("bbox_json"):
        print(f"  Building bbox_mask...")
        bbox_mask = build_bbox_mask_from_json_str(
            params["bbox_json"],
            int(params["num_frames"]),
            int(params["height"]),
            int(params["width"]),
        )
        bbox_mask = bbox_mask.to(
            dtype=pipe_state_dtype(pipe), device=pipe.device
        )
        torch.save(bbox_mask.cpu(), str(exp_dir / "bbox_mask.pt"))

    track_video = None
    has_any_motion = any(params.get(k) for k in ["bbox_json", "camera_json", "point_json"])
    if has_any_motion:
        print(f"  Building track_video...")
        track_video = _build_fallback_track_video(
            params.get("bbox_json", ""),
            params.get("camera_json", ""),
            params.get("point_json", ""),
            int(params["num_frames"]),
            int(params["height"]),
            int(params["width"]),
            pipe_state_dtype(pipe),
            pipe.device,
        )
        if track_video is not None:
            torch.save(track_video.cpu(), str(exp_dir / "track_video.pt"))

    pipe_kwargs["bbox_mask"] = bbox_mask
    pipe_kwargs["track_video"] = track_video

    print(f"  Generating video ({params['width']}x{params['height']}, "
          f"{params['num_frames']} frames, seed={params['seed']})...")
    t0 = time.time()
    video_frames = pipe(**pipe_kwargs)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    if video_frames and len(video_frames) > 0:
        output_path = str(exp_dir / "output.mp4")
        save_video(video_frames[0], output_path, fps=int(params["fps"]), quality=5)
        print(f"  Saved: {output_path}")
    else:
        print(f"  FAILED: no frames generated")

    return exp_dir


def pipe_state_dtype(pipe):
    return next(pipe.parameters()).dtype


def main():
    parser = argparse.ArgumentParser(
        description="MotionCanvas 批量消融实验"
    )
    parser.add_argument("--config", required=True, help="实验配置文件 (.json / .yaml)")
    parser.add_argument("--output_dir", default="./ablations", help="输出目录")
    parser.add_argument("--skip_llm", action="store_true", help="跳过 LLM 调用，仅使用直接参数")
    parser.add_argument("--resume", help="从指定实验名开始（跳过前面的）")
    args = parser.parse_args()

    cfg = load_config(args.config)
    defaults = cfg.get("defaults", {})
    defaults.setdefault("negative_prompt", DEFAULT_NEGATIVE_PROMPT)
    llm_cfg = cfg.get("llm")

    pipe = build_pipeline(cfg)

    experiments = cfg["experiments"]
    resumed = args.resume is None

    for exp in experiments:
        if not resumed:
            if exp["name"] == args.resume:
                resumed = True
            else:
                print(f"Skipping {exp['name']} (resume from {args.resume})")
                continue

        run_experiment(pipe, exp, defaults, llm_cfg, args.output_dir, args.skip_llm)

    print(f"\nAll experiments done. Results in: {args.output_dir}")


if __name__ == "__main__":
    main()
