#!/usr/bin/env python3
"""
MotionCanvas 批量消融实验脚本。

基于 apps/gradio/llm_assistant.py 的真实 LLM 调参逻辑。
支持两种模式：
  1) Direct 模式：从配置文件直接读取 bbox/camera/point JSON，无需 LLM。
  2) LLM 模式：调用 llm_apply_instruction（完整 tool-calling + GDINO/SAM 定位）。

用法：
  python run_ablation.py --config ablation_config.yaml --output_dir ./ablations
  python run_ablation.py --config ablation_config.yaml --output_dir ./ablations --skip_llm

配置文件格式见 examples/ablation_example.yaml。
"""

import argparse
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


def load_config(path):
    raw = Path(path).read_text()
    if path.endswith((".yaml", ".yml")):
        import yaml
        return yaml.safe_load(raw)
    return json.loads(raw)


def load_checkpoint_weights(pipe, checkpoint_path, device="cpu"):
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
        pipe = load_checkpoint_weights(pipe, ckpt, device="cpu")
        pipe.bbox_zeroconv = pipe.bbox_zeroconv.to(dtype=torch_dtype, device=device)

    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    return pipe


def pipe_state_dtype(pipe):
    return next(pipe.parameters()).dtype


def call_llm_apply_instruction(exp, defaults, llm_cfg, input_image):
    """
    调用 llm_assistant.py 的 llm_apply_instruction 获取运动参数。

    返回 (bbox_json, camera_json, point_json, prompt, status_msg)。
    出错时 bbox_json="" 并返回错误信息。
    """
    from apps.gradio.llm_assistant import llm_apply_instruction

    gen_defaults = {
        "height": int(defaults.get("height", 480)),
        "width": int(defaults.get("width", 832)),
        "num_frames": int(defaults.get("num_frames", 49)),
        "fps": int(defaults.get("fps", 15)),
        "num_inference_steps": int(defaults.get("num_inference_steps", 50)),
        "cfg_scale": float(defaults.get("cfg_scale", 5.0)),
        "sigma_shift": float(defaults.get("sigma_shift", 5.0)),
        "seed": int(defaults.get("seed", 42)),
    }

    try:
        result = llm_apply_instruction(
            user_message=exp["llm_instruction"],
            chat_history=[],
            llm_base_url=llm_cfg["base_url"],
            llm_api_key=llm_cfg["api_key"],
            llm_model=llm_cfg["model"],
            llm_timeout=llm_cfg.get("timeout", 120),
            gdino_model_dir=llm_cfg.get("gdino_model_dir", ""),
            sam_ckpt=llm_cfg.get("sam_ckpt", ""),
            sam_type=llm_cfg.get("sam_type", "vit_h"),
            input_image=input_image,
            llm_send_image=llm_cfg.get("send_image", input_image is not None),
            bbox_json_text="",
            camera_json_text="",
            point_json_text="",
            prompt=exp.get("prompt", defaults.get("prompt", "")),
            negative_prompt=defaults.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT),
            **gen_defaults,
            motion_frame_idx=0,
            bbox_kf_state={},
            point_kf_state={},
            camera_kf_state={},
        )

        bbox_json = result[1]
        point_json = result[2]
        camera_json = result[3]
        new_prompt = result[7]
        status = result[25]

        if not bbox_json and not point_json:
            status = f"LLM 未生成运动参数: {status}"
            print(f"  WARNING: {status}")

        return bbox_json, camera_json, point_json, new_prompt, status

    except Exception as e:
        err_msg = f"LLM 调用失败: {e}"
        print(f"  ERROR: {err_msg}")
        return "", "", "", exp.get("prompt", ""), err_msg


def run_experiment(pipe, exp, defaults, llm_cfg, output_dir, skip_llm):
    name = exp["name"]
    exp_dir = Path(output_dir) / name
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    print(f"{'='*60}")

    image_path = exp["image"]
    if not os.path.exists(image_path):
        print(f"  WARNING: image not found: {image_path}, skipping")
        return
    input_image = Image.open(image_path).convert("RGB")
    input_image.save(str(exp_dir / "input.png"))

    # ── Resolve parameters ──
    params = dict(defaults)
    params["prompt"] = exp.get("prompt", defaults.get("prompt", ""))
    for key in ["height", "width", "num_frames", "num_inference_steps",
                 "cfg_scale", "sigma_shift", "seed", "fps", "negative_prompt"]:
        if key in exp:
            params[key] = exp[key]

    has_direct = any(exp.get(k) for k in ["bbox_json", "camera_json", "point_json"])
    has_llm = bool(exp.get("llm_instruction")) and llm_cfg and not skip_llm

    llm_status = ""
    if has_direct:
        params["bbox_json"] = exp.get("bbox_json", "")
        params["camera_json"] = exp.get("camera_json", "")
        params["point_json"] = exp.get("point_json", "")
        print("  Mode: direct (predefined JSON)")
    elif has_llm:
        print(f"  Mode: LLM (instruction: {exp['llm_instruction'][:60]})")
        bbox_j, camera_j, point_j, new_prompt, llm_status = call_llm_apply_instruction(
            exp, defaults, llm_cfg, input_image
        )
        params["bbox_json"] = bbox_j
        params["camera_json"] = camera_j
        params["point_json"] = point_j
        if new_prompt:
            params["prompt"] = new_prompt
        print(f"  LLM status: {llm_status}")
    else:
        params["bbox_json"] = ""
        params["camera_json"] = ""
        params["point_json"] = ""
        print("  Mode: no motion control")

    # Save resolved config
    with open(exp_dir / "config.json", "w") as f:
        f.write(json.dumps({
            "experiment": {k: v for k, v in exp.items() if k != "llm_instruction"},
            "resolved_prompt": params["prompt"],
            "has_bbox": bool(params.get("bbox_json")),
            "has_camera": bool(params.get("camera_json")),
            "has_point": bool(params.get("point_json")),
            "llm_status": llm_status,
            "gen_params": {k: params[k] for k in [
                "height", "width", "num_frames", "num_inference_steps",
                "cfg_scale", "sigma_shift", "seed", "fps"
            ] if k in params},
        }, indent=2, ensure_ascii=False))

    # ── Build control signals ──
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
        print("  Building bbox_mask...")
        bbox_mask = build_bbox_mask_from_json_str(
            params["bbox_json"],
            int(params["num_frames"]),
            int(params["height"]),
            int(params["width"]),
        )
        bbox_mask = bbox_mask.to(dtype=pipe_state_dtype(pipe), device=pipe.device)
        torch.save(bbox_mask.cpu(), str(exp_dir / "bbox_mask.pt"))

    track_video = None
    if any(params.get(k) for k in ["bbox_json", "camera_json", "point_json"]):
        print("  Building track_video...")
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

    # ── Generate ──
    print(f"  Generating ({params['width']}x{params['height']}, "
          f"{params['num_frames']} frames, seed={params['seed']})...")
    t0 = time.time()
    video_frames = pipe(**pipe_kwargs)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    if video_frames and len(video_frames) > 0:
        save_video(video_frames[0], str(exp_dir / "output.mp4"),
                   fps=int(params["fps"]), quality=5)
        print(f"  Saved: {exp_dir / 'output.mp4'}")
    else:
        print(f"  FAILED: no frames generated")

    return exp_dir


def main():
    parser = argparse.ArgumentParser(description="MotionCanvas 批量消融实验")
    parser.add_argument("--config", required=True, help="实验配置文件 (.json / .yaml)")
    parser.add_argument("--output_dir", default="./ablations", help="输出目录")
    parser.add_argument("--skip_llm", action="store_true", help="跳过 LLM 调用，仅使用直接参数")
    parser.add_argument("--resume", help="从指定实验名开始")
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
