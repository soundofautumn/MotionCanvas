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

配置文件格式见 ablation_config.yaml。
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
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


# ────────── helpers ──────────

def _resolve_env(obj):
    if isinstance(obj, str):
        import re
        return re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), m.group(0)), obj)
    if isinstance(obj, dict):
        return {k: _resolve_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env(v) for v in obj]
    return obj


def load_config(path):
    raw = Path(path).read_text()
    if path.endswith((".yaml", ".yml")):
        import yaml
        cfg = yaml.safe_load(raw)
    else:
        cfg = json.loads(raw)
    return _resolve_env(cfg)


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_json_size(obj, max_len=2000):
    s = json.dumps(obj, ensure_ascii=False)
    return s[:max_len] + "..." if len(s) > max_len else s


def load_checkpoint_weights(pipe, checkpoint_path, device="cpu"):
    print(f"  Loading checkpoint: {checkpoint_path}")
    t0 = time.time()
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
    print(f"  Checkpoint loaded in {time.time() - t0:.1f}s")
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

    t0 = time.time()
    print("Loading model manager...")
    model_manager = ModelManager(torch_dtype=torch_dtype, device="cpu")
    model_manager.load_models(model_paths)
    print(f"  Model manager ready in {time.time() - t0:.1f}s")

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


# ────────── LLM call with logging ──────────

def call_llm_apply_instruction(exp, defaults, llm_cfg, input_image):
    """
    调用 llm_assistant.py 获取运动参数，记录完整日志。

    返回 dict:
      bbox_json, camera_json, point_json, prompt, status,
      duration_sec, history, tool_calls, error
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

    t0 = time.time()
    error = None
    tool_calls_log = []

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

        duration = time.time() - t0

        history = result[0]
        bbox_json = result[1]
        point_json = result[2]
        camera_json = result[3]
        new_prompt = result[7]
        status = result[25]

        # Extract tool calls from history
        for msg in history:
            if isinstance(msg, dict) and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", {})
                    tool_calls_log.append({
                        "name": fn.get("name"),
                        "arguments": safe_json_size(fn.get("arguments", "")),
                    })

        if not bbox_json and not point_json:
            status = f"LLM did not generate motion params: {status}"

        return {
            "bbox_json": bbox_json,
            "camera_json": camera_json,
            "point_json": point_json,
            "prompt": new_prompt,
            "status": status,
            "duration_sec": round(duration, 2),
            "history": history,
            "tool_calls": tool_calls_log,
            "error": error,
        }

    except Exception as e:
        duration = time.time() - t0
        error = f"{type(e).__name__}: {e}"
        print(f"  ERROR: {error}")
        return {
            "bbox_json": "",
            "camera_json": "",
            "point_json": "",
            "prompt": exp.get("prompt", ""),
            "status": error,
            "duration_sec": round(duration, 2),
            "history": [],
            "tool_calls": [],
            "error": error,
        }


# ────────── experiment runner ──────────

def run_experiment(pipe, exp, defaults, llm_cfg, output_dir, skip_llm):
    name = exp["name"]
    exp_dir = Path(output_dir) / name
    exp_dir.mkdir(parents=True, exist_ok=True)

    log = {
        "experiment": name,
        "timestamp": now_iso(),
        "mode": None,
        "image": exp["image"],
        "prompt": exp.get("prompt", ""),
        "seed": None,
        "llm": None,
        "signal_building": None,
        "generation": None,
        "total_duration_sec": None,
        "error": None,
    }
    total_t0 = time.time()

    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    print(f"{'='*60}")

    # ── load image ──
    image_path = exp["image"]
    if not os.path.exists(image_path):
        msg = f"image not found: {image_path}"
        print(f"  WARNING: {msg}")
        log["error"] = msg
        (exp_dir / "log.json").write_text(json.dumps(log, indent=2, ensure_ascii=False))
        return
    input_image = Image.open(image_path).convert("RGB")
    input_image.save(str(exp_dir / "input.png"))

    # ── resolve parameters ──
    params = dict(defaults)
    params["prompt"] = exp.get("prompt", defaults.get("prompt", ""))
    for key in ["height", "width", "num_frames", "num_inference_steps",
                 "cfg_scale", "sigma_shift", "seed", "fps", "negative_prompt"]:
        if key in exp:
            params[key] = exp[key]
    log["seed"] = int(params["seed"])

    has_direct = any(exp.get(k) for k in ["bbox_json", "camera_json", "point_json"])
    has_llm = bool(exp.get("llm_instruction")) and llm_cfg and not skip_llm

    llm_log = None
    if has_direct:
        params["bbox_json"] = exp.get("bbox_json", "")
        params["camera_json"] = exp.get("camera_json", "")
        params["point_json"] = exp.get("point_json", "")
        log["mode"] = "direct"
        print("  Mode: direct (predefined JSON)")
    elif has_llm:
        log["mode"] = "llm"
        print(f"  Mode: LLM (instruction: {exp['llm_instruction'][:80]})")
        llm_result = call_llm_apply_instruction(exp, defaults, llm_cfg, input_image)
        params["bbox_json"] = llm_result["bbox_json"]
        params["camera_json"] = llm_result["camera_json"]
        params["point_json"] = llm_result["point_json"]
        if llm_result["prompt"]:
            params["prompt"] = llm_result["prompt"]
        llm_log = {
            "instruction": exp["llm_instruction"],
            "model": llm_cfg["model"],
            "duration_sec": llm_result["duration_sec"],
            "status": llm_result["status"],
            "tool_calls": llm_result["tool_calls"],
            "error": llm_result["error"],
        }
        print(f"  LLM: {llm_result['status']} ({llm_result['duration_sec']:.1f}s, "
              f"{len(llm_result['tool_calls'])} tool calls)")
        log["llm"] = llm_log
    else:
        params["bbox_json"] = ""
        params["camera_json"] = ""
        params["point_json"] = ""
        log["mode"] = "none"
        print("  Mode: no motion control")

    # Save resolved config + llm history
    exp_config = {
        "experiment": {k: v for k, v in exp.items() if k not in ("llm_instruction",)},
        "resolved_prompt": params["prompt"],
        "has_bbox": bool(params.get("bbox_json")),
        "has_camera": bool(params.get("camera_json")),
        "has_point": bool(params.get("point_json")),
        "gen_params": {
            k: params[k] for k in [
                "height", "width", "num_frames", "num_inference_steps",
                "cfg_scale", "sigma_shift", "seed", "fps"
            ] if k in params
        },
    }
    with open(exp_dir / "config.json", "w") as f:
        json.dump(exp_config, f, indent=2, ensure_ascii=False)
    if llm_result and llm_result["history"]:
        with open(exp_dir / "llm_history.json", "w") as f:
            json.dump(llm_result["history"], f, indent=2, ensure_ascii=False)

    # ── build control signals ──
    signal_t0 = time.time()
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
    signal_duration = time.time() - signal_t0
    log["signal_building"] = {
        "duration_sec": round(signal_duration, 2),
        "has_bbox_mask": bbox_mask is not None,
        "has_track_video": track_video is not None,
    }

    # ── generate video ──
    print(f"  Generating ({params['width']}x{params['height']}, "
          f"{params['num_frames']} frames, seed={params['seed']})...")
    gen_t0 = time.time()
    video_frames = pipe(**pipe_kwargs)
    gen_duration = time.time() - gen_t0
    print(f"  Generated in {gen_duration:.1f}s")

    success = bool(video_frames and len(video_frames) > 0)
    if success:
        save_video(video_frames[0], str(exp_dir / "output.mp4"),
                   fps=int(params["fps"]), quality=5)
        print(f"  Saved: {exp_dir / 'output.mp4'}")
    else:
        print(f"  FAILED: no frames generated")

    log["generation"] = {
        "duration_sec": round(gen_duration, 2),
        "num_frames": int(params["num_frames"]),
        "num_inference_steps": int(params["num_inference_steps"]),
        "width": int(params["width"]),
        "height": int(params["height"]),
        "cfg_scale": params["cfg_scale"],
        "sigma_shift": params["sigma_shift"],
        "seed": int(params["seed"]),
        "fps": int(params["fps"]),
        "output": str(exp_dir / "output.mp4") if success else None,
        "success": success,
    }

    total_duration = time.time() - total_t0
    log["total_duration_sec"] = round(total_duration, 2)

    (exp_dir / "log.json").write_text(json.dumps(log, indent=2, ensure_ascii=False))
    print(f"  Total: {total_duration:.1f}s")
    return log


# ────────── main ──────────

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

    # Load models (timed)
    pipe_start = time.time()
    pipe = build_pipeline(cfg)
    pipe_load_time = time.time() - pipe_start
    print(f"Model loading total: {pipe_load_time:.1f}s\n")

    experiments = cfg["experiments"]
    resumed = args.resume is None
    all_logs = []

    for exp in experiments:
        if not resumed:
            if exp["name"] == args.resume:
                resumed = True
            else:
                print(f"Skipping {exp['name']} (resume from {args.resume})")
                continue
        log = run_experiment(pipe, exp, defaults, llm_cfg, args.output_dir, args.skip_llm)
        if log:
            all_logs.append(log)

    # ── summary ──
    output_dir = Path(args.output_dir)
    summary_path = output_dir / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "experiment", "mode", "image", "prompt", "seed",
            "llm_duration", "llm_status", "llm_tool_calls",
            "signal_duration", "gen_duration", "gen_success",
            "total_duration",
        ])
        for log in all_logs:
            llm = log.get("llm") or {}
            sig = log.get("signal_building") or {}
            gen = log.get("generation") or {}
            w.writerow([
                log["experiment"],
                log["mode"],
                log["image"],
                log["prompt"],
                log["seed"],
                llm.get("duration_sec"),
                (llm.get("status") or "")[:80],
                len(llm.get("tool_calls", [])),
                sig.get("duration_sec"),
                gen.get("duration_sec"),
                gen.get("success"),
                log["total_duration_sec"],
            ])

    print(f"\n{'='*60}")
    print(f"All experiments done.")
    print(f"  Summary:  {summary_path}")
    print(f"  Per-exp:  {output_dir / '<name>' / 'log.json'}")
    print(f"  Model load: {pipe_load_time:.1f}s")
    total = sum(l.get("total_duration_sec", 0) for l in all_logs)
    print(f"  Total gen: {total:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
