#!/usr/bin/env python3
"""
MotionCanvas 批量消融实验脚本。

基于 apps/gradio/llm_assistant.py 的真实 LLM 调参逻辑。
LLM 提供 prompt、bbox、camera、point 等全部运动参数。

用法：
  python run_ablation.py --config config.yaml --output_dir ./ablations
  python run_ablation.py --config config.yaml --resume 03_xxx

配置文件格式：
  model:
    dit_path: ...
    vae_path: ...
    text_encoder_path: ...
    image_encoder_path: ...   # 可选
    checkpoint_path: ...      # 可选

  llm:
    base_url: https://api.siliconflow.cn/v1
    api_key: ${API_KEY}       # 或直接写 key
    model: Pro/moonshotai/Kimi-K2.5

  defaults:                   # 生成参数兜底（LLM 未修改时使用）
    height: 480
    width: 832
    num_frames: 49
    ...

  experiments:
    - name: 01_cat
      image: data/input_img/cat.png
      llm_instruction: "a cat walking on a sunny beach, cinematic lighting"
    - name: 02_bear
      image: data/input_img/bear.jpg
      llm_instruction: "a bear walking in a forest, sunlight through trees"
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

FULL_FRAME_BBOX = '{"objects": [{"frames": {"0": [0, 0, 1, 1]}}]}'


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


def try_parse_json_str(s):
    if not s or not isinstance(s, str):
        return s
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return s


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
    """
    完全遵循 apps/gradio/motioncanvas.py 的 load_models 逻辑。
    - dit / vae / text_encoder 为必需
    - image_encoder / motion_controller / vace / checkpoint 均为可选
    """
    mc = cfg["model"]
    torch_dtype = torch.bfloat16
    device = "cuda"

    # 必需模型检查
    required = [("dit_path", mc["dit_path"]), ("vae_path", mc["vae_path"]),
                ("text_encoder_path", mc["text_encoder_path"])]
    for name, p in required:
        if not p or not os.path.exists(p):
            raise FileNotFoundError(f"{name} not found: {p}")

    model_paths = [mc["text_encoder_path"], mc["vae_path"], mc["dit_path"]]

    iep = mc.get("image_encoder_path")
    if iep and os.path.exists(iep):
        model_paths.append(iep)

    mcp = mc.get("motion_controller_path")
    if mcp and os.path.exists(mcp):
        model_paths.append(mcp)

    vace_dir = mc.get("vace_dir")
    if vace_dir and os.path.isdir(vace_dir):
        vace_files = [
            os.path.join(vace_dir, "diffusion_pytorch_model.safetensors"),
            os.path.join(vace_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
            os.path.join(vace_dir, "Wan2.1_VAE.pth"),
        ]
        missing = [f for f in vace_files if not os.path.exists(f)]
        if missing:
            raise FileNotFoundError(f"VACE files missing: {missing}")
        model_paths.extend(vace_files)

    t0 = time.time()
    print("Loading model manager...")
    model_manager = ModelManager(torch_dtype=torch_dtype, device="cpu")
    model_manager.load_models(model_paths)
    print(f"  Model manager ready in {time.time() - t0:.1f}s")

    print("Creating pipeline...")
    pipe = WanVideoPipeline_motioncanvas.from_model_manager(
        model_manager, torch_dtype=torch_dtype, device=device
    )

    ckpt = mc.get("checkpoint_path")
    if ckpt and os.path.exists(ckpt):
        pipe = load_checkpoint_weights(pipe, ckpt, device="cpu")
        pipe.bbox_zeroconv = pipe.bbox_zeroconv.to(dtype=torch_dtype, device=device)

    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    return pipe


def pipe_state_dtype(pipe):
    return next(pipe.parameters()).dtype


# ────────── LLM call ──────────

def call_llm(exp, defaults, llm_cfg, input_image):
    from apps.gradio.llm_assistant import (
        llm_apply_instruction,
        clear_llm_tool_log,
        get_llm_tool_log,
    )

    gen_kw = {
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
    clear_llm_tool_log()
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
            prompt="",
            negative_prompt=defaults.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT),
            **gen_kw,
            motion_frame_idx=0,
            bbox_kf_state={},
            point_kf_state={},
            camera_kf_state={},
        )

        duration = time.time() - t0
        history = result[0]
        bbox_json = result[1] or ""
        point_json = result[2] or ""
        camera_json = result[3] or ""
        new_prompt = result[7] or ""
        status = result[25]

        tool_rounds = get_llm_tool_log()
        tool_summary = []
        for r in tool_rounds:
            if r["type"] == "tool_calls":
                tool_summary.append({
                    "round": r["round"],
                    "calls": [c["name"] for c in r["calls"]],
                })
            elif r["type"] == "fallback_ops":
                tool_summary.append({
                    "round": 0,
                    "calls": [f"ops:{len(r['ops'])} updates:{list(r['updates'].keys())}"],
                })

        return {
            "bbox_json": bbox_json,
            "camera_json": camera_json,
            "point_json": point_json,
            "prompt": new_prompt,
            "status": status,
            "duration_sec": round(duration, 2),
            "history": history,
            "tool_rounds": tool_rounds,
            "tool_summary": tool_summary,
            "error": None,
        }

    except Exception as e:
        duration = time.time() - t0
        err = f"{type(e).__name__}: {e}"
        print(f"  ERROR: {err}")
        return {
            "bbox_json": "", "camera_json": "", "point_json": "",
            "prompt": "", "status": err,
            "duration_sec": round(duration, 2),
            "history": [], "tool_calls": [], "tool_rounds": [], "tool_summary": [],
            "error": err,
        }


# ────────── experiment runner ──────────

def run_experiment(pipe, exp, defaults, llm_cfg, output_dir, skip_llm):
    name = exp["name"]
    exp_dir = Path(output_dir) / name
    exp_dir.mkdir(parents=True, exist_ok=True)

    log = {
        "experiment": name, "timestamp": now_iso(),
        "image": exp["image"], "instruction": exp.get("llm_instruction", ""),
        "prompt": "", "seed": None, "mode": None,
        "llm": None, "signal_building": None, "generation": None,
        "total_duration_sec": None, "error": None,
    }
    total_t0 = time.time()

    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    print(f"{'='*60}")

    # ── image ──
    image_path = exp["image"]
    if not os.path.exists(image_path):
        msg = f"image not found: {image_path}"
        print(f"  WARNING: {msg}")
        log["error"] = msg
        (exp_dir / "log.json").write_text(json.dumps(log, indent=2, ensure_ascii=False))
        return
    input_image = Image.open(image_path).convert("RGB")
    input_image.save(str(exp_dir / "input.png"))

    # ── params (defaults, may be overridden by LLM) ──
    params = dict(defaults)
    params["prompt"] = ""
    for k in ["height", "width", "num_frames", "num_inference_steps",
               "cfg_scale", "sigma_shift", "seed", "fps", "negative_prompt"]:
        if k in exp:
            params[k] = exp[k]

    # ── resolve via LLM or direct ──
    has_direct = any(exp.get(k) for k in ["bbox_json", "camera_json", "point_json"])
    has_llm = bool(exp.get("llm_instruction")) and llm_cfg and not skip_llm

    llm_log = None
    if has_direct:
        params["bbox_json"] = exp.get("bbox_json", "")
        params["camera_json"] = exp.get("camera_json", "")
        params["point_json"] = exp.get("point_json", "")
        log["mode"] = "direct"
        print("  Mode: direct (predefined params)")
    elif has_llm:
        log["mode"] = "llm"
        print(f"  Mode: LLM — {exp['llm_instruction'][:80]}")
        r = call_llm(exp, defaults, llm_cfg, input_image)
        params["bbox_json"] = r["bbox_json"]
        params["camera_json"] = r["camera_json"]
        params["point_json"] = r["point_json"]
        if r["prompt"]:
            params["prompt"] = r["prompt"]
        tc_count = sum(len(r["calls"]) for r in r["tool_rounds"])
        llm_log = {
            "instruction": exp["llm_instruction"],
            "model": llm_cfg["model"],
            "duration_sec": r["duration_sec"],
            "status": r["status"],
            "tool_rounds": r["tool_summary"],
            "error": r["error"],
        }
        print(f"  LLM: {r['status']}  ({r['duration_sec']:.1f}s, {tc_count} tool calls in {len(r['tool_rounds'])} rounds)")
        if r["prompt"]:
            print(f"  Prompt from LLM: {r['prompt'][:100]}")
        log["llm"] = llm_log
        if r["history"]:
            (exp_dir / "llm_history.json").write_text(json.dumps(r["history"], indent=2, ensure_ascii=False))
        if r["tool_rounds"]:
            (exp_dir / "llm_tool_calls.json").write_text(json.dumps(r["tool_rounds"], indent=2, ensure_ascii=False))
    else:
        params["bbox_json"] = ""
        params["camera_json"] = ""
        params["point_json"] = ""
        log["mode"] = "none"
        print("  Mode: no motion (empty params)")

    # ── fallback: bbox is required by pipeline ──
    if not params.get("bbox_json"):
        print("  WARNING: no bbox from LLM, using full-frame fallback")
        params["bbox_json"] = FULL_FRAME_BBOX
        if not params.get("camera_json"):
            params["camera_json"] = '{"camera": {"keyframes": [{"frame": 0, "zoom": 1.0, "pan": [0, 0], "rotation": 0}]}}'

    # ── save config (JSON 字符串转为实际对象) ──
    log["prompt"] = params["prompt"]
    log["seed"] = int(params.get("seed", 42))
    (exp_dir / "config.json").write_text(json.dumps({
        "prompt": params["prompt"],
        "has_bbox": bool(params.get("bbox_json")),
        "has_camera": bool(params.get("camera_json")),
        "has_point": bool(params.get("point_json")),
        "bbox_json": try_parse_json_str(params.get("bbox_json", "")),
        "camera_json": try_parse_json_str(params.get("camera_json", "")),
        "point_json": try_parse_json_str(params.get("point_json", "")),
        "gen_params": {k: params[k] for k in
            ["height", "width", "num_frames", "num_inference_steps",
             "cfg_scale", "sigma_shift", "seed", "fps"] if k in params},
    }, indent=2, ensure_ascii=False))

    # ── build control signals ──
    st0 = time.time()
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
            params["bbox_json"], int(params["num_frames"]),
            int(params["height"]), int(params["width"]),
        )
        bbox_mask = bbox_mask.to(dtype=pipe_state_dtype(pipe), device=pipe.device)
        torch.save(bbox_mask.cpu(), str(exp_dir / "bbox_mask.pt"))

    track_video = None
    if any(params.get(k) for k in ["bbox_json", "camera_json", "point_json"]):
        print("  Building track_video...")
        track_video = _build_fallback_track_video(
            params.get("bbox_json", ""), params.get("camera_json", ""),
            params.get("point_json", ""),
            int(params["num_frames"]), int(params["height"]), int(params["width"]),
            pipe_state_dtype(pipe), pipe.device,
        )
        if track_video is not None:
            torch.save(track_video.cpu(), str(exp_dir / "track_video.pt"))

    pipe_kwargs["bbox_mask"] = bbox_mask
    pipe_kwargs["track_video"] = track_video
    log["signal_building"] = {
        "duration_sec": round(time.time() - st0, 2),
        "has_bbox_mask": bbox_mask is not None,
        "has_track_video": track_video is not None,
    }

    # ── generate ──
    print(f"  Generating ({params['width']}x{params['height']}, "
          f"{params['num_frames']} frames, seed={params['seed']})...")
    gt0 = time.time()
    video_frames = pipe(**pipe_kwargs)
    gdur = time.time() - gt0
    print(f"  Generated in {gdur:.1f}s")

    success = bool(video_frames and len(video_frames) > 0)
    if success:
        save_video(video_frames[0], str(exp_dir / "output.mp4"),
                   fps=int(params["fps"]), quality=5)
        print(f"  Saved: {exp_dir / 'output.mp4'}")
    else:
        print("  FAILED: no frames generated")

    log["generation"] = {
        "duration_sec": round(gdur, 2),
        "num_frames": int(params["num_frames"]),
        "num_inference_steps": int(params["num_inference_steps"]),
        "width": int(params["width"]), "height": int(params["height"]),
        "cfg_scale": params["cfg_scale"], "sigma_shift": params["sigma_shift"],
        "seed": int(params["seed"]), "fps": int(params["fps"]),
        "success": success,
    }
    log["total_duration_sec"] = round(time.time() - total_t0, 2)
    (exp_dir / "log.json").write_text(json.dumps(log, indent=2, ensure_ascii=False))
    print(f"  Total: {log['total_duration_sec']:.1f}s")
    return log


# ────────── main ──────────

def main():
    parser = argparse.ArgumentParser(description="MotionCanvas 批量消融实验")
    parser.add_argument("--config", required=True, help="实验配置文件 (.json / .yaml)")
    parser.add_argument("--output_dir", default="./ablation_results", help="输出目录")
    parser.add_argument("--skip_llm", action="store_true", help="跳过 LLM 调用")
    parser.add_argument("--resume", help="从指定实验名开始")
    args = parser.parse_args()

    cfg = load_config(args.config)
    defaults = cfg.get("defaults", {})
    defaults.setdefault("negative_prompt", DEFAULT_NEGATIVE_PROMPT)
    defaults.setdefault("seed", 42)
    defaults.setdefault("height", 480)
    defaults.setdefault("width", 832)
    defaults.setdefault("num_frames", 49)
    llm_cfg = cfg.get("llm")

    pipe_start = time.time()
    pipe = build_pipeline(cfg)
    print(f"Model loading total: {time.time() - pipe_start:.1f}s\n")

    all_logs = []
    resumed = args.resume is None
    for exp in cfg["experiments"]:
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
    out = Path(args.output_dir)
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "experiment", "mode", "image", "prompt", "seed",
            "llm_duration", "llm_tool_calls", "llm_status",
            "signal_duration", "gen_duration", "gen_success", "total_duration",
        ])
        for log in all_logs:
            llm = log.get("llm") or {}
            sig = log.get("signal_building") or {}
            gen = log.get("generation") or {}
            w.writerow([
                log["experiment"], log["mode"], log["image"], log["prompt"],
                log["seed"],
                llm.get("duration_sec"),
                len(llm.get("tool_calls", [])),
                (llm.get("status") or "")[:80],
                sig.get("duration_sec"),
                gen.get("duration_sec"),
                gen.get("success"),
                log["total_duration_sec"],
            ])

    print(f"\n{'='*60}")
    print("All experiments done.")
    print(f"  Summary:  {out / 'summary.csv'}")
    print(f"  Per-exp:  {out / '<name>' / 'log.json'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
