"""
MotionCanvas Gradio GUI
基于 Gradio 的 MotionCanvas 视频生成界面
"""

import os
import sys
import tempfile
import json
import torch
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import gradio as gr
from diffsynth import ModelManager, save_video
from diffsynth.pipelines.wan_video_motioncanvas import WanVideoPipeline_motioncanvas
from diffsynth.pipelines.tracker_utils import get_video_track_video

DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, "
    "JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
    "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, messy background, three legs, many people in the background, "
    "walking backwards"
)

CUSTOM_CSS = """
.gradio-container, .gradio-container * {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
                 'Noto Sans SC', 'PingFang SC', 'Microsoft YaHei',
                 'Helvetica Neue', Arial, sans-serif !important;
}
.gradio-container code, .gradio-container pre, .gradio-container .cm-editor * {
    font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code',
                 'Source Code Pro', Consolas, monospace !important;
}
.header-banner {
    text-align: center;
    padding: 20px 0 8px;
}
.header-banner h1 {
    font-size: 2.2em !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #6366f1, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 2px !important;
}
.header-banner p {
    color: #6b7280;
    font-size: 0.95em;
}
.status-box textarea { font-size: 14px !important; }
.generate-btn {
    min-height: 52px !important;
    font-size: 1.1em !important;
    font-weight: 600 !important;
}
.kf-label {
    text-align: center;
    font-weight: 600;
    font-size: 0.82em;
    padding: 5px 0;
    border-radius: 6px;
    color: white;
    margin-bottom: 4px;
}
.kf-start { background: linear-gradient(135deg, #22c55e, #16a34a); }
.kf-mid   { background: linear-gradient(135deg, #eab308, #ca8a04); }
.kf-end   { background: linear-gradient(135deg, #ef4444, #dc2626); }
.section-title {
    font-size: 1.05em !important;
    font-weight: 600 !important;
    margin-bottom: 4px !important;
}
"""

pipe_state = {"pipe": None, "torch_dtype": None, "loaded_config": None, "cotracker": None}


# ==================== Model Loading ====================

def load_checkpoint_weights(pipe, checkpoint_path, device="cpu"):
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

    dit_sd, bbox_sd = {}, {}
    for k, v in state_dict.items():
        if k.startswith("pipe."):
            k = k[5:]
        if k.startswith("dit.") or k.startswith("denoising_model."):
            dit_sd[k.replace("denoising_model.", "").replace("dit.", "")] = v
        elif k.startswith("bbox_zeroconv."):
            bbox_sd[k.replace("bbox_zeroconv.", "")] = v

    info = []
    if dit_sd:
        m, u = pipe.dit.load_state_dict(dit_sd, strict=False)
        info.append(f"DiT: {len(dit_sd)} params (missing={len(m)}, unexpected={len(u)})")
    if bbox_sd:
        pipe.bbox_zeroconv.load_state_dict(bbox_sd, strict=True)
        info.append(f"bbox_zeroconv: {len(bbox_sd)} params")
    return pipe, "; ".join(info)


def load_models(dit_path, vae_path, text_encoder_path, image_encoder_path,
                                motion_controller_path, vace_dir, checkpoint_path, dtype_str):
    config_key = (f"{dit_path}|{vae_path}|{text_encoder_path}|"
                                    f"{image_encoder_path}|{motion_controller_path}|{vace_dir}|"
                                    f"{checkpoint_path}|{dtype_str}")
    if pipe_state["loaded_config"] == config_key and pipe_state["pipe"] is not None:
        return "✅ 模型已加载（缓存命中）"

    torch_dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for p, name in [(dit_path, "DiT"), (vae_path, "VAE"),
                    (text_encoder_path, "Text Encoder")]:
        if not p or not os.path.exists(p):
            return f"❌ {name} 路径无效: {p}"
    if motion_controller_path and not os.path.exists(motion_controller_path):
        return f"❌ Motion Controller 路径无效: {motion_controller_path}"

    model_paths = [text_encoder_path, vae_path, dit_path]
    if image_encoder_path and os.path.exists(image_encoder_path):
        model_paths.append(image_encoder_path)
    if motion_controller_path:
        model_paths.append(motion_controller_path)
    if vace_dir and os.path.isdir(vace_dir):
        vace_files = [
            os.path.join(vace_dir, "diffusion_pytorch_model.safetensors"),
            os.path.join(vace_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
            os.path.join(vace_dir, "Wan2.1_VAE.pth"),
        ]
        for file_path in vace_files:
            if not os.path.exists(file_path):
                return f"❌ VACE 文件缺失: {file_path}"
        model_paths.extend(vace_files)
    elif vace_dir:
        return f"❌ VACE 目录无效: {vace_dir}"

    model_manager = ModelManager(torch_dtype=torch_dtype, device="cpu")
    model_manager.load_models(model_paths)

    pipe = WanVideoPipeline_motioncanvas.from_model_manager(
        model_manager, torch_dtype=torch_dtype, device=device
    )

    ckpt_info = ""
    if checkpoint_path and os.path.exists(checkpoint_path):
        pipe, ckpt_info = load_checkpoint_weights(pipe, checkpoint_path, device="cpu")
        pipe.bbox_zeroconv = pipe.bbox_zeroconv.to(dtype=torch_dtype, device=device)
        ckpt_info = f" | Checkpoint: {ckpt_info}"

    pipe.enable_vram_management(num_persistent_param_in_dit=None)

    pipe_state["pipe"] = pipe
    pipe_state["torch_dtype"] = torch_dtype
    pipe_state["loaded_config"] = config_key

    return f"✅ 模型加载成功{ckpt_info}"


# ==================== Bbox / Motion Control ====================

def build_bbox_mask_from_json_str(json_str, num_frames, height, width):
    bbox_data = json.loads(json_str)
    mask = torch.zeros(1, 3, num_frames, height, width)
    for obj in bbox_data.get("objects", []):
        for fi_str, bbox in obj.get("frames", {}).items():
            fi = int(fi_str)
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
    return mask * 2.0 - 1.0


def build_object_masks_from_bbox_json(json_str, num_frames, height, width):
    bbox_data = json.loads(json_str)
    objects = bbox_data.get("objects", [])
    if not objects:
        return None

    obj_masks = []
    for obj in objects:
        obj_mask = torch.zeros(num_frames, 1, height, width, dtype=torch.bool)
        for fi_str, bbox in obj.get("frames", {}).items():
            fi = int(fi_str)
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
            if x2 > x1 and y2 > y1:
                obj_mask[fi, 0, y1:y2, x1:x2] = True
        obj_masks.append(obj_mask)

    if not obj_masks:
        return None
    return torch.stack(obj_masks, dim=0)


def build_object_masks_from_bbox_json_interpolated(json_str, num_frames, height, width):
    bbox_data = json.loads(json_str)
    objects = bbox_data.get("objects", [])
    if not objects:
        return None

    obj_masks = []
    for obj in objects:
        frames = obj.get("frames", {})
        if not frames:
            continue

        keyframes = []
        for fi_str, bbox in frames.items():
            fi = int(fi_str)
            if fi >= num_frames:
                continue
            x1, y1, x2, y2 = bbox
            if all(0 <= v <= 1.0 for v in [x1, y1, x2, y2]):
                x1, x2 = x1 * width, x2 * width
                y1, y2 = y1 * height, y2 * height
            keyframes.append((fi, float(x1), float(y1), float(x2), float(y2)))

        if not keyframes:
            continue

        keyframes = sorted(keyframes, key=lambda x: x[0])
        obj_mask = torch.zeros(num_frames, 1, height, width, dtype=torch.bool)

        for idx in range(len(keyframes) - 1):
            f0, x10, y10, x20, y20 = keyframes[idx]
            f1, x11, y11, x21, y21 = keyframes[idx + 1]
            span = max(1, f1 - f0)
            for f in range(f0, f1 + 1):
                t = (f - f0) / span
                x1 = x10 + (x11 - x10) * t
                y1 = y10 + (y11 - y10) * t
                x2 = x20 + (x21 - x20) * t
                y2 = y20 + (y21 - y20) * t
                x1 = int(max(0, min(width, round(x1))))
                x2 = int(max(0, min(width, round(x2))))
                y1 = int(max(0, min(height, round(y1))))
                y2 = int(max(0, min(height, round(y2))))
                if x2 > x1 and y2 > y1:
                    obj_mask[f, 0, y1:y2, x1:x2] = True

        f0, x10, y10, x20, y20 = keyframes[0]
        for f in range(0, f0):
            x1 = int(max(0, min(width, round(x10))))
            x2 = int(max(0, min(width, round(x20))))
            y1 = int(max(0, min(height, round(y10))))
            y2 = int(max(0, min(height, round(y20))))
            if x2 > x1 and y2 > y1:
                obj_mask[f, 0, y1:y2, x1:x2] = True

        f1, x11, y11, x21, y21 = keyframes[-1]
        for f in range(f1, num_frames):
            x1 = int(max(0, min(width, round(x11))))
            x2 = int(max(0, min(width, round(x21))))
            y1 = int(max(0, min(height, round(y11))))
            y2 = int(max(0, min(height, round(y21))))
            if x2 > x1 and y2 > y1:
                obj_mask[f, 0, y1:y2, x1:x2] = True

        obj_masks.append(obj_mask)

    if not obj_masks:
        return None
    return torch.stack(obj_masks, dim=0)


def build_video_rgb_from_images(input_image, end_image, num_frames, height, width):
    if input_image is None:
        return None

    def to_frame(img):
        img = img.resize((width, height)).convert("RGB")
        arr = np.array(img)
        return torch.from_numpy(arr).permute(2, 0, 1)

    first_frame = to_frame(input_image)
    frames = [first_frame] * int(num_frames)
    if end_image is not None:
        frames[-1] = to_frame(end_image)
    return torch.stack(frames, dim=0)


def build_video_rgb_from_bbox_motion(input_image, bbox_json_text, num_frames, height, width):
    if input_image is None:
        return None

    if not bbox_json_text or not bbox_json_text.strip():
        return None

    bbox_data = json.loads(bbox_json_text)
    objects = bbox_data.get("objects", [])
    if not objects:
        return None

    frames = objects[0].get("frames", {})
    if not frames:
        return None

    keyframes = []
    for fi_str, bbox in frames.items():
        fi = int(fi_str)
        if fi >= num_frames:
            continue
        x1, y1, x2, y2 = bbox
        if all(0 <= v <= 1.0 for v in [x1, y1, x2, y2]):
            x1, x2 = x1 * width, x2 * width
            y1, y2 = y1 * height, y2 * height
        keyframes.append((fi, float(x1), float(y1), float(x2), float(y2)))

    if not keyframes:
        return None

    keyframes = sorted(keyframes, key=lambda x: x[0])
    base = input_image.resize((width, height)).convert("RGB")

    def interp_bbox(f):
        if f <= keyframes[0][0]:
            return keyframes[0][1:]
        if f >= keyframes[-1][0]:
            return keyframes[-1][1:]
        for idx in range(len(keyframes) - 1):
            f0, x10, y10, x20, y20 = keyframes[idx]
            f1, x11, y11, x21, y21 = keyframes[idx + 1]
            if f0 <= f <= f1:
                span = max(1, f1 - f0)
                t = (f - f0) / span
                x1 = x10 + (x11 - x10) * t
                y1 = y10 + (y11 - y10) * t
                x2 = x20 + (x21 - x20) * t
                y2 = y20 + (y21 - y20) * t
                return x1, y1, x2, y2
        return keyframes[-1][1:]

    frames_out = []
    base_cx = width / 2.0
    base_cy = height / 2.0
    for f in range(num_frames):
        x1, y1, x2, y2 = interp_bbox(f)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        dx = base_cx - cx
        dy = base_cy - cy
        shifted = Image.new("RGB", (width, height), (0, 0, 0))
        shifted.paste(base, (int(round(dx)), int(round(dy))))
        frames_out.append(torch.from_numpy(np.array(shifted)).permute(2, 0, 1))

    return torch.stack(frames_out, dim=0)


def build_track_video_preview(track_video, input_image=None, fps=15):
    if track_video is None:
        return None

    track_video = track_video.detach().to("cpu", dtype=torch.float32)
    if track_video.ndim == 5:
        track_video = track_video[0]
    track_video = track_video.abs().sum(dim=0)  # [T, H, W]

    base_img = None
    if input_image is not None:
        base_img = input_image.copy().convert("RGB")

    frames = []
    for t in range(track_video.shape[0]):
        frame = track_video[t]
        frame = frame - frame.min()
        denom = frame.max() - frame.min()
        if denom > 0:
            frame = frame / denom
        frame = (frame * 255.0).clamp(0, 255).to(torch.uint8).numpy()
        heat = Image.fromarray(frame, mode="L").convert("RGB")

        if base_img is not None:
            heat = heat.resize(base_img.size, Image.BILINEAR)
            overlay = Image.blend(base_img, heat, alpha=0.5)
            frames.append(overlay)
        else:
            frames.append(heat)

    if not frames:
        return None

    preview_path = os.path.join(tempfile.gettempdir(), "track_video_preview.mp4")
    save_video(frames, preview_path, fps=int(fps), quality=5)
    return preview_path


def compute_track_video(
    pipe,
    torch_dtype,
    device,
    bbox_mask,
    bbox_json_text,
    input_image,
    end_image,
    num_frames,
    height,
    width,
):
    if bbox_mask is None:
        return None

    object_masks = None
    if bbox_json_text and bbox_json_text.strip():
        object_masks = build_object_masks_from_bbox_json_interpolated(
            bbox_json_text, int(num_frames), int(height), int(width)
        )

    if object_masks is None:
        if bbox_mask.ndim == 5:
            merged = (bbox_mask.squeeze(0) > 0).any(dim=0, keepdim=True)
        else:
            merged = (bbox_mask > 0).any(dim=0, keepdim=True)
        object_masks = merged.unsqueeze(0).to(dtype=torch.bool)

    reference_imgs_indicator = [object_masks.shape[0]]
    video_rgb = build_video_rgb_from_bbox_motion(
        input_image, bbox_json_text, int(num_frames), int(height), int(width)
    )
    if video_rgb is None:
        video_rgb = build_video_rgb_from_images(
            input_image, end_image, int(num_frames), int(height), int(width)
        )
    if video_rgb is None:
        return None

    tiler_kwargs = {"tiled": True, "tile_size": (30, 52), "tile_stride": (15, 26)}
    pipe.load_models_to_device(["vae"])
    bbox_latents = pipe.encode_video(bbox_mask, **tiler_kwargs)
    lat_c = bbox_latents.shape[1]

    device_obj = torch.device(device)
    cotracker = load_cotracker(device=device_obj, dtype=torch.float32)
    video_rgb = video_rgb.unsqueeze(0).to(device=device_obj, dtype=torch.float32)
    object_masks = object_masks.to(device=device_obj)

    object_masks_per_sample = torch.split(object_masks, reference_imgs_indicator, dim=0)
    track_video, _, _ = get_video_track_video(
        cotracker,
        video_rgb,
        object_masks_per_sample,
        pipe.downsample_ratios,
        lat_c,
        grid_size=12,
        device=device_obj,
        dtype=torch.float32,
    )
    return track_video.to(dtype=torch_dtype, device=device)


def load_cotracker(device, dtype):
    if pipe_state.get("cotracker") is not None:
        return pipe_state["cotracker"]

    cotracker_local = os.environ.get("COTRACKER_HUB_DIR")
    if cotracker_local and os.path.isdir(os.path.join(cotracker_local, "facebookresearch_co-tracker_main")):
        torch.hub.set_dir(cotracker_local)
        cotracker = torch.hub.load(
            os.path.join(cotracker_local, "facebookresearch_co-tracker_main"),
            "cotracker3_offline",
            source="local",
        )
    else:
        cotracker = torch.hub.load(
            "facebookresearch/co-tracker",
            "cotracker3_offline",
            trust_repo=True,
        )

    cotracker = cotracker.to(device=device, dtype=dtype)
    cotracker.requires_grad_(False)
    pipe_state["cotracker"] = cotracker
    return cotracker


def extract_bbox_from_editor(editor_data):
    """从 ImageEditor 的涂抹区域提取归一化 bbox [x1, y1, x2, y2]。"""
    if editor_data is None:
        return None
    layers = editor_data.get("layers", [])
    if not layers:
        return None
    for layer in layers:
        if not isinstance(layer, np.ndarray):
            continue
        if layer.ndim == 3 and layer.shape[2] >= 4:
            alpha = layer[:, :, 3]
        elif layer.ndim == 3:
            alpha = np.any(layer > 0, axis=2).astype(np.uint8) * 255
        elif layer.ndim == 2:
            alpha = layer
        else:
            continue
        if not np.any(alpha > 0):
            continue
        rows = np.any(alpha > 0, axis=1)
        cols = np.any(alpha > 0, axis=0)
        y_idx = np.where(rows)[0]
        x_idx = np.where(cols)[0]
        h, w = alpha.shape[:2]
        return [
            round(x_idx[0] / w, 4),
            round(y_idx[0] / h, 4),
            round((x_idx[-1] + 1) / w, 4),
            round((y_idx[-1] + 1) / h, 4),
        ]
    return None


def sync_image_to_editors(input_image):
    """将输入图像同步到三个关键帧画布作为背景。"""
    if input_image is None:
        return None, None, None
    img = np.array(input_image)
    return img, img, img


def generate_bbox_json_from_editors(editor_start, editor_mid, editor_end, num_frames):
    """从三个关键帧画布的涂抹区域提取 bbox 并生成 JSON。"""
    bbox_start = extract_bbox_from_editor(editor_start)
    bbox_mid = extract_bbox_from_editor(editor_mid)
    bbox_end = extract_bbox_from_editor(editor_end)

    if all(b is None for b in [bbox_start, bbox_mid, bbox_end]):
        return ""

    nf = int(num_frames)
    frames = {}
    if bbox_start is not None:
        frames["0"] = bbox_start
    if bbox_mid is not None:
        frames[str(nf // 2)] = bbox_mid
    if bbox_end is not None:
        frames[str(nf - 1)] = bbox_end

    return json.dumps({"objects": [{"frames": frames}]}, indent=2)


def preview_motion_path(input_image, editor_start, editor_mid, editor_end, num_frames):
    """在输入图像上叠加绘制各关键帧的 bbox 矩形，预览运动路径。"""
    if input_image is None:
        return None

    nf = int(num_frames)
    img = input_image.copy().convert("RGBA")
    w, h = img.size

    keyframes = [
        (editor_start, (46, 204, 113),  f"起始帧 (F0)"),
        (editor_mid,   (241, 196, 15),  f"中间帧 (F{nf // 2})"),
        (editor_end,   (231, 76, 60),   f"结束帧 (F{nf - 1})"),
    ]

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    centers = []

    for editor, color, label in keyframes:
        bbox = extract_bbox_from_editor(editor)
        if bbox is None:
            continue
        x1 = int(bbox[0] * w)
        y1 = int(bbox[1] * h)
        x2 = int(bbox[2] * w)
        y2 = int(bbox[3] * h)

        fill_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        fd = ImageDraw.Draw(fill_layer)
        fd.rectangle([x1, y1, x2, y2], fill=(*color, 50))
        overlay = Image.alpha_composite(overlay, fill_layer)

        draw = ImageDraw.Draw(overlay)
        draw.rectangle([x1, y1, x2, y2], outline=(*color, 220), width=3)
        draw.text((x1 + 6, y1 + 6), label, fill=(*color, 255))

        centers.append(((x1 + x2) // 2, (y1 + y2) // 2))

    if len(centers) >= 2:
        draw = ImageDraw.Draw(overlay)
        for i in range(len(centers) - 1):
            draw.line([centers[i], centers[i + 1]],
                      fill=(255, 255, 255, 200), width=2)

    result = Image.alpha_composite(img, overlay)
    return result.convert("RGB")


# ==================== Camera Motion Control ====================

def build_camera_mask_from_json_str(json_str, num_frames, height, width):
    """将相机 JSON 转换为张量。格式: {'camera': {'keyframes': [{'frame': 0, 'zoom': 1.0, 'pan': [0, 0], 'rotation': 0}, ...]}}"""
    try:
        camera_data = json.loads(json_str)
        keyframes_list = camera_data.get("camera", {}).get("keyframes", [])
    except:
        return None

    if not keyframes_list:
        return None

    # 创建张量存储相机参数 (1, 4, num_frames, height, width)
    # 4 通道: zoom, pan_x, pan_y, rotation
    mask = torch.zeros(1, 4, num_frames, height, width)

    # 提取关键帧信息
    kf_dict = {}
    for kf in keyframes_list:
        frame_idx = int(kf.get("frame", 0))
        kf_dict[frame_idx] = {
            "zoom": float(kf.get("zoom", 1.0)),
            "pan_x": float(kf.get("pan", [0, 0])[0]),
            "pan_y": float(kf.get("pan", [0, 0])[1]),
            "rotation": float(kf.get("rotation", 0)),
        }

    # 线性插值填充所有帧
    frame_indices = sorted(kf_dict.keys())
    if not frame_indices:
        return None

    for frame_idx in range(num_frames):
        # 找到相邻的关键帧
        prev_idx = 0
        next_idx = num_frames - 1
        for idx in frame_indices:
            if idx <= frame_idx:
                prev_idx = idx
            if idx >= frame_idx and next_idx == num_frames - 1:
                next_idx = idx

        if prev_idx == next_idx:
            # 在一个关键帧处或之前
            kf_data = kf_dict.get(prev_idx, {"zoom": 1.0, "pan_x": 0, "pan_y": 0, "rotation": 0})
        else:
            # 在两个关键帧之间线性插值
            t = (frame_idx - prev_idx) / (next_idx - prev_idx)
            prev_kf = kf_dict.get(prev_idx, {"zoom": 1.0, "pan_x": 0, "pan_y": 0, "rotation": 0})
            next_kf = kf_dict.get(next_idx, {"zoom": 1.0, "pan_x": 0, "pan_y": 0, "rotation": 0})
            kf_data = {
                "zoom": prev_kf["zoom"] * (1 - t) + next_kf["zoom"] * t,
                "pan_x": prev_kf["pan_x"] * (1 - t) + next_kf["pan_x"] * t,
                "pan_y": prev_kf["pan_y"] * (1 - t) + next_kf["pan_y"] * t,
                "rotation": prev_kf["rotation"] * (1 - t) + next_kf["rotation"] * t,
            }

        # 归一化到 [-1, 1] 范围
        # zoom: 0.5-2.0 -> [-1, 1]
        zoom_normalized = (kf_data["zoom"] - 0.5) / 0.75  # 1.0 -> 0, 0.5 -> -1, 2.0 -> 2
        zoom_normalized = max(-1.0, min(1.0, zoom_normalized))

        # pan: -100 to +100 -> [-1, 1]
        pan_x_normalized = kf_data["pan_x"] / 100.0
        pan_y_normalized = kf_data["pan_y"] / 100.0
        pan_x_normalized = max(-1.0, min(1.0, pan_x_normalized))
        pan_y_normalized = max(-1.0, min(1.0, pan_y_normalized))

        # rotation: -45 to +45 -> [-1, 1]
        rotation_normalized = kf_data["rotation"] / 45.0
        rotation_normalized = max(-1.0, min(1.0, rotation_normalized))

        mask[:, 0, frame_idx, :, :] = zoom_normalized
        mask[:, 1, frame_idx, :, :] = pan_x_normalized
        mask[:, 2, frame_idx, :, :] = pan_y_normalized
        mask[:, 3, frame_idx, :, :] = rotation_normalized

    return mask * 1.0


def build_camera_params_from_json(json_str, num_frames):
    try:
        camera_data = json.loads(json_str)
        keyframes_list = camera_data.get("camera", {}).get("keyframes", [])
    except Exception:
        return None

    if not keyframes_list:
        return None

    kf_dict = {}
    for kf in keyframes_list:
        frame_idx = int(kf.get("frame", 0))
        kf_dict[frame_idx] = {
            "zoom": float(kf.get("zoom", 1.0)),
            "pan_x": float(kf.get("pan", [0, 0])[0]),
            "pan_y": float(kf.get("pan", [0, 0])[1]),
            "rotation": float(kf.get("rotation", 0)),
        }

    frame_indices = sorted(kf_dict.keys())
    if not frame_indices:
        return None

    params = []
    for frame_idx in range(num_frames):
        prev_idx = 0
        next_idx = num_frames - 1
        for idx in frame_indices:
            if idx <= frame_idx:
                prev_idx = idx
            if idx >= frame_idx and next_idx == num_frames - 1:
                next_idx = idx

        if prev_idx == next_idx:
            kf_data = kf_dict.get(prev_idx, {"zoom": 1.0, "pan_x": 0, "pan_y": 0, "rotation": 0})
        else:
            t = (frame_idx - prev_idx) / (next_idx - prev_idx)
            prev_kf = kf_dict.get(prev_idx, {"zoom": 1.0, "pan_x": 0, "pan_y": 0, "rotation": 0})
            next_kf = kf_dict.get(next_idx, {"zoom": 1.0, "pan_x": 0, "pan_y": 0, "rotation": 0})
            kf_data = {
                "zoom": prev_kf["zoom"] * (1 - t) + next_kf["zoom"] * t,
                "pan_x": prev_kf["pan_x"] * (1 - t) + next_kf["pan_x"] * t,
                "pan_y": prev_kf["pan_y"] * (1 - t) + next_kf["pan_y"] * t,
                "rotation": prev_kf["rotation"] * (1 - t) + next_kf["rotation"] * t,
            }

        params.append(kf_data)

    return params


def apply_camera_transform(image, zoom, pan_x, pan_y, rotation):
    w, h = image.size

    zoom = max(0.1, float(zoom))
    new_w = max(1, int(round(w * zoom)))
    new_h = max(1, int(round(h * zoom)))
    resized = image.resize((new_w, new_h), Image.BILINEAR)

    if zoom >= 1.0:
        left = (new_w - w) // 2
        top = (new_h - h) // 2
        cropped = resized.crop((left, top, left + w, top + h))
    else:
        cropped = Image.new("RGB", (w, h), (0, 0, 0))
        left = (w - new_w) // 2
        top = (h - new_h) // 2
        cropped.paste(resized, (left, top))

    rotated = cropped.rotate(rotation, resample=Image.BILINEAR, expand=False)

    shifted = Image.new("RGB", (w, h), (0, 0, 0))
    shifted.paste(rotated, (int(round(pan_x)), int(round(pan_y))))
    return shifted


def transform_point(x, y, params, width, height):
    cx = width / 2.0
    cy = height / 2.0
    dx = x - cx
    dy = y - cy

    zoom = params.get("zoom", 1.0)
    dx *= zoom
    dy *= zoom

    rot = np.deg2rad(params.get("rotation", 0.0))
    cos_r = np.cos(rot)
    sin_r = np.sin(rot)
    rx = dx * cos_r - dy * sin_r
    ry = dx * sin_r + dy * cos_r

    tx = rx + cx + params.get("pan_x", 0.0)
    ty = ry + cy + params.get("pan_y", 0.0)
    return tx, ty


def build_interpolated_bboxes(json_str, num_frames, height, width):
    if not json_str or not json_str.strip():
        return None

    bbox_data = json.loads(json_str)
    objects = bbox_data.get("objects", [])
    if not objects:
        return None

    all_boxes = []
    for obj in objects:
        frames = obj.get("frames", {})
        if not frames:
            continue

        keyframes = []
        for fi_str, bbox in frames.items():
            fi = int(fi_str)
            if fi >= num_frames:
                continue
            x1, y1, x2, y2 = bbox
            if all(0 <= v <= 1.0 for v in [x1, y1, x2, y2]):
                x1, x2 = x1 * width, x2 * width
                y1, y2 = y1 * height, y2 * height
            keyframes.append((fi, float(x1), float(y1), float(x2), float(y2)))

        if not keyframes:
            continue

        keyframes = sorted(keyframes, key=lambda x: x[0])
        per_frame = []
        for f in range(num_frames):
            if f <= keyframes[0][0]:
                per_frame.append(keyframes[0][1:])
                continue
            if f >= keyframes[-1][0]:
                per_frame.append(keyframes[-1][1:])
                continue
            for idx in range(len(keyframes) - 1):
                f0, x10, y10, x20, y20 = keyframes[idx]
                f1, x11, y11, x21, y21 = keyframes[idx + 1]
                if f0 <= f <= f1:
                    span = max(1, f1 - f0)
                    t = (f - f0) / span
                    x1 = x10 + (x11 - x10) * t
                    y1 = y10 + (y11 - y10) * t
                    x2 = x20 + (x21 - x20) * t
                    y2 = y20 + (y21 - y20) * t
                    per_frame.append((x1, y1, x2, y2))
                    break

        all_boxes.append(per_frame)

    if not all_boxes:
        return None
    return all_boxes


def build_point_tracks_from_json(json_str, num_frames, height, width):
    if not json_str or not json_str.strip():
        return None

    data = json.loads(json_str)
    points = data.get("points", [])
    if not points:
        return None

    tracks = []
    for pt in points:
        frames = pt.get("frames", {})
        if not frames:
            continue

        keyframes = []
        for fi_str, xy in frames.items():
            fi = int(fi_str)
            if fi >= num_frames:
                continue
            x, y = xy
            if 0 <= x <= 1.0 and 0 <= y <= 1.0:
                x = x * width
                y = y * height
            keyframes.append((fi, float(x), float(y)))

        if not keyframes:
            continue

        keyframes = sorted(keyframes, key=lambda x: x[0])
        per_frame = []
        for f in range(num_frames):
            if f <= keyframes[0][0]:
                per_frame.append(keyframes[0][1:])
                continue
            if f >= keyframes[-1][0]:
                per_frame.append(keyframes[-1][1:])
                continue
            for idx in range(len(keyframes) - 1):
                f0, x0, y0 = keyframes[idx]
                f1, x1, y1 = keyframes[idx + 1]
                if f0 <= f <= f1:
                    span = max(1, f1 - f0)
                    t = (f - f0) / span
                    per_frame.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
                    break

        tracks.append(per_frame)

    if not tracks:
        return None
    return tracks


def build_motion_signals_preview(
    input_image,
    camera_json_text,
    bbox_json_text,
    point_json_text,
    num_frames,
    height,
    width,
    fps,
):
    if input_image is None:
        return None

    base = input_image.resize((width, height)).convert("RGB")
    camera_params = build_camera_params_from_json(camera_json_text, num_frames)
    if camera_params is None:
        camera_params = [
            {"zoom": 1.0, "pan_x": 0.0, "pan_y": 0.0, "rotation": 0.0}
            for _ in range(num_frames)
        ]

    bbox_tracks = build_interpolated_bboxes(bbox_json_text, num_frames, height, width)
    point_tracks = build_point_tracks_from_json(point_json_text, num_frames, height, width)

    frames = []
    colors = [(255, 0, 0), (0, 255, 0), (0, 128, 255), (255, 128, 0)]
    for f in range(num_frames):
        params = camera_params[f]
        frame = apply_camera_transform(base, params["zoom"], params["pan_x"], params["pan_y"], params["rotation"])
        draw = ImageDraw.Draw(frame)

        if bbox_tracks is not None:
            for obj_idx, obj_frames in enumerate(bbox_tracks):
                if f >= len(obj_frames):
                    continue
                x1, y1, x2, y2 = obj_frames[f]
                p1 = transform_point(x1, y1, params, width, height)
                p2 = transform_point(x2, y2, params, width, height)
                x_min = max(0, min(width, int(round(min(p1[0], p2[0])))))
                y_min = max(0, min(height, int(round(min(p1[1], p2[1])))))
                x_max = max(0, min(width, int(round(max(p1[0], p2[0])))))
                y_max = max(0, min(height, int(round(max(p1[1], p2[1])))))
                draw.rectangle([x_min, y_min, x_max, y_max], outline=colors[obj_idx % len(colors)], width=3)

        if point_tracks is not None:
            for pt_idx, pt_frames in enumerate(point_tracks):
                if f >= len(pt_frames):
                    continue
                x, y = pt_frames[f]
                tx, ty = transform_point(x, y, params, width, height)
                r = 4
                draw.ellipse([tx - r, ty - r, tx + r, ty + r], fill=colors[pt_idx % len(colors)])

        frames.append(frame)

    if not frames:
        return None

    preview_path = os.path.join(tempfile.gettempdir(), "motion_signals_preview.mp4")
    save_video(frames, preview_path, fps=int(fps), quality=5)
    return preview_path


def generate_camera_json_from_sliders(zoom_start, pan_x_start, pan_y_start, rotation_start,
                                      zoom_mid, pan_x_mid, pan_y_mid, rotation_mid,
                                      zoom_end, pan_x_end, pan_y_end, rotation_end, num_frames):
    """从滑块值生成相机 JSON"""
    nf = int(num_frames)
    keyframes = [
        {"frame": 0, "zoom": zoom_start, "pan": [pan_x_start, pan_y_start], "rotation": rotation_start},
        {"frame": nf // 2, "zoom": zoom_mid, "pan": [pan_x_mid, pan_y_mid], "rotation": rotation_mid},
        {"frame": nf - 1, "zoom": zoom_end, "pan": [pan_x_end, pan_y_end], "rotation": rotation_end},
    ]
    return json.dumps({"camera": {"keyframes": keyframes}}, indent=2)


def preview_camera_motion(input_image, zoom_start, pan_x_start, pan_y_start, rotation_start,
                          zoom_mid, pan_x_mid, pan_y_mid, rotation_mid,
                          zoom_end, pan_x_end, pan_y_end, rotation_end, num_frames):
    """在输入图像上叠加绘制相机轨迹预览。显示视口矩形和轨迹线。"""
    if input_image is None:
        return None

    nf = int(num_frames)
    img = input_image.copy().convert("RGBA")
    w, h = img.size

    # 定义三个关键帧的相机参数
    keyframes = [
        (zoom_start, pan_x_start, pan_y_start, rotation_start, (76, 175, 255), "起始帧 (F0)"),
        (zoom_mid, pan_x_mid, pan_y_mid, rotation_mid, (255, 193, 7), f"中间帧 (F{nf // 2})"),
        (zoom_end, pan_x_end, pan_y_end, rotation_end, (244, 67, 54), f"结束帧 (F{nf - 1})"),
    ]

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    centers = []
    draw_overlay = ImageDraw.Draw(overlay)

    for zoom, pan_x, pan_y, rotation, color, label in keyframes:
        # 计算视口矩形
        # zoom: 1.0 = original, 0.5 = 2x zoomed out (larger rect), 2.0 = 2x zoomed in (smaller rect)
        viewport_w = w / zoom
        viewport_h = h / zoom

        # 计算视口左上角位置 (基于 pan 和中心)
        center_x = w / 2 + pan_x
        center_y = h / 2 + pan_y

        x1 = int(center_x - viewport_w / 2)
        y1 = int(center_y - viewport_h / 2)
        x2 = int(center_x + viewport_w / 2)
        y2 = int(center_y + viewport_h / 2)

        # 裁剪到图像边界
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        # 绘制半透明矩形
        fill_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        fd = ImageDraw.Draw(fill_layer)
        fd.rectangle([x1, y1, x2, y2], fill=(*color, 40))
        overlay = Image.alpha_composite(overlay, fill_layer)

        # 绘制矩形边框
        draw_overlay.rectangle([x1, y1, x2, y2], outline=(*color, 200), width=2)

        # 添加标签
        draw_overlay.text((x1 + 6, y1 + 6), label, fill=(*color, 255))

        # 记录中心点用于绘制轨迹线
        centers.append((center_x, center_y))

    # 绘制轨迹线连接各关键帧
    if len(centers) >= 2:
        for i in range(len(centers) - 1):
            draw_overlay.line([centers[i], centers[i + 1]],
                             fill=(255, 255, 255, 180), width=3)

    result = Image.alpha_composite(img, overlay)
    return result.convert("RGB")


# ==================== Video Generation ====================

def generate_video(
    prompt, negative_prompt,
    input_image, end_image,
    height, width, num_frames, num_inference_steps,
    cfg_scale, sigma_shift, seed, fps,
    bbox_mask_file, track_video_file, bbox_json_text, camera_json_text,
    progress=gr.Progress()
):
    if pipe_state["pipe"] is None:
        raise gr.Error("请先加载模型！")

    pipe = pipe_state["pipe"]
    torch_dtype = pipe_state["torch_dtype"]
    device = pipe.device

    bbox_mask = None
    if bbox_mask_file is not None:
        bbox_mask = torch.load(bbox_mask_file, map_location="cpu")
        bbox_mask = bbox_mask.to(dtype=torch_dtype, device=device)
    elif bbox_json_text and bbox_json_text.strip():
        try:
            bbox_mask = build_bbox_mask_from_json_str(
                bbox_json_text, int(num_frames), int(height), int(width)
            )
            bbox_mask = bbox_mask.to(dtype=torch_dtype, device=device)
        except Exception as e:
            raise gr.Error(f"Bbox JSON 解析失败: {e}")

    # 处理相机运动
    camera_mask = None
    if camera_json_text and camera_json_text.strip():
        try:
            camera_mask = build_camera_mask_from_json_str(
                camera_json_text, int(num_frames), int(height), int(width)
            )
            if camera_mask is not None:
                camera_mask = camera_mask.to(dtype=torch_dtype, device=device)
        except Exception as e:
            print(f"警告: 相机 JSON 解析失败: {e}")
            camera_mask = None

    debug_lines = []
    track_video = None
    if track_video_file is not None:
        track_video = torch.load(track_video_file, map_location="cpu")
        track_video = track_video.to(dtype=torch_dtype, device=device)
        debug_lines.append(
            f"track_video loaded: shape={tuple(track_video.shape)}, dtype={track_video.dtype}, device={track_video.device}"
        )

    if track_video is None and bbox_mask is not None:
        try:
            track_video = compute_track_video(
                pipe,
                torch_dtype,
                device,
                bbox_mask,
                bbox_json_text,
                input_image,
                end_image,
                num_frames,
                height,
                width,
            )
        except Exception as e:
            raise gr.Error(f"Track video 生成失败: {e}")

        if track_video is None:
            debug_lines.append("track_video skipped: video_rgb is None")
        else:
            debug_lines.append(
                f"track_video generated: shape={tuple(track_video.shape)}, dtype={track_video.dtype}, device={track_video.device}"
            )

    # 构建管道参数
    pipeline_kwargs = {
        "prompt": [prompt],
        "negative_prompt": negative_prompt,
        "input_image": input_image,
        "end_image": end_image,
        "num_inference_steps": int(num_inference_steps),
        "height": int(height),
        "width": int(width),
        "num_frames": int(num_frames),
        "cfg_scale": cfg_scale,
        "sigma_shift": sigma_shift,
        "seed": int(seed),
        "tiled": True,
        "tile_size": (30, 52),
        "tile_stride": (15, 26),
        "bbox_mask": bbox_mask,
        "track_video": track_video,
        "progress_bar_cmd": progress.tqdm,
    }

    video_frames = pipe(**pipeline_kwargs)

    if not video_frames or len(video_frames) == 0:
        raise gr.Error("生成失败，没有输出帧")

    output_path = os.path.join(tempfile.gettempdir(), "motioncanvas_output.mp4")
    save_video(video_frames[0], output_path, fps=int(fps), quality=5)
    track_preview_path = build_track_video_preview(track_video, input_image=input_image, fps=fps)
    if not debug_lines:
        debug_lines.append("track_video not provided and not generated")

    for line in debug_lines:
        print(line)

    return output_path, track_preview_path


def preview_track_video(
    input_image,
    end_image,
    height,
    width,
    num_frames,
    fps,
    bbox_mask_file,
    bbox_json_text,
):
    if pipe_state["pipe"] is None:
        raise gr.Error("请先加载模型！")

    pipe = pipe_state["pipe"]
    torch_dtype = pipe_state["torch_dtype"]
    device = pipe.device

    bbox_mask = None
    if bbox_mask_file is not None:
        bbox_mask = torch.load(bbox_mask_file, map_location="cpu")
        bbox_mask = bbox_mask.to(dtype=torch_dtype, device=device)
    elif bbox_json_text and bbox_json_text.strip():
        bbox_mask = build_bbox_mask_from_json_str(
            bbox_json_text, int(num_frames), int(height), int(width)
        )
        bbox_mask = bbox_mask.to(dtype=torch_dtype, device=device)

    track_video = compute_track_video(
        pipe,
        torch_dtype,
        device,
        bbox_mask,
        bbox_json_text,
        input_image,
        end_image,
        num_frames,
        height,
        width,
    )

    return build_track_video_preview(track_video, input_image=input_image, fps=fps)


def preview_motion_signals(
    input_image,
    camera_json_text,
    bbox_json_text,
    point_json_text,
    num_frames,
    height,
    width,
    fps,
):
    if input_image is None:
        raise gr.Error("请先上传输入图像！")

    return build_motion_signals_preview(
        input_image,
        camera_json_text,
        bbox_json_text,
        point_json_text,
        int(num_frames),
        int(height),
        int(width),
        fps,
    )


# ==================== UI ====================

with gr.Blocks(
    title="MotionCanvas",
) as app:

    gr.HTML(
        '<div class="header-banner">'
        "<h1>MotionCanvas</h1>"
        "<p>基于 WAN Video 的运动可控视频生成 · 支持 T2V / I2V 模式</p>"
        "</div>"
    )

    # ---- 模型配置 ----
    with gr.Accordion("模型配置", open=True):
        with gr.Row():
            with gr.Column(scale=1):
                dit_path = gr.Textbox(
                    label="DiT 模型路径",
                    value="/root/autodl-tmp/models/wan_1.3b/"
                          "diffusion_pytorch_model.safetensors",
                )
                vae_path = gr.Textbox(
                    label="VAE 模型路径",
                    value="/root/autodl-tmp/models/wan_1.3b/Wan2.1_VAE.pth",
                )
                text_encoder_path = gr.Textbox(
                    label="Text Encoder 路径",
                    value="/root/autodl-tmp/models/wan_1.3b/"
                          "models_t5_umt5-xxl-enc-bf16.pth",
                )
            with gr.Column(scale=1):
                image_encoder_path = gr.Textbox(
                    label="Image Encoder 路径（I2V 可选）",
                    value="/root/autodl-tmp/models/wan_1.3b/"
                          "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                )
                motion_controller_path = gr.Textbox(
                    label="Motion Controller 路径（可选）",
                    value="/root/autodl-tmp/models/DiffSynth-Studio/"
                          "Wan2.1-1.3b-speedcontrol-v1/model.safetensors",
                )
                vace_dir = gr.Textbox(
                    label="VACE 目录（可选）",
                    value="/root/autodl-tmp/models/iic/"
                          "VACE-Wan2.1-1.3B-Preview",
                )
                checkpoint_path = gr.Textbox(
                    label="MotionCanvas Checkpoint 路径",
                    value="/root/autodl-tmp/models/motioncanvas/model.pt",
                )
                dtype_choice = gr.Radio(
                    choices=["bfloat16", "float16"], value="bfloat16",
                    label="数据精度",
                )
        with gr.Row():
            load_btn = gr.Button("加载模型", variant="primary", scale=1)
            model_status = gr.Textbox(
                label="状态", value="尚未加载模型", interactive=False,
                elem_classes="status-box", scale=3,
            )
        load_btn.click(
            fn=load_models,
            inputs=[dit_path, vae_path, text_encoder_path,
                    image_encoder_path, motion_controller_path, vace_dir,
                    checkpoint_path, dtype_choice],
            outputs=model_status,
        )

    # ---- 主区域 ----
    with gr.Row():
        # ---- 左侧：提示词 + 参数 ----
        with gr.Column(scale=2, min_width=340):
            gr.Markdown("### 提示词", elem_classes="section-title")
            prompt = gr.Textbox(
                label="正面提示词", lines=3,
                placeholder="描述你想生成的视频内容...",
                value="A beautiful woman walking on the beach",
            )
            negative_prompt = gr.Textbox(
                label="负面提示词", lines=2, value=DEFAULT_NEGATIVE_PROMPT,
            )

            with gr.Accordion("输入图像（I2V 模式）", open=True):
                input_image = gr.Image(
                    label="起始帧图像", type="pil", sources=["upload"],
                )
                end_image = gr.Image(
                    label="结束帧图像（可选）", type="pil", sources=["upload"],
                )

            with gr.Accordion("生成参数", open=True):
                with gr.Row():
                    height = gr.Slider(
                        256, 1280, value=480, step=16, label="高度",
                    )
                    width = gr.Slider(
                        256, 1280, value=832, step=16, label="宽度",
                    )
                with gr.Row():
                    num_frames = gr.Slider(
                        5, 121, value=49, step=4, label="帧数",
                    )
                    fps = gr.Slider(
                        8, 30, value=15, step=1, label="输出 FPS",
                    )
                with gr.Row():
                    num_inference_steps = gr.Slider(
                        10, 100, value=50, step=1, label="推理步数",
                    )
                    cfg_scale = gr.Slider(
                        1.0, 15.0, value=5.0, step=0.1, label="CFG Scale",
                    )
                with gr.Row():
                    sigma_shift = gr.Slider(
                        1.0, 15.0, value=5.0, step=0.1, label="Sigma Shift",
                    )
                    seed = gr.Number(
                        value=42, label="随机种子", precision=0,
                    )

        # ---- 右侧：运动控制 + 输出 ----
        with gr.Column(scale=3, min_width=480):
            with gr.Accordion("运动控制", open=True):
                with gr.Tabs():
                    # ---- 可视化选区 Tab ----
                    with gr.Tab("可视化选区"):
                        gr.Markdown(
                            "上传输入图像后，在下方关键帧画布上**涂抹标记物体区域**，"
                            "系统会提取涂抹边界作为 Bbox。分别标记起始、中间、结束帧"
                            "中物体的位置，即可定义运动轨迹。"
                        )
                        sync_btn = gr.Button(
                            "同步输入图像到画布", size="sm", variant="secondary",
                        )

                        with gr.Tabs():
                            with gr.Tab("起始帧"):
                                gr.HTML(
                                    '<div class="kf-label kf-start">'
                                    '起始帧 (Frame 0) — 绿色笔刷</div>'
                                )
                                editor_start = gr.ImageEditor(
                                    canvas_size=(832, 480),
                                    sources=None,
                                    layers=False,
                                    interactive=True,
                                    image_mode="RGBA",
                                    brush=gr.Brush(
                                        default_size=40,
                                        default_color="#2ecc71",
                                        colors=["#2ecc71"],
                                    ),
                                    eraser=gr.Eraser(default_size=40),
                                    label="在此涂抹标记物体起始位置",
                                )

                            with gr.Tab("中间帧"):
                                gr.HTML(
                                    '<div class="kf-label kf-mid">'
                                    '中间帧 — 黄色笔刷</div>'
                                )
                                editor_mid = gr.ImageEditor(
                                    canvas_size=(832, 480),
                                    sources=None,
                                    layers=False,
                                    interactive=True,
                                    image_mode="RGBA",
                                    brush=gr.Brush(
                                        default_size=40,
                                        default_color="#f1c40f",
                                        colors=["#f1c40f"],
                                    ),
                                    eraser=gr.Eraser(default_size=40),
                                    label="在此涂抹标记物体中间位置",
                                )

                            with gr.Tab("结束帧"):
                                gr.HTML(
                                    '<div class="kf-label kf-end">'
                                    '结束帧 — 红色笔刷</div>'
                                )
                                editor_end = gr.ImageEditor(
                                    canvas_size=(832, 480),
                                    sources=None,
                                    layers=False,
                                    interactive=True,
                                    image_mode="RGBA",
                                    brush=gr.Brush(
                                        default_size=40,
                                        default_color="#e74c3c",
                                        colors=["#e74c3c"],
                                    ),
                                    eraser=gr.Eraser(default_size=40),
                                    label="在此涂抹标记物体结束位置",
                                )

                        with gr.Row():
                            extract_btn = gr.Button(
                                "提取选区 → 生成 JSON", variant="secondary",
                            )
                            preview_btn = gr.Button(
                                "预览运动路径", variant="secondary",
                            )
                        motion_preview = gr.Image(
                            label="运动路径预览", interactive=False,
                        )

                    # ---- 相机运动 Tab ----
                    with gr.Tab("相机运动"):
                        gr.Markdown(
                            "通过调整三个关键帧的相机参数来定义相机轨迹。\n"
                            "包括缩放（Zoom）、平移（Pan）和旋转（Rotation）。"
                        )

                        gr.Markdown("#### 起始帧相机参数", elem_classes="section-title")
                        with gr.Row():
                            camera_zoom_start = gr.Slider(
                                0.5, 2.0, value=1.0, step=0.1, label="缩放 (Zoom)",
                            )
                            camera_pan_x_start = gr.Slider(
                                -100, 100, value=0, step=5, label="平移 X (Pan X)",
                            )
                            camera_pan_y_start = gr.Slider(
                                -100, 100, value=0, step=5, label="平移 Y (Pan Y)",
                            )
                            camera_rotation_start = gr.Slider(
                                -45, 45, value=0, step=5, label="旋转 (°)",
                            )

                        gr.Markdown("#### 中间帧相机参数", elem_classes="section-title")
                        with gr.Row():
                            camera_zoom_mid = gr.Slider(
                                0.5, 2.0, value=1.0, step=0.1, label="缩放 (Zoom)",
                            )
                            camera_pan_x_mid = gr.Slider(
                                -100, 100, value=0, step=5, label="平移 X (Pan X)",
                            )
                            camera_pan_y_mid = gr.Slider(
                                -100, 100, value=0, step=5, label="平移 Y (Pan Y)",
                            )
                            camera_rotation_mid = gr.Slider(
                                -45, 45, value=0, step=5, label="旋转 (°)",
                            )

                        gr.Markdown("#### 结束帧相机参数", elem_classes="section-title")
                        with gr.Row():
                            camera_zoom_end = gr.Slider(
                                0.5, 2.0, value=1.0, step=0.1, label="缩放 (Zoom)",
                            )
                            camera_pan_x_end = gr.Slider(
                                -100, 100, value=0, step=5, label="平移 X (Pan X)",
                            )
                            camera_pan_y_end = gr.Slider(
                                -100, 100, value=0, step=5, label="平移 Y (Pan Y)",
                            )
                            camera_rotation_end = gr.Slider(
                                -45, 45, value=0, step=5, label="旋转 (°)",
                            )

                        with gr.Row():
                            camera_extract_btn = gr.Button(
                                "生成相机 JSON", variant="secondary",
                            )
                            camera_preview_btn = gr.Button(
                                "预览相机轨迹", variant="secondary",
                            )
                        camera_motion_preview = gr.Image(
                            label="相机轨迹预览", interactive=False,
                        )

                    # ---- JSON / 高级 Tab ----
                    with gr.Tab("JSON / 高级选项"):
                        gr.Markdown("#### 物体运动")
                        bbox_json_text = gr.Code(
                            label="Bbox JSON（可由可视化选区自动生成，也可手动编辑）",
                            language="json",
                            value="",
                            lines=12,
                        )

                        gr.Markdown("#### 相机运动")
                        camera_json_text = gr.Code(
                            label="相机 JSON（可由相机运动 Tab 自动生成，也可手动编辑）",
                            language="json",
                            value="",
                            lines=12,
                        )

                        gr.Markdown("#### 局部运动（点轨迹）")
                        point_json_text = gr.Code(
                            label="Point Trajectory JSON",
                            language="json",
                            value=(
                                "{\n"
                                "  \"points\": [\n"
                                "    {\n"
                                "      \"frames\": {\n"
                                "        \"0\": [0.5, 0.5],\n"
                                "        \"24\": [0.6, 0.55],\n"
                                "        \"48\": [0.7, 0.6]\n"
                                "      }\n"
                                "    }\n"
                                "  ]\n"
                                "}"
                            ),
                            lines=10,
                        )

                        gr.Markdown("#### 高级输入")
                        with gr.Row():
                            bbox_mask_file = gr.File(
                                label="Bbox Mask (.pt)", file_types=[".pt"],
                            )
                            track_video_file = gr.File(
                                label="Track Video (.pt)", file_types=[".pt"],
                            )

            generate_btn = gr.Button(
                "生成视频", variant="primary", size="lg",
                elem_classes="generate-btn",
            )
            output_video = gr.Video(label="生成结果", interactive=False)
            track_preview = gr.Video(label="Track Video 预览", interactive=False)
            motion_signals_preview = gr.Video(label="Motion Signals 预览", interactive=False)
            track_preview_btn = gr.Button("预览 Track Video", variant="secondary")
            motion_preview_btn = gr.Button("预览 Motion Signals", variant="secondary")

    # ---- 事件绑定 ----

    input_image.change(
        fn=sync_image_to_editors,
        inputs=[input_image],
        outputs=[editor_start, editor_mid, editor_end],
    )

    sync_btn.click(
        fn=sync_image_to_editors,
        inputs=[input_image],
        outputs=[editor_start, editor_mid, editor_end],
    )

    extract_btn.click(
        fn=generate_bbox_json_from_editors,
        inputs=[editor_start, editor_mid, editor_end, num_frames],
        outputs=[bbox_json_text],
    ).then(
        fn=preview_motion_path,
        inputs=[input_image, editor_start, editor_mid, editor_end, num_frames],
        outputs=[motion_preview],
    )

    preview_btn.click(
        fn=preview_motion_path,
        inputs=[input_image, editor_start, editor_mid, editor_end, num_frames],
        outputs=[motion_preview],
    )

    # ---- 相机运动事件绑定 ----
    camera_extract_btn.click(
        fn=generate_camera_json_from_sliders,
        inputs=[
            camera_zoom_start, camera_pan_x_start, camera_pan_y_start, camera_rotation_start,
            camera_zoom_mid, camera_pan_x_mid, camera_pan_y_mid, camera_rotation_mid,
            camera_zoom_end, camera_pan_x_end, camera_pan_y_end, camera_rotation_end,
            num_frames,
        ],
        outputs=[camera_json_text],
    ).then(
        fn=preview_camera_motion,
        inputs=[
            input_image,
            camera_zoom_start, camera_pan_x_start, camera_pan_y_start, camera_rotation_start,
            camera_zoom_mid, camera_pan_x_mid, camera_pan_y_mid, camera_rotation_mid,
            camera_zoom_end, camera_pan_x_end, camera_pan_y_end, camera_rotation_end,
            num_frames,
        ],
        outputs=[camera_motion_preview],
    )

    camera_preview_btn.click(
        fn=preview_camera_motion,
        inputs=[
            input_image,
            camera_zoom_start, camera_pan_x_start, camera_pan_y_start, camera_rotation_start,
            camera_zoom_mid, camera_pan_x_mid, camera_pan_y_mid, camera_rotation_mid,
            camera_zoom_end, camera_pan_x_end, camera_pan_y_end, camera_rotation_end,
            num_frames,
        ],
        outputs=[camera_motion_preview],
    )

    generate_btn.click(
        fn=generate_video,
        inputs=[
            prompt, negative_prompt,
            input_image, end_image,
            height, width, num_frames, num_inference_steps,
            cfg_scale, sigma_shift, seed, fps,
            bbox_mask_file, track_video_file, bbox_json_text, camera_json_text,
        ],
        outputs=[output_video, track_preview],
    )

    track_preview_btn.click(
        fn=preview_track_video,
        inputs=[
            input_image, end_image,
            height, width, num_frames, fps,
            bbox_mask_file, bbox_json_text,
        ],
        outputs=track_preview,
    )

    motion_preview_btn.click(
        fn=preview_motion_signals,
        inputs=[
            input_image,
            camera_json_text,
            bbox_json_text,
            point_json_text,
            num_frames,
            height,
            width,
            fps,
        ],
        outputs=motion_signals_preview,
    )


if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=6006,
        share=False,
        theme=gr.themes.Soft(
            primary_hue="violet",
            secondary_hue="indigo",
            neutral_hue="slate",
        ),
        css=CUSTOM_CSS,
    )
