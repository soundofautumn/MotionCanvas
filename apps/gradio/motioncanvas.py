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

pipe_state = {"pipe": None, "torch_dtype": None, "loaded_config": None}


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
                checkpoint_path, dtype_str):
    config_key = (f"{dit_path}|{vae_path}|{text_encoder_path}|"
                  f"{image_encoder_path}|{checkpoint_path}|{dtype_str}")
    if pipe_state["loaded_config"] == config_key and pipe_state["pipe"] is not None:
        return "✅ 模型已加载（缓存命中）"

    torch_dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for p, name in [(dit_path, "DiT"), (vae_path, "VAE"),
                    (text_encoder_path, "Text Encoder")]:
        if not p or not os.path.exists(p):
            return f"❌ {name} 路径无效: {p}"

    model_paths = [text_encoder_path, vae_path, dit_path]
    if image_encoder_path and os.path.exists(image_encoder_path):
        model_paths.append(image_encoder_path)

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


# ==================== Video Generation ====================

def generate_video(
    prompt, negative_prompt,
    input_image, end_image,
    height, width, num_frames, num_inference_steps,
    cfg_scale, sigma_shift, seed, fps,
    bbox_mask_file, track_video_file, bbox_json_text,
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

    track_video = None
    if track_video_file is not None:
        track_video = torch.load(track_video_file, map_location="cpu")
        track_video = track_video.to(dtype=torch_dtype, device=device)

    video_frames = pipe(
        prompt=[prompt],
        negative_prompt=negative_prompt,
        input_image=input_image,
        end_image=end_image,
        num_inference_steps=int(num_inference_steps),
        height=int(height),
        width=int(width),
        num_frames=int(num_frames),
        cfg_scale=cfg_scale,
        sigma_shift=sigma_shift,
        seed=int(seed),
        tiled=True,
        tile_size=(30, 52),
        tile_stride=(15, 26),
        bbox_mask=bbox_mask,
        track_video=track_video,
        progress_bar_cmd=progress.tqdm,
    )

    if not video_frames or len(video_frames) == 0:
        raise gr.Error("生成失败，没有输出帧")

    output_path = os.path.join(tempfile.gettempdir(), "motioncanvas_output.mp4")
    save_video(video_frames[0], output_path, fps=int(fps), quality=5)
    return output_path


# ==================== UI ====================

with gr.Blocks(
    title="MotionCanvas",
    theme=gr.themes.Soft(
        primary_hue="violet",
        secondary_hue="indigo",
        neutral_hue="slate",
    ),
    css=CUSTOM_CSS,
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
                    image_encoder_path, checkpoint_path, dtype_choice],
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

            with gr.Accordion("输入图像（I2V 模式）", open=False):
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

                    # ---- JSON / 高级 Tab ----
                    with gr.Tab("JSON / 高级选项"):
                        bbox_json_text = gr.Code(
                            label="Bbox JSON（可由可视化选区自动生成，也可手动编辑）",
                            language="json",
                            value="",
                            lines=12,
                        )
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

    generate_btn.click(
        fn=generate_video,
        inputs=[
            prompt, negative_prompt,
            input_image, end_image,
            height, width, num_frames, num_inference_steps,
            cfg_scale, sigma_shift, seed, fps,
            bbox_mask_file, track_video_file, bbox_json_text,
        ],
        outputs=output_video,
    )


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=6006, share=False)
