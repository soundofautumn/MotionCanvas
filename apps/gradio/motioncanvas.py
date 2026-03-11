"""
MotionCanvas Gradio GUI
基于 Gradio 的 MotionCanvas 视频生成界面
"""

import os
import sys
import tempfile
import torch
import numpy as np
from PIL import Image

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

pipe_state = {"pipe": None, "torch_dtype": None, "loaded_config": None}


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


def load_models(dit_path, vae_path, text_encoder_path, image_encoder_path, checkpoint_path, dtype_str):
    config_key = f"{dit_path}|{vae_path}|{text_encoder_path}|{image_encoder_path}|{checkpoint_path}|{dtype_str}"
    if pipe_state["loaded_config"] == config_key and pipe_state["pipe"] is not None:
        return "✅ 模型已加载（缓存命中）"

    torch_dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for p, name in [(dit_path, "DiT"), (vae_path, "VAE"), (text_encoder_path, "Text Encoder")]:
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


def build_bbox_mask_from_json_str(json_str, num_frames, height, width):
    import json
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

    # bbox_mask
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

    # track_video
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
    theme=gr.themes.Soft(),
    css="""
    .header { text-align: center; margin-bottom: 8px; }
    .status-box textarea { font-size: 14px !important; }
    """
) as app:

    gr.Markdown(
        """
        # MotionCanvas — 视频生成控制台
        基于 WAN Video 的运动可控视频生成。加载模型后，输入提示词即可生成视频。
        支持 **T2V**（文本生视频）和 **I2V**（图像生视频）两种模式。
        """,
        elem_classes="header"
    )

    # ---- 模型加载 ----
    with gr.Accordion("🔧 模型配置", open=True):
        with gr.Row():
            with gr.Column(scale=1):
                dit_path = gr.Textbox(
                    label="DiT 模型路径",
                    value="/root/autodl-tmp/models/wan_1.3b/diffusion_pytorch_model.safetensors",
                )
                vae_path = gr.Textbox(
                    label="VAE 模型路径",
                    value="/root/autodl-tmp/models/wan_1.3b/Wan2.1_VAE.pth",
                )
                text_encoder_path = gr.Textbox(
                    label="Text Encoder 路径",
                    value="/root/autodl-tmp/models/wan_1.3b/models_t5_umt5-xxl-enc-bf16.pth",
                )
            with gr.Column(scale=1):
                image_encoder_path = gr.Textbox(
                    label="Image Encoder 路径（I2V 可选）",
                    value="/root/autodl-tmp/models/wan_1.3b/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                )
                checkpoint_path = gr.Textbox(
                    label="MotionCanvas Checkpoint 路径",
                    value="/root/autodl-tmp/models/motioncanvas/model.pt",
                )
                dtype_choice = gr.Radio(
                    choices=["bfloat16", "float16"], value="bfloat16", label="数据精度"
                )
        with gr.Row():
            load_btn = gr.Button("📦 加载模型", variant="primary", scale=1)
            model_status = gr.Textbox(
                label="状态", value="尚未加载模型", interactive=False,
                elem_classes="status-box", scale=3
            )
        load_btn.click(
            fn=load_models,
            inputs=[dit_path, vae_path, text_encoder_path, image_encoder_path, checkpoint_path, dtype_choice],
            outputs=model_status
        )

    with gr.Row():
        # ---- 左侧：参数 ----
        with gr.Column(scale=2, min_width=360):
            with gr.Accordion("📝 提示词", open=True):
                prompt = gr.Textbox(
                    label="正面提示词", lines=3, placeholder="描述你想生成的视频内容...",
                    value="A beautiful woman walking on the beach"
                )
                negative_prompt = gr.Textbox(
                    label="负面提示词", lines=2, value=DEFAULT_NEGATIVE_PROMPT
                )

            with gr.Accordion("🖼️ 输入图像（I2V 模式，可选）", open=False):
                input_image = gr.Image(
                    label="起始帧图像", type="pil", sources=["upload"],
                )
                end_image = gr.Image(
                    label="结束帧图像（可选）", type="pil", sources=["upload"],
                )

            with gr.Accordion("⚙️ 生成参数", open=True):
                with gr.Row():
                    height = gr.Slider(
                        minimum=256, maximum=1280, value=480, step=16,
                        label="高度", interactive=True
                    )
                    width = gr.Slider(
                        minimum=256, maximum=1280, value=832, step=16,
                        label="宽度", interactive=True
                    )
                with gr.Row():
                    num_frames = gr.Slider(
                        minimum=5, maximum=121, value=49, step=4,
                        label="帧数", interactive=True
                    )
                    fps = gr.Slider(
                        minimum=8, maximum=30, value=15, step=1,
                        label="输出 FPS", interactive=True
                    )
                with gr.Row():
                    num_inference_steps = gr.Slider(
                        minimum=10, maximum=100, value=50, step=1,
                        label="推理步数", interactive=True
                    )
                    cfg_scale = gr.Slider(
                        minimum=1.0, maximum=15.0, value=5.0, step=0.1,
                        label="CFG Scale", interactive=True
                    )
                with gr.Row():
                    sigma_shift = gr.Slider(
                        minimum=1.0, maximum=15.0, value=5.0, step=0.1,
                        label="Sigma Shift", interactive=True
                    )
                    seed = gr.Number(
                        value=42, label="随机种子", precision=0, interactive=True
                    )

            with gr.Accordion("🎯 MotionCanvas 运动控制（可选）", open=False):
                gr.Markdown(
                    "上传预计算的 `.pt` 张量文件，或在下方输入 bbox JSON 来控制物体运动。"
                )
                bbox_mask_file = gr.File(
                    label="Bbox Mask (.pt)", file_types=[".pt"],
                )
                track_video_file = gr.File(
                    label="Track Video (.pt)", file_types=[".pt"],
                )
                bbox_json_text = gr.Code(
                    label="Bbox JSON（替代 .pt 文件）",
                    language="json",
                    value="""{
  "objects": [
    {
      "frames": {
        "0":  [0.1, 0.3, 0.4, 0.8],
        "24": [0.5, 0.3, 0.8, 0.8],
        "48": [0.6, 0.2, 0.9, 0.7]
      }
    }
  ]
}""",
                    lines=12,
                )

        # ---- 右侧：生成与结果 ----
        with gr.Column(scale=3, min_width=480):
            generate_btn = gr.Button(
                "🚀 生成视频", variant="primary", size="lg"
            )
            output_video = gr.Video(label="生成结果", interactive=False)

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
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
