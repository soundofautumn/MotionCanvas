"""
MotionCanvas Gradio GUI
基于 Gradio 的 MotionCanvas 视频生成界面
"""

import os
import sys
import tempfile
import json
import math
import re
import torch
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import gradio as gr
from diffsynth import ModelManager, save_video
from diffsynth.pipelines.wan_video_motioncanvas import WanVideoPipeline_motioncanvas
from diffsynth.pipelines.tracker_utils import get_video_track_video, create_pos_feature_map

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

DEFAULT_DOWNSAMPLE_RATIOS = [4, 8, 8]
DEFAULT_POS_EMB_DIM = 16


# ==================== LLM (OpenAI-Compatible) ====================

LLM_SYSTEM_PROMPT = """你是一个 MotionCanvas 的动作/参数编辑助手。

你必须只输出一个 JSON 对象（不要输出 Markdown，不要输出代码块，不要输出多余文本）。

优先输出“更上层指令 ops”，让程序把 ops 转换成关键帧 state 与 JSON；
除非用户明确要求你直接输出 bbox_json / point_json / camera_json。

JSON 结构：
{
  "assistant_message": "给用户的简短说明（可选）",
    "ops": [
        {"op": "camera.zoom_linear", "start_frame": 0, "end_frame": 48, "start": 1.0, "end": 1.6},
        {"op": "camera.pan_linear", "start_frame": 0, "end_frame": 48, "start": [0, 0], "end": [30, 0]},
        {"op": "camera.rotation_linear", "start_frame": 0, "end_frame": 48, "start": 0, "end": 15},
        {"op": "camera.set", "frame": 10, "zoom": 1.2, "pan": [10, 0], "rotation": 0},
        {"op": "bbox.translate", "object": 0, "start_frame": 0, "end_frame": 48, "dx": 0.05, "dy": 0.0, "space": "norm"},
        {"op": "points.translate", "start_frame": 0, "end_frame": 48, "dx": 0.03, "dy": 0.0, "space": "norm"}
    ],
  "updates": {
    "bbox_json": "...可选，字符串或对象，符合Bbox JSON格式...",
    "point_json": "...可选，字符串或对象，符合Point JSON格式...",
    "camera_json": "...可选，字符串或对象，符合Camera JSON格式...",
    "prompt": "...可选...",
    "negative_prompt": "...可选...",
    "height": 480,
    "width": 832,
    "num_frames": 49,
    "fps": 15,
    "num_inference_steps": 50,
    "cfg_scale": 5.0,
    "sigma_shift": 5.0,
    "seed": 42
  }
}

规则：
- 只在你非常确定时才改动；不需要改动就省略对应字段。
- ops 更推荐；若使用 ops，帧索引必须在 [0, num_frames-1] 范围内。
- bbox_json / point_json / camera_json 若提供，必须是严格可解析的 JSON（对象或字符串均可）。
- num_frames/width/height 等需给出合理范围内的值。
"""


def _snap_to_step(value, minimum, step, maximum):
    try:
        v = float(value)
    except Exception:
        return None
    v = max(float(minimum), min(float(maximum), v))
    if step and step > 0:
        k = round((v - float(minimum)) / float(step))
        v = float(minimum) + k * float(step)
        v = max(float(minimum), min(float(maximum), v))
    return v


def _extract_json_object(text):
    if text is None:
        raise ValueError("空响应")
    text = str(text).strip()
    if not text:
        raise ValueError("空响应")

    # 直接是 JSON
    try:
        return json.loads(text)
    except Exception:
        pass

    # 尝试从 ```json ... ``` 中提取
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, flags=re.IGNORECASE)
    if m:
        return json.loads(m.group(1))

    # 尝试从首个 {...} 提取（贪婪会吞太多，这里做最小匹配）
    m2 = re.search(r"(\{[\s\S]*\})", text)
    if m2:
        return json.loads(m2.group(1))

    raise ValueError("无法从模型输出中解析 JSON")


def _ensure_json_text(value):
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        return s
    # dict/list → 转成格式化 JSON
    return json.dumps(value, ensure_ascii=False, indent=2)


def _normalize_openai_base_url(base_url):
    s = (base_url or "").strip()
    if not s:
        raise ValueError("base_url 不能为空")
    # 允许用户填 https://api.deepseek.com 或 https://api.deepseek.com/v1
    s = s.rstrip("/")
    return s


def _openai_chat_complete(
    base_url,
    api_key,
    model,
    messages,
    temperature=0.2,
    timeout=60,
    force_json=True,
):
    base = _normalize_openai_base_url(base_url)
    # OpenAI SDK 默认 base_url 类似 https://api.openai.com/v1
    if not base.endswith("/v1"):
        base = base + "/v1"

    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError(
            "未安装 OpenAI Python SDK（openai）。请先安装：pip install openai"
        ) from e

    client = OpenAI(
        base_url=base,
        api_key=(str(api_key).strip() if api_key is not None else ""),
        timeout=float(timeout),
    )

    kwargs = {
        "model": (model or "").strip(),
        "messages": messages,
        "temperature": float(temperature),
    }
    if force_json:
        kwargs["response_format"] = {"type": "json_object"}

    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception:
        # 兼容部分后端不支持 response_format
        if force_json and "response_format" in kwargs:
            kwargs.pop("response_format", None)
            resp = client.chat.completions.create(**kwargs)
        else:
            raise

    # 返回与旧结构一致的 dict，方便下游处理与测试 mock
    try:
        dumped = resp.model_dump()
        return dumped
    except Exception:
        # 兜底：手动拼装
        content = None
        try:
            content = resp.choices[0].message.content
        except Exception:
            content = ""
        return {"choices": [{"message": {"content": content}}]}


def _bbox_state_from_json_text(bbox_json_text):
    if not bbox_json_text or not str(bbox_json_text).strip():
        return {}
    data = json.loads(bbox_json_text)
    objects = data.get("objects", [])
    if not objects:
        return {}
    frames = objects[0].get("frames", {})
    out = {}
    for fi_str, bbox in (frames or {}).items():
        out[str(int(fi_str))] = bbox
    return out


def _point_state_from_json_text(point_json_text):
    if not point_json_text or not str(point_json_text).strip():
        return {}
    data = json.loads(point_json_text)
    points = data.get("points", [])
    if not points:
        return {}
    frame_to_points = {}
    for pt in points:
        frames = pt.get("frames", {})
        for fi_str, xy in (frames or {}).items():
            fi = str(int(fi_str))
            frame_to_points.setdefault(fi, []).append(xy)
    # 保持每帧点的顺序稳定（按出现顺序）
    return frame_to_points


def _camera_state_from_json_text(camera_json_text):
    if not camera_json_text or not str(camera_json_text).strip():
        return {}
    data = json.loads(camera_json_text)
    keyframes = data.get("camera", {}).get("keyframes", [])
    if not keyframes:
        return {}
    out = {}
    for kf in keyframes:
        fi = str(int(kf.get("frame", 0)))
        pan = kf.get("pan", [0.0, 0.0])
        out[fi] = {
            "zoom": float(kf.get("zoom", 1.0)),
            "pan_x": float(pan[0] if isinstance(pan, list) and len(pan) > 0 else 0.0),
            "pan_y": float(pan[1] if isinstance(pan, list) and len(pan) > 1 else 0.0),
            "rotation": float(kf.get("rotation", 0.0)),
        }
    return out


def _clamp01(v):
    return max(0.0, min(1.0, float(v)))


def _interp_bbox_norm_for_frame(frames, frame_idx):
    items = []
    for fi_str, bbox in (frames or {}).items():
        try:
            fi = int(fi_str)
        except Exception:
            continue
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [float(x) for x in bbox]
        items.append((fi, x1, y1, x2, y2))
    if not items:
        return None

    items = sorted(items, key=lambda x: x[0])
    if frame_idx <= items[0][0]:
        _, x1, y1, x2, y2 = items[0]
        return [x1, y1, x2, y2]
    if frame_idx >= items[-1][0]:
        _, x1, y1, x2, y2 = items[-1]
        return [x1, y1, x2, y2]

    for idx in range(len(items) - 1):
        f0, x10, y10, x20, y20 = items[idx]
        f1, x11, y11, x21, y21 = items[idx + 1]
        if f0 <= frame_idx <= f1:
            span = max(1, f1 - f0)
            t = (frame_idx - f0) / span
            return [
                _lerp(x10, x11, t),
                _lerp(y10, y11, t),
                _lerp(x20, x21, t),
                _lerp(y20, y21, t),
            ]
    return None


def apply_ops_to_states(
    ops,
    bbox_state,
    point_state,
    camera_state,
    num_frames,
    width,
    height,
    bbox_json_text="",
    point_json_text="",
    camera_json_text="",
):
    nf = int(num_frames)
    w = int(width)
    h = int(height)

    bbox_state = dict(bbox_state or {})
    point_state = dict(point_state or {})
    camera_state = dict(camera_state or {})

    # 若 UI state 为空，但 JSON 有值，优先从 JSON 同步一份作为基线
    if (not bbox_state) and bbox_json_text and str(bbox_json_text).strip():
        try:
            bbox_state = _bbox_state_from_json_text(bbox_json_text)
        except Exception:
            bbox_state = dict(bbox_state or {})
    if (not point_state) and point_json_text and str(point_json_text).strip():
        try:
            point_state = _point_state_from_json_text(point_json_text)
        except Exception:
            point_state = dict(point_state or {})
    if (not camera_state) and camera_json_text and str(camera_json_text).strip():
        try:
            camera_state = _camera_state_from_json_text(camera_json_text)
        except Exception:
            camera_state = dict(camera_state or {})

    def _cap_frame(v, default_v=0):
        try:
            vv = int(v)
        except Exception:
            vv = int(default_v)
        return max(0, min(max(0, nf - 1), vv))

    ops_list = ops if isinstance(ops, list) else []
    for item in ops_list:
        if not isinstance(item, dict):
            continue
        op = str(item.get("op", "")).strip()
        if not op:
            continue

        if op == "camera.set":
            f = _cap_frame(item.get("frame", 0), 0)
            fi = str(int(f))
            cur = dict(camera_state.get(fi) or {})
            if "zoom" in item and item.get("zoom") is not None:
                cur["zoom"] = float(item.get("zoom"))
            if "pan" in item and item.get("pan") is not None:
                pan = item.get("pan")
                if isinstance(pan, (list, tuple)) and len(pan) >= 2:
                    cur["pan_x"] = float(pan[0])
                    cur["pan_y"] = float(pan[1])
            if "rotation" in item and item.get("rotation") is not None:
                cur["rotation"] = float(item.get("rotation"))
            camera_state[fi] = {
                "zoom": float(cur.get("zoom", 1.0)),
                "pan_x": float(cur.get("pan_x", 0.0)),
                "pan_y": float(cur.get("pan_y", 0.0)),
                "rotation": float(cur.get("rotation", 0.0)),
            }
            continue

        if op in {"camera.zoom_linear", "camera.pan_linear", "camera.rotation_linear"}:
            sf = _cap_frame(item.get("start_frame", 0), 0)
            ef = _cap_frame(item.get("end_frame", nf - 1), nf - 1)
            if ef < sf:
                sf, ef = ef, sf

            sfi = str(int(sf))
            efi = str(int(ef))
            s_cur = dict(camera_state.get(sfi) or {})
            e_cur = dict(camera_state.get(efi) or {})

            if op == "camera.zoom_linear":
                s_cur["zoom"] = float(item.get("start", s_cur.get("zoom", 1.0)))
                e_cur["zoom"] = float(item.get("end", e_cur.get("zoom", 1.0)))
            elif op == "camera.pan_linear":
                sp = item.get("start", [s_cur.get("pan_x", 0.0), s_cur.get("pan_y", 0.0)])
                ep = item.get("end", [e_cur.get("pan_x", 0.0), e_cur.get("pan_y", 0.0)])
                if isinstance(sp, (list, tuple)) and len(sp) >= 2:
                    s_cur["pan_x"], s_cur["pan_y"] = float(sp[0]), float(sp[1])
                if isinstance(ep, (list, tuple)) and len(ep) >= 2:
                    e_cur["pan_x"], e_cur["pan_y"] = float(ep[0]), float(ep[1])
            else:
                s_cur["rotation"] = float(item.get("start", s_cur.get("rotation", 0.0)))
                e_cur["rotation"] = float(item.get("end", e_cur.get("rotation", 0.0)))

            camera_state[sfi] = {
                "zoom": float(s_cur.get("zoom", 1.0)),
                "pan_x": float(s_cur.get("pan_x", 0.0)),
                "pan_y": float(s_cur.get("pan_y", 0.0)),
                "rotation": float(s_cur.get("rotation", 0.0)),
            }
            camera_state[efi] = {
                "zoom": float(e_cur.get("zoom", 1.0)),
                "pan_x": float(e_cur.get("pan_x", 0.0)),
                "pan_y": float(e_cur.get("pan_y", 0.0)),
                "rotation": float(e_cur.get("rotation", 0.0)),
            }
            continue

        if op == "bbox.translate":
            space = str(item.get("space", "norm")).strip().lower()
            dx = float(item.get("dx", 0.0))
            dy = float(item.get("dy", 0.0))
            if space in {"px", "pixel", "pixels"}:
                dx = dx / max(1.0, float(w))
                dy = dy / max(1.0, float(h))

            sf = _cap_frame(item.get("start_frame", 0), 0)
            ef = _cap_frame(item.get("end_frame", nf - 1), nf - 1)
            if ef < sf:
                sf, ef = ef, sf

            # 用原始 state 做插值与更新，避免对同一帧重复平移
            orig_bbox_state = dict(bbox_state or {})
            existing_frames = {str(int(k)) for k in orig_bbox_state.keys()}
            start_bbox = _interp_bbox_norm_for_frame(orig_bbox_state, sf)
            end_bbox = _interp_bbox_norm_for_frame(orig_bbox_state, ef)
            if start_bbox is None or end_bbox is None:
                continue

            def _shift(bb):
                x1, y1, x2, y2 = [float(x) for x in bb]
                x1, x2 = _clamp01(x1 + dx), _clamp01(x2 + dx)
                y1, y2 = _clamp01(y1 + dy), _clamp01(y2 + dy)
                if x2 <= x1:
                    x2 = _clamp01(x1 + 1e-4)
                if y2 <= y1:
                    y2 = _clamp01(y1 + 1e-4)
                return [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]

            # 先平移区间内已有关键帧（每帧最多一次）
            for fi_str, bb in orig_bbox_state.items():
                try:
                    fi = int(fi_str)
                except Exception:
                    continue
                if sf <= fi <= ef:
                    bbox_state[fi_str] = _shift(bb)

            # 再补齐 start/end（如果原来没有对应关键帧）
            sfi = str(int(sf))
            efi = str(int(ef))
            if sfi not in existing_frames:
                bbox_state[sfi] = _shift(start_bbox)
            if efi not in existing_frames:
                bbox_state[efi] = _shift(end_bbox)
            continue

        if op == "points.translate":
            space = str(item.get("space", "norm")).strip().lower()
            dx = float(item.get("dx", 0.0))
            dy = float(item.get("dy", 0.0))
            if space in {"px", "pixel", "pixels"}:
                dx = dx / max(1.0, float(w))
                dy = dy / max(1.0, float(h))

            sf = _cap_frame(item.get("start_frame", 0), 0)
            ef = _cap_frame(item.get("end_frame", nf - 1), nf - 1)
            if ef < sf:
                sf, ef = ef, sf

            for fi_str, pts in list(point_state.items()):
                try:
                    fi = int(fi_str)
                except Exception:
                    continue
                if not (sf <= fi <= ef):
                    continue
                if not isinstance(pts, list):
                    continue
                new_pts = []
                for xy in pts:
                    if not isinstance(xy, (list, tuple)) or len(xy) < 2:
                        continue
                    x, y = float(xy[0]), float(xy[1])
                    new_pts.append([round(_clamp01(x + dx), 4), round(_clamp01(y + dy), 4)])
                point_state[fi_str] = new_pts
            continue

    return bbox_state, point_state, camera_state


def llm_apply_instruction(
    user_message,
    chat_history,
    llm_base_url,
    llm_api_key,
    llm_model,
    llm_timeout,
    bbox_json_text,
    camera_json_text,
    point_json_text,
    prompt,
    negative_prompt,
    height,
    width,
    num_frames,
    fps,
    num_inference_steps,
    cfg_scale,
    sigma_shift,
    seed,
    motion_frame_idx,
    bbox_kf_state,
    point_kf_state,
    camera_kf_state,
):
    user_message = (user_message or "").strip()
    if not user_message:
        raise gr.Error("请输入你的要求")

    history = list(chat_history or [])

    # 组装 messages：系统提示 + 历史对话 + 本次用户输入（附带当前状态）
    messages = [{"role": "system", "content": LLM_SYSTEM_PROMPT}]
    for u, a in history:
        if u:
            messages.append({"role": "user", "content": str(u)})
        if a:
            messages.append({"role": "assistant", "content": str(a)})

    state_blob = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "height": int(height),
        "width": int(width),
        "num_frames": int(num_frames),
        "fps": int(fps),
        "num_inference_steps": int(num_inference_steps),
        "cfg_scale": float(cfg_scale),
        "sigma_shift": float(sigma_shift),
        "seed": int(seed),
        "bbox_json": bbox_json_text or "",
        "camera_json": camera_json_text or "",
        "point_json": point_json_text or "",
        "current_frame_idx": int(motion_frame_idx),
    }

    user_payload = (
        "用户需求：\n"
        + user_message
        + "\n\n当前状态（可作为你生成 ops / updates 的依据）：\n"
        + json.dumps(state_blob, ensure_ascii=False)
    )
    messages.append({"role": "user", "content": user_payload})

    try:
        resp = _openai_chat_complete(
            base_url=llm_base_url,
            api_key=llm_api_key,
            model=(llm_model or "").strip(),
            messages=messages,
            temperature=0.2,
            timeout=float(llm_timeout),
            force_json=True,
        )
        content = resp["choices"][0]["message"]["content"]
        obj = _extract_json_object(content)
    except Exception as e:
        history.append((user_message, f"❌ LLM 调用失败：{e}"))
        return (
            history,
            bbox_json_text,
            point_json_text,
            camera_json_text,
            bbox_kf_state,
            point_kf_state,
            camera_kf_state,
            prompt,
            negative_prompt,
            height,
            width,
            num_frames,
            fps,
            num_inference_steps,
            cfg_scale,
            sigma_shift,
            seed,
            gr.update(),
            "LLM 调用失败（见对话）",
            "",
        )

    updates = obj.get("updates", {}) if isinstance(obj, dict) else {}
    ops = obj.get("ops", []) if isinstance(obj, dict) else []
    assistant_msg = obj.get("assistant_message") if isinstance(obj, dict) else None

    new_bbox_json = bbox_json_text
    new_point_json = point_json_text
    new_camera_json = camera_json_text
    new_prompt = prompt
    new_negative_prompt = negative_prompt
    new_height = height
    new_width = width
    new_num_frames = num_frames
    new_fps = fps
    new_steps = num_inference_steps
    new_cfg = cfg_scale
    new_sigma = sigma_shift
    new_seed = seed

    # 基线：先基于现有 state/json 应用 ops（若有），再应用 updates（兼容旧格式）
    try:
        base_bbox_state = dict(bbox_kf_state or {})
        base_point_state = dict(point_kf_state or {})
        base_camera_state = dict(camera_kf_state or {})

        if ops and isinstance(ops, list):
            base_bbox_state, base_point_state, base_camera_state = apply_ops_to_states(
                ops,
                base_bbox_state,
                base_point_state,
                base_camera_state,
                num_frames=int(num_frames),
                width=int(width),
                height=int(height),
                bbox_json_text=bbox_json_text or "",
                point_json_text=point_json_text or "",
                camera_json_text=camera_json_text or "",
            )
            new_bbox_state = base_bbox_state
            new_point_state = base_point_state
            new_camera_state = base_camera_state
            new_bbox_json = _bbox_state_to_json(new_bbox_state)
            new_point_json = _point_state_to_json(new_point_state)
            new_camera_json = _camera_state_to_json(new_camera_state)

        if "bbox_json" in updates:
            new_bbox_json = _ensure_json_text(updates.get("bbox_json")) or ""
            if new_bbox_json.strip():
                json.loads(new_bbox_json)

        if "point_json" in updates:
            new_point_json = _ensure_json_text(updates.get("point_json")) or ""
            if new_point_json.strip():
                json.loads(new_point_json)

        if "camera_json" in updates:
            new_camera_json = _ensure_json_text(updates.get("camera_json")) or ""
            if new_camera_json.strip():
                json.loads(new_camera_json)

        if "prompt" in updates and updates.get("prompt") is not None:
            new_prompt = str(updates.get("prompt"))
        if "negative_prompt" in updates and updates.get("negative_prompt") is not None:
            new_negative_prompt = str(updates.get("negative_prompt"))

        if "height" in updates:
            snapped = _snap_to_step(updates.get("height"), 256, 16, 1280)
            if snapped is not None:
                new_height = int(snapped)
        if "width" in updates:
            snapped = _snap_to_step(updates.get("width"), 256, 16, 1280)
            if snapped is not None:
                new_width = int(snapped)
        if "num_frames" in updates:
            snapped = _snap_to_step(updates.get("num_frames"), 5, 4, 121)
            if snapped is not None:
                new_num_frames = int(snapped)
        if "fps" in updates:
            snapped = _snap_to_step(updates.get("fps"), 8, 1, 30)
            if snapped is not None:
                new_fps = int(snapped)
        if "num_inference_steps" in updates:
            snapped = _snap_to_step(updates.get("num_inference_steps"), 10, 1, 100)
            if snapped is not None:
                new_steps = int(snapped)
        if "cfg_scale" in updates:
            val = float(updates.get("cfg_scale"))
            new_cfg = max(1.0, min(15.0, val))
        if "sigma_shift" in updates:
            val = float(updates.get("sigma_shift"))
            new_sigma = max(1.0, min(15.0, val))
        if "seed" in updates:
            new_seed = int(updates.get("seed"))

        # 同步 State（保证后续“保存/删除关键帧”不和 JSON 脱节）
        # 如果 ops 已生成过 state，则只有在 updates 覆盖了 JSON 时才需要重建
        if not (ops and isinstance(ops, list)) or any(
            k in updates for k in ("bbox_json", "point_json", "camera_json")
        ):
            new_bbox_state = _bbox_state_from_json_text(new_bbox_json)
            new_point_state = _point_state_from_json_text(new_point_json)
            new_camera_state = _camera_state_from_json_text(new_camera_json)
    except Exception as e:
        history.append((user_message, f"❌ 解析/应用更新失败：{e}"))
        return (
            history,
            bbox_json_text,
            point_json_text,
            camera_json_text,
            bbox_kf_state,
            point_kf_state,
            camera_kf_state,
            prompt,
            negative_prompt,
            height,
            width,
            num_frames,
            fps,
            num_inference_steps,
            cfg_scale,
            sigma_shift,
            seed,
            gr.update(),
            "LLM 输出不合法（未应用）",
            "",
        )

    msg = assistant_msg or "✅ 已应用更新"
    history.append((user_message, msg))

    # 如果帧数变化，更新全局帧滑条范围
    nf = int(new_num_frames)
    max_frame = max(0, nf - 1)
    cur_frame = int(motion_frame_idx)
    new_frame_val = min(cur_frame, max_frame)
    frame_update = gr.update(minimum=0, maximum=max_frame, value=new_frame_val)

    return (
        history,
        new_bbox_json,
        new_point_json,
        new_camera_json,
        new_bbox_state,
        new_point_state,
        new_camera_state,
        new_prompt,
        new_negative_prompt,
        new_height,
        new_width,
        new_num_frames,
        new_fps,
        new_steps,
        new_cfg,
        new_sigma,
        new_seed,
        frame_update,
        "✅ 已应用 LLM 更新",
        "",
    )


def llm_clear_chat():
    return [], ""


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


def _read_json_file(file_obj):
    if file_obj is None:
        return None
    if isinstance(file_obj, str):
        path = file_obj
    elif isinstance(file_obj, dict) and "name" in file_obj:
        path = file_obj["name"]
    else:
        path = getattr(file_obj, "name", None)
    if not path or not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def _lerp(a, b, t):
    return a * (1.0 - t) + b * t


def _euler_yaw_pitch_roll_to_matrix(yaw_deg, pitch_deg, roll_deg):
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)

    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)

    r_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    r_x = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float32)
    r_z = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    return r_z @ r_x @ r_y


def _interpolate_keyframes(keyframes, num_frames, fields):
    if not keyframes:
        return []
    keyframes_sorted = sorted(keyframes, key=lambda k: int(k.get("frame", 0)))
    frames_out = []
    for frame_idx in range(num_frames):
        prev_kf = keyframes_sorted[0]
        next_kf = keyframes_sorted[-1]
        for kf in keyframes_sorted:
            if int(kf.get("frame", 0)) <= frame_idx:
                prev_kf = kf
            if int(kf.get("frame", 0)) >= frame_idx:
                next_kf = kf
                break

        f0 = int(prev_kf.get("frame", 0))
        f1 = int(next_kf.get("frame", 0))
        t = 0.0 if f0 == f1 else (frame_idx - f0) / max(1, f1 - f0)

        out = {"frame": frame_idx}
        for field in fields:
            v0 = prev_kf.get(field)
            v1 = next_kf.get(field)
            if isinstance(v0, list) and isinstance(v1, list):
                out[field] = [_lerp(v0[i], v1[i], t) for i in range(len(v0))]
            else:
                out[field] = _lerp(float(v0), float(v1), t)
        frames_out.append(out)
    return frames_out


def _project_world_to_pixel(point_w, cam_pos, cam_rot, fx, fy, cx, cy):
    p_cam = cam_rot.T @ (point_w - cam_pos)
    z = p_cam[2]
    if z <= 1e-6:
        return None
    x = (p_cam[0] / z) * fx + cx
    y = (p_cam[1] / z) * fy + cy
    return x, y, z



def _interpolate_object_frames(obj, num_frames):
    keyframes = obj.get("keyframes", [])
    return _interpolate_keyframes(keyframes, num_frames, ["center", "size"])


def _project_object_bbox(obj_frames, camera_poses, width, height, fx, fy, cx, cy):
    frames_out = {}
    for frame in obj_frames:
        f = int(frame["frame"])
        center = np.array(frame.get("center", [0.0, 0.0, 0.0]), dtype=np.float32)
        size = np.array(frame.get("size", [1.0, 1.0, 1.0]), dtype=np.float32)
        half = size * 0.5

        corners = []
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    corners.append(center + half * np.array([sx, sy, sz], dtype=np.float32))

        cam = camera_poses[f]
        pts_2d = []
        for c in corners:
            proj = _project_world_to_pixel(c, cam["pos"], cam["rot_m"], fx, fy, cx, cy)
            if proj is not None:
                pts_2d.append((proj[0], proj[1]))

        if not pts_2d:
            continue

        xs = [p[0] for p in pts_2d]
        ys = [p[1] for p in pts_2d]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        x1 = max(0.0, min(width - 1.0, x1))
        y1 = max(0.0, min(height - 1.0, y1))
        x2 = max(0.0, min(width - 1.0, x2))
        y2 = max(0.0, min(height - 1.0, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        frames_out[str(f)] = [x1 / width, y1 / height, x2 / width, y2 / height]

    return frames_out


def _interpolate_point_keyframes(keyframes, num_frames):
    kf_items = sorted([(int(k), v) for k, v in keyframes.items()], key=lambda x: x[0])
    if not kf_items:
        return []

    tracks = []
    for frame_idx in range(num_frames):
        prev_kf = kf_items[0]
        next_kf = kf_items[-1]
        for kf in kf_items:
            if kf[0] <= frame_idx:
                prev_kf = kf
            if kf[0] >= frame_idx:
                next_kf = kf
                break

        f0 = prev_kf[0]
        f1 = next_kf[0]
        t = 0.0 if f0 == f1 else (frame_idx - f0) / max(1, f1 - f0)
        v0 = prev_kf[1]
        v1 = next_kf[1]
        tracks.append([_lerp(v0[i], v1[i], t) for i in range(len(v0))])

    return tracks


def _object_frame_lookup(objects_json, num_frames):
    lookup = {}
    for obj in objects_json.get("objects", []):
        obj_id = obj.get("id")
        if not obj_id:
            continue
        lookup[obj_id] = _interpolate_object_frames(obj, num_frames)
    return lookup


def _point_world_from_object_local(local_xyz, obj_frame):
    center = np.array(obj_frame.get("center", [0.0, 0.0, 0.0]), dtype=np.float32)
    size = np.array(obj_frame.get("size", [1.0, 1.0, 1.0]), dtype=np.float32)
    local = np.array(local_xyz, dtype=np.float32) - 0.5
    return center + local * size


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


def build_video_rgb_from_bbox_motion(input_image, bbox_json_text, camera_json_text, num_frames, height, width):
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
    camera_params = build_camera_params_from_json(camera_json_text, num_frames)
    if camera_params is None:
        camera_params = [
            {"zoom": 1.0, "pan_x": 0.0, "pan_y": 0.0, "rotation": 0.0}
            for _ in range(num_frames)
        ]

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
        params = camera_params[f]
        warped = apply_camera_transform(
            shifted,
            params["zoom"],
            params["pan_x"],
            params["pan_y"],
            params["rotation"],
        )
        frames_out.append(torch.from_numpy(np.array(warped)).permute(2, 0, 1))

    return torch.stack(frames_out, dim=0)


def compute_track_video(
    pipe,
    torch_dtype,
    device,
    bbox_mask,
    bbox_json_text,
    camera_json_text,
    point_json_text,
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

    point_tracks = build_point_tracks_from_json(point_json_text, int(num_frames), int(height), int(width))
    if point_tracks is not None:
        point_masks = build_point_masks_from_tracks(point_tracks, int(num_frames), int(height), int(width), radius=6)
        if point_masks is not None:
            object_masks = torch.cat([object_masks, point_masks], dim=0)

    reference_imgs_indicator = [object_masks.shape[0]]
    video_rgb = build_video_rgb_from_bbox_motion(
        input_image, bbox_json_text, camera_json_text, int(num_frames), int(height), int(width)
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


def extract_points_from_editor(editor_data, max_points=20):
    if editor_data is None:
        return []
    layers = editor_data.get("layers", [])
    if not layers:
        return []

    points = []
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
        coords = np.argwhere(alpha > 0)
        if coords.size == 0:
            continue
        if coords.shape[0] > max_points:
            idx = np.linspace(0, coords.shape[0] - 1, max_points).astype(int)
            coords = coords[idx]
        for y, x in coords:
            points.append((float(x), float(y)))

    return points[:max_points]


def sync_image_to_editors(input_image):
    """将输入图像同步到画布作为背景。"""
    if input_image is None:
        return None, None
    img = np.array(input_image)
    return img, img


def _frame_slider_updates(num_frames):
    nf = int(num_frames)
    max_v = max(0, nf - 1)
    upd = gr.update(minimum=0, maximum=max_v, value=0)
    return upd


def _bbox_state_to_json(bbox_state):
    frames = {}
    for k, v in (bbox_state or {}).items():
        frames[str(int(k))] = v
    frames = dict(sorted(frames.items(), key=lambda kv: int(kv[0])))
    if not frames:
        return ""
    return json.dumps({"objects": [{"frames": frames}]}, indent=2)


def save_bbox_keyframe(bbox_editor_data, frame_idx, bbox_state):
    bbox = extract_bbox_from_editor(bbox_editor_data)
    state = dict(bbox_state or {})
    fi = str(int(frame_idx))
    if bbox is None:
        if fi in state:
            del state[fi]
    else:
        state[fi] = bbox
    return state, _bbox_state_to_json(state)


def delete_bbox_keyframe(frame_idx, bbox_state):
    state = dict(bbox_state or {})
    fi = str(int(frame_idx))
    if fi in state:
        del state[fi]
    return state, _bbox_state_to_json(state)


def _extract_points_norm_from_editor(editor_data, max_points=20):
    if editor_data is None:
        return []
    layers = editor_data.get("layers", [])
    if not layers:
        return []

    points = []
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

        h, w = alpha.shape[:2]
        coords = np.argwhere(alpha > 0)
        if coords.size == 0:
            continue
        if coords.shape[0] > max_points:
            idx = np.linspace(0, coords.shape[0] - 1, max_points).astype(int)
            coords = coords[idx]
        for y, x in coords:
            if w <= 0 or h <= 0:
                continue
            points.append((round(float(x) / w, 4), round(float(y) / h, 4)))

    return points[:max_points]


def _point_state_to_json(point_state):
    state = {str(int(k)): v for k, v in (point_state or {}).items()}
    state = dict(sorted(state.items(), key=lambda kv: int(kv[0])))
    if not state:
        return ""

    max_len = 0
    for pts in state.values():
        if isinstance(pts, list):
            max_len = max(max_len, len(pts))

    tracks = []
    for idx in range(max_len):
        frames = {}
        for fi_str, pts in state.items():
            if idx < len(pts):
                frames[fi_str] = list(pts[idx])
        if frames:
            tracks.append({"frames": frames})

    if not tracks:
        return ""
    return json.dumps({"points": tracks}, indent=2)


def save_point_keyframe(point_editor_data, frame_idx, point_state):
    pts = _extract_points_norm_from_editor(point_editor_data, max_points=20)
    state = dict(point_state or {})
    fi = str(int(frame_idx))
    if not pts:
        if fi in state:
            del state[fi]
    else:
        state[fi] = pts
    return state, _point_state_to_json(state)


def delete_point_keyframe(frame_idx, point_state):
    state = dict(point_state or {})
    fi = str(int(frame_idx))
    if fi in state:
        del state[fi]
    return state, _point_state_to_json(state)


def _camera_state_to_json(camera_state):
    state = {str(int(k)): v for k, v in (camera_state or {}).items()}
    state = dict(sorted(state.items(), key=lambda kv: int(kv[0])))
    if not state:
        return ""
    keyframes = []
    for fi_str, params in state.items():
        keyframes.append(
            {
                "frame": int(fi_str),
                "zoom": float(params.get("zoom", 1.0)),
                "pan": [float(params.get("pan_x", 0.0)), float(params.get("pan_y", 0.0))],
                "rotation": float(params.get("rotation", 0.0)),
            }
        )
    return json.dumps({"camera": {"keyframes": keyframes}}, indent=2)


def load_camera_keyframe(frame_idx, camera_state):
    state = dict(camera_state or {})
    fi = str(int(frame_idx))
    params = state.get(fi)
    if not isinstance(params, dict):
        return 1.0, 0.0, 0.0, 0.0
    return (
        float(params.get("zoom", 1.0)),
        float(params.get("pan_x", 0.0)),
        float(params.get("pan_y", 0.0)),
        float(params.get("rotation", 0.0)),
    )


def save_camera_keyframe(frame_idx, zoom, pan_x, pan_y, rotation, camera_state):
    state = dict(camera_state or {})
    fi = str(int(frame_idx))
    state[fi] = {
        "zoom": float(zoom),
        "pan_x": float(pan_x),
        "pan_y": float(pan_y),
        "rotation": float(rotation),
    }
    return state, _camera_state_to_json(state)


def delete_camera_keyframe(frame_idx, camera_state):
    state = dict(camera_state or {})
    fi = str(int(frame_idx))
    if fi in state:
        del state[fi]
    return state, _camera_state_to_json(state)




def generate_model_params_from_ui(
    input_image,
    end_image,
    num_frames,
    height,
    width,
    bbox_json_text,
    camera_json_text,
    point_json_text,
):
    if not bbox_json_text or not bbox_json_text.strip():
        raise gr.Error("请先提供 Bbox JSON")

    bbox_mask = build_bbox_mask_from_json_str(
        bbox_json_text, int(num_frames), int(height), int(width)
    )
    bbox_mask_path = os.path.join(tempfile.gettempdir(), "motioncanvas_bbox_mask_ui.pt")
    torch.save(bbox_mask, bbox_mask_path)

    camera_params = build_camera_params_from_json(camera_json_text, int(num_frames))
    if camera_params is None:
        camera_params = [
            {"zoom": 1.0, "pan_x": 0.0, "pan_y": 0.0, "rotation": 0.0}
            for _ in range(int(num_frames))
        ]

    background_tracks = generate_background_tracks(
        camera_params,
        int(num_frames),
        int(height),
        int(width),
        bbox_mask=bbox_mask,
        grid_size=14,
    )

    local_tracks = build_point_tracks_from_json(point_json_text, int(num_frames), int(height), int(width))
    camera_applied_tracks = []
    if local_tracks is not None:
        for track in local_tracks:
            cam_track = []
            for f, (x, y) in enumerate(track):
                params = camera_params[f]
                tx, ty = apply_camera_transform_to_point(
                    x,
                    y,
                    int(width),
                    int(height),
                    params["zoom"],
                    params["pan_x"],
                    params["pan_y"],
                    params["rotation"],
                )
                cam_track.append((tx, ty))
            camera_applied_tracks.append(cam_track)

    all_tracks = background_tracks + camera_applied_tracks
    track_video = build_track_video_from_tracks(all_tracks, int(num_frames), int(height), int(width))

    track_video_path = None
    if track_video is not None:
        track_video_path = os.path.join(tempfile.gettempdir(), "motioncanvas_track_video_ui.pt")
        torch.save(track_video, track_video_path)
        status = "✅ 已生成 bbox_mask 和 track_video"
    else:
        status = "⚠️ 已生成 bbox_mask，但 track_video 为空（缺少轨迹）"

    return bbox_mask_path, track_video_path, status


def preview_control_overlay(
    input_image,
    num_frames,
    height,
    width,
    bbox_json_text,
    camera_json_text,
    point_json_text,
    frame_idx=0,
):
    if input_image is None:
        raise gr.Error("请先上传输入图像")

    base = input_image.resize((int(width), int(height))).convert("RGB")
    draw = ImageDraw.Draw(base)

    try:
        bbox_data = json.loads(bbox_json_text) if bbox_json_text else {"objects": []}
    except Exception as e:
        raise gr.Error(f"Bbox JSON 解析失败: {e}")

    for obj in bbox_data.get("objects", []):
        frames = obj.get("frames", {})
        bbox = _interp_bbox_for_frame(frames, frame_idx, int(width), int(height))
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline=(255, 80, 80), width=3)

    camera_params = build_camera_params_from_json(camera_json_text, int(num_frames))
    if camera_params is None:
        camera_params = [
            {"zoom": 1.0, "pan_x": 0.0, "pan_y": 0.0, "rotation": 0.0}
            for _ in range(int(num_frames))
        ]

    bbox_mask = None
    if bbox_json_text and bbox_json_text.strip():
        bbox_mask = build_bbox_mask_from_json_str(
            bbox_json_text, int(num_frames), int(height), int(width)
        )

    bg_tracks = generate_background_tracks(
        camera_params,
        int(num_frames),
        int(height),
        int(width),
        bbox_mask=bbox_mask,
        grid_size=14,
    )

    local_tracks = build_point_tracks_from_json(point_json_text, int(num_frames), int(height), int(width))
    camera_applied = []
    if local_tracks is not None:
        for track in local_tracks:
            if frame_idx < len(track):
                x, y = track[frame_idx]
            else:
                x, y = track[0]
            params = camera_params[frame_idx] if frame_idx < len(camera_params) else camera_params[0]
            tx, ty = apply_camera_transform_to_point(
                x,
                y,
                int(width),
                int(height),
                params["zoom"],
                params["pan_x"],
                params["pan_y"],
                params["rotation"],
            )
            camera_applied.append((tx, ty))

    for track in bg_tracks:
        if frame_idx < len(track):
            x, y = track[frame_idx]
        else:
            x, y = track[0]
        draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(80, 160, 255))

    for x, y in camera_applied:
        draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(255, 80, 80))

    return base


def preview_control_video(
    input_image,
    num_frames,
    height,
    width,
    bbox_json_text,
    camera_json_text,
    point_json_text,
    fps=15,
):
    if input_image is None:
        raise gr.Error("请先上传输入图像")

    num_frames = int(num_frames)
    height = int(height)
    width = int(width)
    fps = int(fps)

    base_frame = input_image.resize((width, height)).convert("RGB")

    try:
        bbox_data = json.loads(bbox_json_text) if bbox_json_text else {"objects": []}
    except Exception as e:
        raise gr.Error(f"Bbox JSON 解析失败: {e}")

    camera_params = build_camera_params_from_json(camera_json_text, num_frames)
    if camera_params is None:
        camera_params = [
            {"zoom": 1.0, "pan_x": 0.0, "pan_y": 0.0, "rotation": 0.0}
            for _ in range(num_frames)
        ]

    bbox_mask = None
    if bbox_json_text and bbox_json_text.strip():
        bbox_mask = build_bbox_mask_from_json_str(bbox_json_text, num_frames, height, width)

    bg_tracks = generate_background_tracks(
        camera_params,
        num_frames,
        height,
        width,
        bbox_mask=bbox_mask,
        grid_size=14,
    )

    local_tracks = build_point_tracks_from_json(point_json_text, num_frames, height, width)

    frames = []
    for frame_idx in range(num_frames):
        img = base_frame.copy()
        draw = ImageDraw.Draw(img)

        for obj in bbox_data.get("objects", []):
            obj_frames = obj.get("frames", {})
            bbox = _interp_bbox_for_frame(obj_frames, frame_idx, width, height)
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            draw.rectangle([x1, y1, x2, y2], outline=(255, 80, 80), width=3)

        for track in bg_tracks:
            if frame_idx < len(track):
                x, y = track[frame_idx]
            else:
                x, y = track[0]
            draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(80, 160, 255))

        if local_tracks is not None:
            params = camera_params[frame_idx] if frame_idx < len(camera_params) else camera_params[0]
            for track in local_tracks:
                if frame_idx < len(track):
                    x, y = track[frame_idx]
                else:
                    x, y = track[0]
                tx, ty = apply_camera_transform_to_point(
                    x,
                    y,
                    width,
                    height,
                    params["zoom"],
                    params["pan_x"],
                    params["pan_y"],
                    params["rotation"],
                )
                draw.ellipse([tx - 3, ty - 3, tx + 3, ty + 3], fill=(255, 80, 80))

        frames.append(img)

    tmp = tempfile.NamedTemporaryFile(prefix="motioncanvas_preview_", suffix=".mp4", delete=False)
    tmp_path = tmp.name
    tmp.close()
    save_video(frames, tmp_path, fps=max(1, fps), quality=5)
    return tmp_path

# ==================== Camera Motion Control ====================

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


def _interp_bbox_for_frame(frames, frame_idx, width, height):
    keyframes = []
    for fi_str, bbox in frames.items():
        fi = int(fi_str)
        x1, y1, x2, y2 = bbox
        if all(0 <= v <= 1.0 for v in [x1, y1, x2, y2]):
            x1, x2 = x1 * width, x2 * width
            y1, y2 = y1 * height, y2 * height
        keyframes.append((fi, float(x1), float(y1), float(x2), float(y2)))

    if not keyframes:
        return None

    keyframes = sorted(keyframes, key=lambda x: x[0])
    if frame_idx <= keyframes[0][0]:
        _, x1, y1, x2, y2 = keyframes[0]
    elif frame_idx >= keyframes[-1][0]:
        _, x1, y1, x2, y2 = keyframes[-1]
    else:
        x1 = y1 = x2 = y2 = None
        for idx in range(len(keyframes) - 1):
            f0, x10, y10, x20, y20 = keyframes[idx]
            f1, x11, y11, x21, y21 = keyframes[idx + 1]
            if f0 <= frame_idx <= f1:
                span = max(1, f1 - f0)
                t = (frame_idx - f0) / span
                x1 = x10 + (x11 - x10) * t
                y1 = y10 + (y11 - y10) * t
                x2 = x20 + (x21 - x20) * t
                y2 = y20 + (y21 - y20) * t
                break
        if x1 is None:
            return None

    x1 = max(0.0, min(width - 1.0, x1))
    y1 = max(0.0, min(height - 1.0, y1))
    x2 = max(0.0, min(width - 1.0, x2))
    y2 = max(0.0, min(height - 1.0, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


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


def apply_camera_transform_to_point(x, y, width, height, zoom, pan_x, pan_y, rotation):
    cx = width / 2.0
    cy = height / 2.0
    dx = x - cx
    dy = y - cy
    dx *= zoom
    dy *= zoom
    theta = math.radians(rotation)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    rx = dx * cos_t - dy * sin_t
    ry = dx * sin_t + dy * cos_t
    return rx + cx + pan_x, ry + cy + pan_y


def generate_background_tracks(camera_params, num_frames, height, width, bbox_mask=None, grid_size=14):
    xs = np.linspace(0, width - 1, grid_size)
    ys = np.linspace(0, height - 1, grid_size)
    points = [(float(x), float(y)) for y in ys for x in xs]

    if bbox_mask is not None:
        mask = (bbox_mask[0, :, 0] > 0).any(dim=0).cpu().numpy()
        points = [p for p in points if not mask[int(round(p[1])), int(round(p[0]))]]

    tracks = []
    for x, y in points:
        track = []
        for f in range(num_frames):
            params = camera_params[f]
            tx, ty = apply_camera_transform_to_point(
                x,
                y,
                width,
                height,
                params["zoom"],
                params["pan_x"],
                params["pan_y"],
                params["rotation"],
            )
            track.append((tx, ty))
        tracks.append(track)
    return tracks


def build_track_video_from_tracks(tracks, num_frames, height, width):
    if not tracks:
        return None
    n = len(tracks)
    pred_tracks = torch.full((1, num_frames, n, 2), -1.0, dtype=torch.float32)
    pred_visibility = torch.zeros((1, num_frames, n), dtype=torch.bool)

    for i, track in enumerate(tracks):
        for f, (x, y) in enumerate(track):
            if 0 <= x < width and 0 <= y < height:
                pred_tracks[0, f, i, 0] = float(x)
                pred_tracks[0, f, i, 1] = float(y)
                pred_visibility[0, f, i] = True

    track_video, _ = create_pos_feature_map(
        pred_tracks,
        pred_visibility,
        DEFAULT_DOWNSAMPLE_RATIOS,
        height,
        width,
        DEFAULT_POS_EMB_DIM,
        track_num=-1,
        t_down_strategy="sample",
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    return track_video.permute(0, 4, 1, 2, 3)


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


def build_point_masks_from_tracks(point_tracks, num_frames, height, width, radius=6):
    if not point_tracks:
        return None

    masks = []
    for track in point_tracks:
        mask = torch.zeros(num_frames, 1, height, width, dtype=torch.bool)
        for f in range(min(num_frames, len(track))):
            x, y = track[f]
            cx = int(round(x))
            cy = int(round(y))
            if cx < 0 or cy < 0 or cx >= width or cy >= height:
                continue
            x0 = max(0, cx - radius)
            x1 = min(width - 1, cx + radius)
            y0 = max(0, cy - radius)
            y1 = min(height - 1, cy + radius)
            for yy in range(y0, y1 + 1):
                for xx in range(x0, x1 + 1):
                    if (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2:
                        mask[f, 0, yy, xx] = True
        masks.append(mask)

    return torch.stack(masks, dim=0)


def _reset_editor_canvas(input_image):
    if input_image is None:
        return None
    return np.array(input_image)




# ==================== Video Generation ====================

def generate_video(
    prompt, negative_prompt,
    input_image, end_image,
    height, width, num_frames, num_inference_steps,
    cfg_scale, sigma_shift, seed, fps,
    bbox_mask_file, track_video_file, bbox_json_text, camera_json_text, point_json_text,
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
                camera_json_text,
                point_json_text,
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
    if not debug_lines:
        debug_lines.append("track_video not provided and not generated")

    for line in debug_lines:
        print(line)

    return output_path


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
                image_encoder_path = gr.Textbox(
                    label="Image Encoder 路径（I2V 可选）",
                    value="/root/autodl-tmp/models/wan_1.3b/"
                          "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                )
            with gr.Column(scale=1):
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
                gr.Markdown(
                    "**流程**：用户在 UI 中设置相机/物体/局部运动 → 生成 2D 控制信号 → "
                    "编码为模型参数（bbox_mask/track_video） → 视频生成。"
                )
                with gr.Tabs():
                    # ---- 运动编辑（合并）Tab ----
                    with gr.Tab("运动编辑"):
                        gr.Markdown(
                            "在同一个页面里编辑三种运动：**物体Bbox**、**局部点**、**相机**。\n"
                            "用各自的帧滑条选择当前帧 → 编辑 → 保存为关键帧；系统会自动生成对应 JSON。"
                        )

                        sync_btn = gr.Button(
                            "同步输入图像到画布", size="sm", variant="secondary",
                        )

                        with gr.Row():
                            motion_frame_idx = gr.Slider(
                                minimum=0,
                                maximum=48,
                                value=0,
                                step=1,
                                label="当前编辑帧 (全局)",
                                interactive=True,
                            )

                        gr.Markdown("### 物体运动（Bbox）", elem_classes="section-title")
                        bbox_kf_state = gr.State({})
                        bbox_editor = gr.ImageEditor(
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
                            label="在此涂抹标记物体区域（保存为该帧关键帧）",
                        )
                        with gr.Row():
                            bbox_save_btn = gr.Button("保存当前帧选区", variant="secondary")
                            bbox_delete_btn = gr.Button("删除当前帧选区", variant="secondary")

                        gr.Markdown("### 局部运动（点轨迹）", elem_classes="section-title")
                        point_kf_state = gr.State({})
                        point_editor = gr.ImageEditor(
                            canvas_size=(832, 480),
                            sources=None,
                            layers=False,
                            interactive=True,
                            image_mode="RGBA",
                            brush=gr.Brush(
                                default_size=10,
                                default_color="#ffffff",
                                colors=["#ffffff"],
                            ),
                            eraser=gr.Eraser(default_size=10),
                            label="在此标记局部点（保存为该帧关键帧）",
                        )
                        with gr.Row():
                            point_save_btn = gr.Button("保存当前帧点", variant="secondary")
                            point_delete_btn = gr.Button("删除当前帧点", variant="secondary")

                        gr.Markdown("### 相机运动", elem_classes="section-title")
                        camera_kf_state = gr.State({})
                        gr.Markdown("#### 当前帧相机参数", elem_classes="section-title")
                        with gr.Row():
                            camera_zoom = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="缩放 (Zoom)")
                            camera_pan_x = gr.Slider(-100, 100, value=0, step=5, label="平移 X (Pan X)")
                            camera_pan_y = gr.Slider(-100, 100, value=0, step=5, label="平移 Y (Pan Y)")
                            camera_rotation = gr.Slider(-45, 45, value=0, step=5, label="旋转 (°)")

                        with gr.Row():
                            camera_save_btn = gr.Button("保存当前帧相机", variant="secondary")
                            camera_delete_btn = gr.Button("删除当前帧相机", variant="secondary")

                        gr.Markdown("### 轨迹预览", elem_classes="section-title")
                        with gr.Row():
                            preview_btn = gr.Button(
                                "预览 2D 控制", variant="secondary"
                            )
                        preview_video = gr.Video(
                            label="2D 控制预览视频", interactive=False
                        )

                        gr.Markdown("### LLM 助手（DeepSeek / OpenAI 兼容）", elem_classes="section-title")
                        with gr.Accordion("对话与配置", open=True):
                            with gr.Row():
                                llm_base_url = gr.Textbox(
                                    label="Base URL",
                                    value="https://api.deepseek.com",
                                    placeholder="例如：https://api.deepseek.com 或 http://127.0.0.1:8000",
                                )
                                llm_model = gr.Textbox(
                                    label="Model",
                                    value="deepseek-chat",
                                    placeholder="例如：deepseek-chat / gpt-4o-mini / 你的本地模型名",
                                )
                            with gr.Row():
                                llm_api_key = gr.Textbox(
                                    label="API Key",
                                    type="password",
                                    placeholder="若服务端不需要可留空",
                                )
                                llm_timeout = gr.Slider(
                                    5, 180, value=60, step=1, label="请求超时 (秒)"
                                )

                            llm_chatbot = gr.Chatbot(
                                label="对话",
                                height=260,
                            )
                            llm_status = gr.Textbox(
                                label="LLM 状态",
                                value="",
                                interactive=False,
                            )
                            with gr.Row():
                                llm_user_msg = gr.Textbox(
                                    label="你的要求",
                                    placeholder="例如：让相机逐渐推近，同时把局部点轨迹改成从左到右",
                                )
                            with gr.Row():
                                llm_send_btn = gr.Button("发送并应用", variant="primary")
                                llm_clear_btn = gr.Button("清空对话", variant="secondary")

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
                            value="",
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
                        with gr.Row():
                            gen_params_btn = gr.Button(
                                "生成模型参数 (.pt)", variant="secondary"
                            )
                            params_status = gr.Textbox(
                                label="参数状态", value="尚未生成", interactive=False
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
        outputs=[bbox_editor, point_editor],
    )

    sync_btn.click(
        fn=sync_image_to_editors,
        inputs=[input_image],
        outputs=[bbox_editor, point_editor],
    )

    # ---- 帧滑条范围随 num_frames 更新 ----
    num_frames.change(
        fn=_frame_slider_updates,
        inputs=[num_frames],
        outputs=[motion_frame_idx],
    )

    # ---- 切换帧时，重置画布为输入图像（避免跨帧残留笔迹） ----
    motion_frame_idx.change(fn=_reset_editor_canvas, inputs=[input_image], outputs=[bbox_editor])
    motion_frame_idx.change(fn=_reset_editor_canvas, inputs=[input_image], outputs=[point_editor])

    # ---- Bbox 关键帧保存/删除 ----
    bbox_save_btn.click(
        fn=save_bbox_keyframe,
        inputs=[bbox_editor, motion_frame_idx, bbox_kf_state],
        outputs=[bbox_kf_state, bbox_json_text],
    )
    bbox_delete_btn.click(
        fn=delete_bbox_keyframe,
        inputs=[motion_frame_idx, bbox_kf_state],
        outputs=[bbox_kf_state, bbox_json_text],
    )

    # ---- Point 关键帧保存/删除 ----
    point_save_btn.click(
        fn=save_point_keyframe,
        inputs=[point_editor, motion_frame_idx, point_kf_state],
        outputs=[point_kf_state, point_json_text],
    )
    point_delete_btn.click(
        fn=delete_point_keyframe,
        inputs=[motion_frame_idx, point_kf_state],
        outputs=[point_kf_state, point_json_text],
    )

    # ---- Camera 关键帧加载/保存/删除 ----
    motion_frame_idx.change(
        fn=load_camera_keyframe,
        inputs=[motion_frame_idx, camera_kf_state],
        outputs=[camera_zoom, camera_pan_x, camera_pan_y, camera_rotation],
    )
    camera_save_btn.click(
        fn=save_camera_keyframe,
        inputs=[motion_frame_idx, camera_zoom, camera_pan_x, camera_pan_y, camera_rotation, camera_kf_state],
        outputs=[camera_kf_state, camera_json_text],
    )
    camera_delete_btn.click(
        fn=delete_camera_keyframe,
        inputs=[motion_frame_idx, camera_kf_state],
        outputs=[camera_kf_state, camera_json_text],
    )

    gen_params_btn.click(
        fn=generate_model_params_from_ui,
        inputs=[
            input_image,
            end_image,
            num_frames,
            height,
            width,
            bbox_json_text,
            camera_json_text,
            point_json_text,
        ],
        outputs=[bbox_mask_file, track_video_file, params_status],
    )

    preview_btn.click(
        fn=preview_control_video,
        inputs=[
            input_image,
            num_frames,
            height,
            width,
            bbox_json_text,
            camera_json_text,
            point_json_text,
            fps,
        ],
        outputs=[preview_video],
    )

    llm_send_btn.click(
        fn=llm_apply_instruction,
        inputs=[
            llm_user_msg,
            llm_chatbot,
            llm_base_url,
            llm_api_key,
            llm_model,
            llm_timeout,
            bbox_json_text,
            camera_json_text,
            point_json_text,
            prompt,
            negative_prompt,
            height,
            width,
            num_frames,
            fps,
            num_inference_steps,
            cfg_scale,
            sigma_shift,
            seed,
            motion_frame_idx,
            bbox_kf_state,
            point_kf_state,
            camera_kf_state,
        ],
        outputs=[
            llm_chatbot,
            bbox_json_text,
            point_json_text,
            camera_json_text,
            bbox_kf_state,
            point_kf_state,
            camera_kf_state,
            prompt,
            negative_prompt,
            height,
            width,
            num_frames,
            fps,
            num_inference_steps,
            cfg_scale,
            sigma_shift,
            seed,
            motion_frame_idx,
            llm_status,
            llm_user_msg,
        ],
    )

    llm_user_msg.submit(
        fn=llm_apply_instruction,
        inputs=[
            llm_user_msg,
            llm_chatbot,
            llm_base_url,
            llm_api_key,
            llm_model,
            llm_timeout,
            bbox_json_text,
            camera_json_text,
            point_json_text,
            prompt,
            negative_prompt,
            height,
            width,
            num_frames,
            fps,
            num_inference_steps,
            cfg_scale,
            sigma_shift,
            seed,
            motion_frame_idx,
            bbox_kf_state,
            point_kf_state,
            camera_kf_state,
        ],
        outputs=[
            llm_chatbot,
            bbox_json_text,
            point_json_text,
            camera_json_text,
            bbox_kf_state,
            point_kf_state,
            camera_kf_state,
            prompt,
            negative_prompt,
            height,
            width,
            num_frames,
            fps,
            num_inference_steps,
            cfg_scale,
            sigma_shift,
            seed,
            motion_frame_idx,
            llm_status,
            llm_user_msg,
        ],
    )

    llm_clear_btn.click(
        fn=llm_clear_chat,
        inputs=[],
        outputs=[llm_chatbot, llm_user_msg],
    )

    generate_btn.click(
        fn=generate_video,
        inputs=[
            prompt, negative_prompt,
            input_image, end_image,
            height, width, num_frames, num_inference_steps,
            cfg_scale, sigma_shift, seed, fps,
            bbox_mask_file, track_video_file, bbox_json_text, camera_json_text, point_json_text,
        ],
        outputs=[output_video],
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
