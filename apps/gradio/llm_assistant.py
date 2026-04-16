"""LLM assistant helpers for MotionCanvas Gradio UI.

This module is intentionally UI-facing (uses gradio types/errors) and is imported by
apps/gradio/motioncanvas.py.

It provides:
- OpenAI-compatible chat completion via OpenAI Python SDK
- Optional multimodal (send input image)
- Tool-calling interface (OpenAI function calling) for high-level motion edits
- Fallback to JSON output (ops/updates) when tools are unsupported
"""

from __future__ import annotations

import base64
import io
import json
import math
import re
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
from PIL import Image


# ==================== Prompts ====================

LLM_SYSTEM_PROMPT = """你是一个 MotionCanvas 的动作/参数编辑助手。

你必须只输出一个 JSON 对象（不要输出 Markdown，不要输出代码块，不要输出多余文本）。

优先使用 tools（函数调用）来表达修改意图：例如调用 camera.zoom_linear / bbox.translate 等工具。
如果后端不支持 tools，你才退化为输出 JSON（包含 ops 或 updates）。

当你输出 JSON 时，结构为：
{
  "assistant_message": "给用户的简短说明（可选）",
  "ops": [ ... 可选 ... ],
  "updates": { ... 可选 ... }
}

规则：
- ops / tools 中的帧索引必须在 [0, num_frames-1]。
- 不需要改动就省略对应字段。
"""


# ==================== JSON helpers ====================


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


def _extract_json_object(text: Any):
    if text is None:
        raise ValueError("空响应")
    text = str(text).strip()
    if not text:
        raise ValueError("空响应")

    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, flags=re.IGNORECASE)
    if m:
        return json.loads(m.group(1))

    m2 = re.search(r"(\{[\s\S]*\})", text)
    if m2:
        return json.loads(m2.group(1))

    raise ValueError("无法从模型输出中解析 JSON")


def _ensure_json_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, indent=2)


# ==================== Multimodal image encoding ====================


def pil_to_data_url(input_image: Image.Image, max_side: int = 768, image_format: str = "JPEG", jpeg_quality: int = 85) -> str:
    if input_image is None:
        raise ValueError("input_image 为空")

    img = input_image.convert("RGB")
    w, h = img.size
    if w <= 0 or h <= 0:
        raise ValueError("无效的图像尺寸")

    max_side = int(max_side)
    if max_side > 0:
        scale = min(1.0, float(max_side) / float(max(w, h)))
        if scale < 1.0:
            img = img.resize(
                (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                Image.BILINEAR,
            )

    buf = io.BytesIO()
    fmt = str(image_format or "JPEG").upper()
    if fmt == "PNG":
        img.save(buf, format="PNG")
        mime = "image/png"
    else:
        img.save(buf, format="JPEG", quality=int(jpeg_quality), optimize=True)
        mime = "image/jpeg"

    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:{mime};base64,{b64}"


# ==================== OpenAI-compatible chat ====================


def _normalize_openai_base_url(base_url: str) -> str:
    s = (base_url or "").strip()
    if not s:
        raise ValueError("base_url 不能为空")
    s = s.rstrip("/")
    return s


def _openai_chat_complete(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float = 0.2,
    timeout: float = 60,
    force_json: bool = True,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[str] = None,
) -> Dict[str, Any]:
    base = _normalize_openai_base_url(base_url)
    if not base.endswith("/v1"):
        base = base + "/v1"

    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("未安装 OpenAI Python SDK（openai）。请先安装：pip install openai") from e

    client = OpenAI(
        base_url=base,
        api_key=(str(api_key).strip() if api_key is not None else ""),
        timeout=float(timeout),
    )

    kwargs: Dict[str, Any] = {
        "model": (model or "").strip(),
        "messages": messages,
        "temperature": float(temperature),
    }
    if tools:
        kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice

    if force_json:
        kwargs["response_format"] = {"type": "json_object"}

    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception:
        if force_json and "response_format" in kwargs:
            # 兼容部分后端不支持 response_format
            kwargs.pop("response_format", None)
            resp = client.chat.completions.create(**kwargs)
        else:
            raise

    try:
        return resp.model_dump()
    except Exception:
        content = ""
        try:
            content = resp.choices[0].message.content
        except Exception:
            pass
        out: Dict[str, Any] = {"choices": [{"message": {"content": content}}]}
        # 尝试补齐 tool_calls
        try:
            out["choices"][0]["message"]["tool_calls"] = resp.choices[0].message.tool_calls
        except Exception:
            pass
        return out


# ==================== State <-> JSON (motion) ====================


def _lerp(a: float, b: float, t: float) -> float:
    return a * (1.0 - t) + b * t


def _bbox_state_to_json(bbox_state: Dict[str, Any]) -> str:
    frames: Dict[str, Any] = {}
    for k, v in (bbox_state or {}).items():
        frames[str(int(k))] = v
    frames = dict(sorted(frames.items(), key=lambda kv: int(kv[0])))
    if not frames:
        return ""
    return json.dumps({"objects": [{"frames": frames}]}, indent=2, ensure_ascii=False)


def _point_state_to_json(point_state: Dict[str, Any]) -> str:
    state = {str(int(k)): v for k, v in (point_state or {}).items()}
    state = dict(sorted(state.items(), key=lambda kv: int(kv[0])))
    if not state:
        return ""

    max_len = 0
    for pts in state.values():
        if isinstance(pts, list):
            max_len = max(max_len, len(pts))

    tracks: List[Dict[str, Any]] = []
    for idx in range(max_len):
        frames: Dict[str, Any] = {}
        for fi_str, pts in state.items():
            if idx < len(pts):
                frames[fi_str] = list(pts[idx])
        if frames:
            tracks.append({"frames": frames})

    if not tracks:
        return ""
    return json.dumps({"points": tracks}, indent=2, ensure_ascii=False)


def _camera_state_to_json(camera_state: Dict[str, Any]) -> str:
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
    return json.dumps({"camera": {"keyframes": keyframes}}, indent=2, ensure_ascii=False)


def _bbox_state_from_json_text(bbox_json_text: str) -> Dict[str, Any]:
    if not bbox_json_text or not str(bbox_json_text).strip():
        return {}
    data = json.loads(bbox_json_text)
    objects = data.get("objects", [])
    if not objects:
        return {}
    frames = objects[0].get("frames", {})
    out: Dict[str, Any] = {}
    for fi_str, bbox in (frames or {}).items():
        out[str(int(fi_str))] = bbox
    return out


def _point_state_from_json_text(point_json_text: str) -> Dict[str, Any]:
    if not point_json_text or not str(point_json_text).strip():
        return {}
    data = json.loads(point_json_text)
    points = data.get("points", [])
    if not points:
        return {}

    frame_to_points: Dict[str, Any] = {}
    for pt in points:
        frames = pt.get("frames", {})
        for fi_str, xy in (frames or {}).items():
            fi = str(int(fi_str))
            frame_to_points.setdefault(fi, []).append(xy)
    return frame_to_points


def _camera_state_from_json_text(camera_json_text: str) -> Dict[str, Any]:
    if not camera_json_text or not str(camera_json_text).strip():
        return {}
    data = json.loads(camera_json_text)
    keyframes = data.get("camera", {}).get("keyframes", [])
    if not keyframes:
        return {}

    out: Dict[str, Any] = {}
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


# ==================== Ops interpreter ====================


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _interp_bbox_norm_for_frame(frames: Dict[str, Any], frame_idx: int):
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
    ops: Any,
    bbox_state: Dict[str, Any],
    point_state: Dict[str, Any],
    camera_state: Dict[str, Any],
    num_frames: int,
    width: int,
    height: int,
    bbox_json_text: str = "",
    point_json_text: str = "",
    camera_json_text: str = "",
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    nf = int(num_frames)
    w = int(width)
    h = int(height)

    bbox_state = dict(bbox_state or {})
    point_state = dict(point_state or {})
    camera_state = dict(camera_state or {})

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

            for fi_str, bb in orig_bbox_state.items():
                try:
                    fi = int(fi_str)
                except Exception:
                    continue
                if sf <= fi <= ef:
                    bbox_state[fi_str] = _shift(bb)

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


# ==================== Tools (function calling) ====================


def get_motion_tools() -> List[Dict[str, Any]]:
    # Keep schemas minimal and stable for OpenAI-compatible backends.
    return [
        {
            "type": "function",
            "function": {
                "name": "camera_set",
                "description": "Set camera parameters for a specific frame.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "frame": {"type": "integer"},
                        "zoom": {"type": "number"},
                        "pan": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                        "rotation": {"type": "number"},
                    },
                    "required": ["frame"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "camera_zoom_linear",
                "description": "Create/adjust camera zoom with linear keyframes between two frames.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start_frame": {"type": "integer"},
                        "end_frame": {"type": "integer"},
                        "start": {"type": "number"},
                        "end": {"type": "number"},
                    },
                    "required": ["start_frame", "end_frame", "start", "end"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "camera_pan_linear",
                "description": "Create/adjust camera pan with linear keyframes between two frames.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start_frame": {"type": "integer"},
                        "end_frame": {"type": "integer"},
                        "start": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                        "end": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                    },
                    "required": ["start_frame", "end_frame", "start", "end"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "camera_rotation_linear",
                "description": "Create/adjust camera rotation with linear keyframes between two frames.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start_frame": {"type": "integer"},
                        "end_frame": {"type": "integer"},
                        "start": {"type": "number"},
                        "end": {"type": "number"},
                    },
                    "required": ["start_frame", "end_frame", "start", "end"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "bbox_translate",
                "description": "Translate bbox keyframes by dx/dy (norm or px) between frames.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start_frame": {"type": "integer"},
                        "end_frame": {"type": "integer"},
                        "dx": {"type": "number"},
                        "dy": {"type": "number"},
                        "space": {"type": "string", "enum": ["norm", "px"]},
                    },
                    "required": ["start_frame", "end_frame", "dx", "dy"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "points_translate",
                "description": "Translate point keyframes by dx/dy (norm or px) between frames.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start_frame": {"type": "integer"},
                        "end_frame": {"type": "integer"},
                        "dx": {"type": "number"},
                        "dy": {"type": "number"},
                        "space": {"type": "string", "enum": ["norm", "px"]},
                    },
                    "required": ["start_frame", "end_frame", "dx", "dy"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "set_generation_params",
                "description": "Update prompt/negative_prompt and generation parameters.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string"},
                        "negative_prompt": {"type": "string"},
                        "height": {"type": "integer"},
                        "width": {"type": "integer"},
                        "num_frames": {"type": "integer"},
                        "fps": {"type": "integer"},
                        "num_inference_steps": {"type": "integer"},
                        "cfg_scale": {"type": "number"},
                        "sigma_shift": {"type": "number"},
                        "seed": {"type": "integer"},
                    },
                    "required": [],
                },
            },
        },
    ]


def _tool_calls_from_response(resp: Dict[str, Any]) -> List[Dict[str, Any]]:
    try:
        msg = resp["choices"][0]["message"]
        tc = msg.get("tool_calls")
        return tc if isinstance(tc, list) else []
    except Exception:
        return []


def _apply_tool_calls(
    tool_calls: List[Dict[str, Any]],
    *,
    bbox_state: Dict[str, Any],
    point_state: Dict[str, Any],
    camera_state: Dict[str, Any],
    num_frames: int,
    width: int,
    height: int,
    bbox_json_text: str,
    point_json_text: str,
    camera_json_text: str,
    prompt: str,
    negative_prompt: str,
    gen_params: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], str, str, Dict[str, Any], List[str]]:
    msgs: List[str] = []

    for call in tool_calls:
        fn = (call.get("function") or {})
        name = str(fn.get("name") or "").strip()
        args_raw = fn.get("arguments")
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
        except Exception:
            args = {}

        if name == "camera_set":
            op = {
                "op": "camera.set",
                "frame": args.get("frame", 0),
                "zoom": args.get("zoom"),
                "pan": args.get("pan"),
                "rotation": args.get("rotation"),
            }
            bbox_state, point_state, camera_state = apply_ops_to_states(
                [op], bbox_state, point_state, camera_state, num_frames, width, height, bbox_json_text=bbox_json_text, point_json_text=point_json_text, camera_json_text=camera_json_text
            )
            msgs.append("camera_set")
        elif name == "camera_zoom_linear":
            op = {"op": "camera.zoom_linear", **args}
            bbox_state, point_state, camera_state = apply_ops_to_states(
                [op], bbox_state, point_state, camera_state, num_frames, width, height, bbox_json_text=bbox_json_text, point_json_text=point_json_text, camera_json_text=camera_json_text
            )
            msgs.append("camera_zoom_linear")
        elif name == "camera_pan_linear":
            op = {"op": "camera.pan_linear", **args}
            bbox_state, point_state, camera_state = apply_ops_to_states(
                [op], bbox_state, point_state, camera_state, num_frames, width, height, bbox_json_text=bbox_json_text, point_json_text=point_json_text, camera_json_text=camera_json_text
            )
            msgs.append("camera_pan_linear")
        elif name == "camera_rotation_linear":
            op = {"op": "camera.rotation_linear", **args}
            bbox_state, point_state, camera_state = apply_ops_to_states(
                [op], bbox_state, point_state, camera_state, num_frames, width, height, bbox_json_text=bbox_json_text, point_json_text=point_json_text, camera_json_text=camera_json_text
            )
            msgs.append("camera_rotation_linear")
        elif name == "bbox_translate":
            op = {"op": "bbox.translate", **args}
            bbox_state, point_state, camera_state = apply_ops_to_states(
                [op], bbox_state, point_state, camera_state, num_frames, width, height, bbox_json_text=bbox_json_text, point_json_text=point_json_text, camera_json_text=camera_json_text
            )
            msgs.append("bbox_translate")
        elif name == "points_translate":
            op = {"op": "points.translate", **args}
            bbox_state, point_state, camera_state = apply_ops_to_states(
                [op], bbox_state, point_state, camera_state, num_frames, width, height, bbox_json_text=bbox_json_text, point_json_text=point_json_text, camera_json_text=camera_json_text
            )
            msgs.append("points_translate")
        elif name == "set_generation_params":
            # Only update fields that exist
            if isinstance(args, dict):
                if args.get("prompt") is not None:
                    prompt = str(args.get("prompt"))
                if args.get("negative_prompt") is not None:
                    negative_prompt = str(args.get("negative_prompt"))
                for k in ("height", "width", "num_frames", "fps", "num_inference_steps", "seed"):
                    if k in args and args.get(k) is not None:
                        gen_params[k] = args.get(k)
                for k in ("cfg_scale", "sigma_shift"):
                    if k in args and args.get(k) is not None:
                        gen_params[k] = args.get(k)
            msgs.append("set_generation_params")
        else:
            msgs.append(f"unknown_tool:{name}")

    return bbox_state, point_state, camera_state, prompt, negative_prompt, gen_params, msgs


# ==================== Public UI callback ====================


def llm_apply_instruction(
    user_message,
    chat_history,
    llm_base_url,
    llm_api_key,
    llm_model,
    llm_timeout,
    input_image,
    llm_send_image,
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

    messages: List[Dict[str, Any]] = [{"role": "system", "content": LLM_SYSTEM_PROMPT}]
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
        + "\n\n当前状态（可作为你生成 tools / ops / updates 的依据）：\n"
        + json.dumps(state_blob, ensure_ascii=False)
    )

    if bool(llm_send_image):
        if input_image is None:
            raise gr.Error("已勾选发送图片，但未上传起始帧图像")
        data_url = pil_to_data_url(input_image, max_side=768)
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_payload},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        )
    else:
        messages.append({"role": "user", "content": user_payload})

    # Prepare baseline states
    base_bbox_state = dict(bbox_kf_state or {})
    base_point_state = dict(point_kf_state or {})
    base_camera_state = dict(camera_kf_state or {})

    tools = get_motion_tools()

    try:
        resp = _openai_chat_complete(
            base_url=llm_base_url,
            api_key=llm_api_key,
            model=(llm_model or "").strip(),
            messages=messages,
            temperature=0.2,
            timeout=float(llm_timeout),
            force_json=False,  # tool calling responses may not be JSON
            tools=tools,
            tool_choice="auto",
        )
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

    tool_calls = _tool_calls_from_response(resp)

    # New outputs
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

    assistant_msg = None

    try:
        if tool_calls:
            gen_params: Dict[str, Any] = {}
            gen_params["height"] = height
            gen_params["width"] = width
            gen_params["num_frames"] = num_frames
            gen_params["fps"] = fps
            gen_params["num_inference_steps"] = num_inference_steps
            gen_params["cfg_scale"] = cfg_scale
            gen_params["sigma_shift"] = sigma_shift
            gen_params["seed"] = seed

            (
                base_bbox_state,
                base_point_state,
                base_camera_state,
                new_prompt,
                new_negative_prompt,
                gen_params,
                tool_names,
            ) = _apply_tool_calls(
                tool_calls,
                bbox_state=base_bbox_state,
                point_state=base_point_state,
                camera_state=base_camera_state,
                num_frames=int(num_frames),
                width=int(width),
                height=int(height),
                bbox_json_text=bbox_json_text or "",
                point_json_text=point_json_text or "",
                camera_json_text=camera_json_text or "",
                prompt=new_prompt,
                negative_prompt=new_negative_prompt,
                gen_params=gen_params,
            )

            # Apply param updates (snap/clip)
            if "height" in gen_params:
                snapped = _snap_to_step(gen_params.get("height"), 256, 16, 1280)
                if snapped is not None:
                    new_height = int(snapped)
            if "width" in gen_params:
                snapped = _snap_to_step(gen_params.get("width"), 256, 16, 1280)
                if snapped is not None:
                    new_width = int(snapped)
            if "num_frames" in gen_params:
                snapped = _snap_to_step(gen_params.get("num_frames"), 5, 4, 121)
                if snapped is not None:
                    new_num_frames = int(snapped)
            if "fps" in gen_params:
                snapped = _snap_to_step(gen_params.get("fps"), 8, 1, 30)
                if snapped is not None:
                    new_fps = int(snapped)
            if "num_inference_steps" in gen_params:
                snapped = _snap_to_step(gen_params.get("num_inference_steps"), 10, 1, 100)
                if snapped is not None:
                    new_steps = int(snapped)
            if "cfg_scale" in gen_params and gen_params.get("cfg_scale") is not None:
                val = float(gen_params.get("cfg_scale"))
                new_cfg = max(1.0, min(15.0, val))
            if "sigma_shift" in gen_params and gen_params.get("sigma_shift") is not None:
                val = float(gen_params.get("sigma_shift"))
                new_sigma = max(1.0, min(15.0, val))
            if "seed" in gen_params and gen_params.get("seed") is not None:
                new_seed = int(gen_params.get("seed"))

            new_bbox_json = _bbox_state_to_json(base_bbox_state)
            new_point_json = _point_state_to_json(base_point_state)
            new_camera_json = _camera_state_to_json(base_camera_state)

            assistant_msg = f"✅ 已通过 tools 应用：{', '.join(tool_names)}"
            new_bbox_state = base_bbox_state
            new_point_state = base_point_state
            new_camera_state = base_camera_state

        else:
            # Fallback: parse as JSON output (ops/updates)
            content = ""
            try:
                content = resp["choices"][0]["message"].get("content")
            except Exception:
                content = ""
            obj = _extract_json_object(content)

            updates = obj.get("updates", {}) if isinstance(obj, dict) else {}
            ops = obj.get("ops", []) if isinstance(obj, dict) else []
            assistant_msg = obj.get("assistant_message") if isinstance(obj, dict) else None

            # Apply ops first
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
                new_bbox_json = _bbox_state_to_json(base_bbox_state)
                new_point_json = _point_state_to_json(base_point_state)
                new_camera_json = _camera_state_to_json(base_camera_state)

            # Apply updates (legacy)
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

            # Sync states from JSON after updates
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
