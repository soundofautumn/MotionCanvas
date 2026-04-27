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
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
from PIL import Image


def _to_norm_xyxy(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> List[float]:
    w = max(1, int(w))
    h = max(1, int(h))
    x1n = _clamp01(float(x1) / float(w))
    y1n = _clamp01(float(y1) / float(h))
    x2n = _clamp01(float(x2) / float(w))
    y2n = _clamp01(float(y2) / float(h))
    if x2n <= x1n:
        x2n = _clamp01(x1n + 1e-4)
    if y2n <= y1n:
        y2n = _clamp01(y1n + 1e-4)
    return [round(x1n, 4), round(y1n, 4), round(x2n, 4), round(y2n, 4)]


# ==================== Optional GroundingDINO + SAM ====================


_GDINO_PROCESSOR = None
_GDINO_MODEL = None
_GDINO_LAST_MODEL_ID = None
_GDINO_MODEL_ID_OVERRIDE = None
_SAM_PREDICTOR = None
_SAM_LAST_CKPT = None
_SAM_LAST_TYPE = None
_SAM_CKPT_OVERRIDE = None
_SAM_TYPE_OVERRIDE = None


def _get_torch_device() -> str:
    try:
        import torch  # type: ignore

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _get_gdino_model_and_processor():
    """Load GroundingDINO via Hugging Face Transformers (optional dependency).

    Model path is configured by GUI (no env vars).
    """

    global _GDINO_MODEL, _GDINO_PROCESSOR
    if _GDINO_MODEL is not None and _GDINO_PROCESSOR is not None:
        return _GDINO_MODEL, _GDINO_PROCESSOR

    try:
        import torch  # type: ignore
        from transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "未安装 GroundingDINO 所需依赖（transformers/torch）。\n"
            "建议在 venv 中安装：/home/qjming/MotionCanvas/.venv/bin/pip install -U torch transformers\n"
            "（如需 GPU 版 torch 请按你的环境选择正确的安装方式）"
        ) from e

    model_id = (_GDINO_MODEL_ID_OVERRIDE or "").strip() if _GDINO_MODEL_ID_OVERRIDE is not None else ""
    if not model_id:
        local_candidates = [
            "/root/autodl-tmp/models/grounding_dino/GroundingDINO",
            "/root/autodl-tmp/models/grounding_dino/grounding-dino-base",
        ]
        for p in local_candidates:
            try:
                if os.path.isdir(p):
                    model_id = p
                    break
            except Exception:
                continue

    # To avoid unexpected network downloads, require a local directory.
    if not model_id or not os.path.isdir(model_id):
        raise RuntimeError(
            "未配置或未找到 GroundingDINO 模型目录。\n"
            "请先运行 download_models.sh 下载模型，然后在 GUI 里填写 GroundingDINO 模型目录路径。"
        )

    device = _get_torch_device()

    processor = GroundingDinoProcessor.from_pretrained(model_id)
    model = GroundingDinoForObjectDetection.from_pretrained(model_id)
    model.to(device)
    model.eval()

    _GDINO_PROCESSOR = processor
    _GDINO_MODEL = model
    return model, processor


def _gdino_best_box_norm(
    image: Image.Image,
    *,
    query: str,
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
) -> Optional[Dict[str, Any]]:
    """Return best matched bbox (norm) for a text query using GroundingDINO.

    Output: {"score": float, "bbox_norm": [x1,y1,x2,y2], "label": str}
    """

    if image is None:
        return None
    q = (query or "").strip()
    if not q:
        return None

    model, processor = _get_gdino_model_and_processor()
    device = next(model.parameters()).device

    # transformers expects text prompt; add punctuation improves grounding for some backends
    text = q
    if not text.endswith("."):
        text = text + "."

    inputs = processor(images=image.convert("RGB"), text=text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    try:
        import torch  # type: ignore

        with torch.no_grad():
            outputs = model(**inputs)
    except Exception as e:
        raise RuntimeError(f"GroundingDINO 推理失败: {e}") from e

    w, h = image.size
    try:
        import torch  # type: ignore

        target_sizes = torch.tensor([[int(h), int(w)]], device=device)
        processed = processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=float(box_threshold),
            text_threshold=float(text_threshold),
            target_sizes=target_sizes,
        )
    except Exception as e:
        raise RuntimeError(
            "GroundingDINO 后处理失败（transformers 版本可能不匹配）。\n"
            "请尝试升级 transformers：/home/qjming/MotionCanvas/.venv/bin/pip install -U transformers"
        ) from e

    if not processed:
        return None
    p0 = processed[0] or {}
    boxes = p0.get("boxes")
    scores = p0.get("scores")
    labels = p0.get("labels")
    if boxes is None or scores is None:
        return None

    best_i = None
    best_s = -1.0
    try:
        n = int(boxes.shape[0])
    except Exception:
        n = len(boxes)
    for i in range(n):
        try:
            s = float(scores[i])
        except Exception:
            continue
        if s > best_s:
            best_s = s
            best_i = i

    if best_i is None:
        return None

    try:
        x1, y1, x2, y2 = [float(v) for v in boxes[best_i].tolist()]
    except Exception:
        return None

    bbox_norm = _to_norm_xyxy(x1, y1, x2, y2, w, h)
    label = ""
    try:
        if labels is not None:
            label = str(labels[best_i])
    except Exception:
        label = ""

    return {"score": float(best_s), "bbox_norm": bbox_norm, "label": label}


def _get_sam_predictor():
    """Load SAM predictor (optional dependency).

    Model path is configured by GUI (no env vars).
    """

    global _SAM_PREDICTOR
    if _SAM_PREDICTOR is not None:
        return _SAM_PREDICTOR

    ckpt = (_SAM_CKPT_OVERRIDE or "").strip() if _SAM_CKPT_OVERRIDE is not None else ""
    if not ckpt:
        raise RuntimeError(
            "未配置 SAM checkpoint。请在 GUI（LLM 助手）里填写 SAM checkpoint 路径。"
        )
    if not os.path.isfile(ckpt):
        raise RuntimeError(f"SAM checkpoint 文件不存在: {ckpt}")

    try:
        import torch  # type: ignore
        from segment_anything import SamPredictor, sam_model_registry  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "未安装 SAM 依赖（segment-anything）。\n"
            "你可以从官方仓库安装：pip install git+https://github.com/facebookresearch/segment-anything.git\n"
            "或者将其加入你的环境后再重试。"
        ) from e

    sam_type = ((_SAM_TYPE_OVERRIDE or "").strip() if _SAM_TYPE_OVERRIDE is not None else "") or "vit_h"
    device = _get_torch_device()

    sam = sam_model_registry[sam_type](checkpoint=ckpt)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    _SAM_PREDICTOR = predictor
    return predictor


def _sam_point_from_box(
    image: Image.Image,
    *,
    box_norm: List[float],
    method: str = "centroid",
) -> Optional[List[float]]:
    """Use SAM to refine a point from a bbox.

    method:
      - centroid: mask centroid
      - top_center: top-most row's center x
    Returns point in norm coords [x,y]
    """

    if image is None:
        return None
    if not isinstance(box_norm, list) or len(box_norm) != 4:
        return None

    predictor = _get_sam_predictor()

    img = image.convert("RGB")
    w, h = img.size
    img_np = np.array(img)
    predictor.set_image(img_np)

    x1n, y1n, x2n, y2n = [float(v) for v in box_norm]
    x1 = max(0.0, min(float(w - 1), x1n * float(w)))
    y1 = max(0.0, min(float(h - 1), y1n * float(h)))
    x2 = max(0.0, min(float(w - 1), x2n * float(w)))
    y2 = max(0.0, min(float(h - 1), y2n * float(h)))
    if x2 <= x1 or y2 <= y1:
        return None

    box_xyxy = np.array([x1, y1, x2, y2], dtype=np.float32)
    masks, scores, _ = predictor.predict(box=box_xyxy, multimask_output=True)
    if masks is None or len(masks) == 0:
        return None

    best_i = int(np.argmax(scores)) if scores is not None and len(scores) > 0 else 0
    mask = masks[best_i]
    if mask is None:
        return None

    ys, xs = np.where(mask.astype(bool))
    if xs.size == 0 or ys.size == 0:
        return None

    m = (method or "centroid").strip().lower()
    if m == "top_center":
        y_min = int(ys.min())
        xs_top = xs[ys == y_min]
        if xs_top.size == 0:
            return None
        cx = float(xs_top.mean())
        cy = float(y_min)
    else:
        cx = float(xs.mean())
        cy = float(ys.mean())

    return [round(_clamp01(cx / float(w)), 4), round(_clamp01(cy / float(h)), 4)]


def llm_preload_localizers(
    gdino_model_dir: str,
    sam_ckpt: str,
    sam_type: str,
) -> str:
    """Preload GroundingDINO + SAM once (best effort) for faster tool usage.

    This is designed to be called by Gradio `app.load(...)`.
    """

    global _GDINO_MODEL, _GDINO_PROCESSOR, _GDINO_LAST_MODEL_ID, _GDINO_MODEL_ID_OVERRIDE
    global _SAM_PREDICTOR, _SAM_LAST_CKPT, _SAM_LAST_TYPE, _SAM_CKPT_OVERRIDE, _SAM_TYPE_OVERRIDE

    gdino_model_dir_s = (gdino_model_dir or "").strip()
    sam_ckpt_s = (sam_ckpt or "").strip()
    sam_type_s = (sam_type or "").strip() or "vit_h"

    # Update GroundingDINO config; clear cache if changed.
    if gdino_model_dir_s != (_GDINO_LAST_MODEL_ID or ""):
        _GDINO_MODEL_ID_OVERRIDE = gdino_model_dir_s
        _GDINO_MODEL = None
        _GDINO_PROCESSOR = None
        _GDINO_LAST_MODEL_ID = gdino_model_dir_s

    # Update SAM config; clear predictor if changed.
    if sam_ckpt_s != (_SAM_LAST_CKPT or "") or sam_type_s != (_SAM_LAST_TYPE or ""):
        _SAM_CKPT_OVERRIDE = sam_ckpt_s
        _SAM_TYPE_OVERRIDE = sam_type_s
        _SAM_PREDICTOR = None
        _SAM_LAST_CKPT = sam_ckpt_s
        _SAM_LAST_TYPE = sam_type_s

    parts: List[str] = []
    try:
        _get_gdino_model_and_processor()
        parts.append("✅ GroundingDINO 已加载")
    except Exception as e:
        parts.append(f"⚠️ GroundingDINO 加载失败: {e}")

    try:
        _get_sam_predictor()
        parts.append("✅ SAM 已加载")
    except Exception as e:
        parts.append(f"⚠️ SAM 加载失败: {e}")

    return " | ".join(parts)


# ==================== Debug logging ====================


def _truncate_for_log(text: Any, limit: int = 2000) -> str:
    s = "" if text is None else str(text)
    if len(s) <= int(limit):
        return s
    return s[: int(limit)] + f"... (truncated, len={len(s)})"


def _redact_data_urls(text: str) -> str:
    # Avoid printing large base64 payloads (e.g., data:image/jpeg;base64,...)
    if not text:
        return text
    return re.sub(r"data:image\/[^;]+;base64,[A-Za-z0-9+/=]+", "data:image/<redacted>;base64,<redacted>", text)


def _print_llm_debug(resp: Dict[str, Any]) -> None:
    try:
        msg = ((resp or {}).get("choices") or [{}])[0].get("message") or {}
        content = msg.get("content")
        tool_calls = msg.get("tool_calls")

        print("\n[MOTIONCANVAS][LLM] raw response")

        if tool_calls:
            print("[MOTIONCANVAS][LLM] tool_calls:")
            try:
                for tc in tool_calls:
                    fn = (tc or {}).get("function") or {}
                    name = fn.get("name")
                    args = fn.get("arguments")
                    args_s = _truncate_for_log(args, limit=800)
                    args_s = _redact_data_urls(args_s)
                    print(f"  - {name}: {args_s}")
            except Exception:
                print(_truncate_for_log(tool_calls, limit=2000))

        if content is not None and str(content).strip() != "":
            s = _truncate_for_log(content, limit=4000)
            s = _redact_data_urls(s)
            print("[MOTIONCANVAS][LLM] content:")
            print(s)

        print("[MOTIONCANVAS][LLM] end\n", flush=True)
    except Exception:
        # Never fail the UI due to logging
        return


# ==================== Prompts ====================

LLM_SYSTEM_PROMPT = """你是 MotionCanvas 的“镜头/物体运动控制 + 生成参数编辑”助手。

项目作用（你需要理解后再给出可执行的修改）：
- MotionCanvas 从一张静态起始帧图像生成一段短视频。
- 用户可以同时控制：
    1) 相机运动（zoom/pan/rotation，按帧关键帧插值）
    2) 全局物体运动区域（用 bbox 序列定义；用于生成时的 bbox mask 条件）
    3) 局部物体运动（用点位轨迹 point tracks 定义；用于生成时的轨迹条件）
- 你的输出目标不是“描述怎么做”，而是“把 UI 的状态改到用户想要的效果”，例如：移动/缩放相机、平移 bbox、平移点轨迹、或更新 prompt/推理参数。

坐标/数据格式约定（非常重要）：
- bbox 的格式为 [x1, y1, x2, y2]。
    - 推荐使用归一化坐标（norm）：范围 [0,1]，以当前 width/height 为基准。
    - 也允许像素坐标（px），但你必须在输出里明确（通过 ops 的 space=px，或在 updates 里写入 px 坐标并在说明里注明）。
- point 的格式为 [x, y]，同样优先使用归一化坐标（norm，范围 [0,1]）。
- 帧索引 frame 都是 0-based，必须满足 [0, num_frames-1]。

你可以修改的 UI 状态字段：
- prompt / negative_prompt
- 生成参数：height, width, num_frames, fps, num_inference_steps, cfg_scale, sigma_shift, seed
- 运动 JSON：bbox_json、camera_json、point_json

优先策略：
1) 优先使用 tools（函数调用）表达修改意图：
    - 相机：camera_set / camera_zoom_linear / camera_pan_linear / camera_rotation_linear
    - bbox：bbox_set（精确写入某帧）/ bbox_translate（平移一段帧区间）
    - 点轨迹：points_set（精确写入某帧）/ points_translate（平移一段帧区间）
    - 生成参数：set_generation_params
2) 如果后端不支持 tools，或你需要一次性写入完整 JSON（例如直接给出 bbox_json / point_json 的完整结构），才退化为输出 JSON（包含 ops 或 updates）。

你可以连续多轮调用 tools：
- 如果一次 tool_calls 不足以完成用户需求，可以先调用一批 tools，等待状态更新后继续调用下一批 tools，直到完成。
- 注意避免无限循环：通常 1-3 轮就应完成。

关于起始帧图像的发送规则：
- 起始帧图像默认不会在每次消息都发送。
- 如果用户在 UI 勾选了“发送图片”，则本次请求会强制附带一次起始帧图像；发送完成后 UI 会自动取消勾选。
- 后续如果你仍然需要查看图像，请通过 tools 主动调用 get_input_image 获取压缩后的图像 data_url。
- 对于“获取物体 bbox/点位”的需求：
    - 优先调用 gdino_detect_bbox（用文本描述定位目标框）。
    - 如果需要更精确的“某个部位点”，优先调用 gdino_sam_detect_point（先用 GroundingDINO 得到框，再用 SAM 分割并从 mask 计算点）。

当你需要用户确认/澄清信息时：
- 直接调用 ask_user 提出问题（可以给出 choices）。
- 在得到用户答复前，不要擅自猜测，也不要继续调用会修改状态的 tools。

当你输出 JSON 时：
- 必须只输出一个 JSON 对象（不要输出 Markdown，不要输出代码块，不要输出多余文本）。
- 结构为：
    {
        "assistant_message": "给用户的简短说明（必填）",
        "ops": [ ... 可选 ... ],
        "updates": { ... 可选 ... }
    }

ops 语义（可选，用于增量编辑）：
- camera.set / camera.zoom_linear / camera.pan_linear / camera.rotation_linear
- bbox.set（精确设置某一帧 bbox；bbox=[x1,y1,x2,y2]，space=norm/px，px 会换算到 norm）
- bbox.translate（dx/dy 可用 norm 或 px；px 时会按 width/height 自动换算到 norm；可选 transition=step/linear，linear 用于平滑过渡并在 end_frame 后保持最终位移）
- points.set（精确设置某一帧的点列表；points=[[x,y],...]，space=norm/px，px 会换算到 norm）
- points.translate（dx/dy 可用 norm 或 px；px 时会按 width/height 自动换算到 norm；可选 transition=step/linear，linear 用于平滑过渡并在 end_frame 后保持最终位移）

updates 语义（可选，用于直接覆盖字段）：
- updates.bbox_json / updates.camera_json / updates.point_json 可以直接给完整 JSON 字符串或对象。

通用规则：
- 不需要改动就省略对应字段。
- 不要编造不存在的 tool 名称。
- 如果用户要“精确调整某个物体”，优先建议先获得该物体的 bbox/关键点（来自用户提供、或由视觉模型估计、或由外部定位 tool 输出），再进行平移/关键帧调整。
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


def _interp_points_norm_for_frame(point_state: Dict[str, Any], frame_idx: int) -> Optional[List[List[float]]]:
    """Interpolate point tracks at a given frame.

    point_state format: {"frame_idx": [[x,y], [x,y], ...], ...}
    Tracks are implied by list index.

    Returns a list of points (len=max_tracks) in norm coords, or None if no tracks exist.
    """

    items: List[Tuple[int, List[Any]]] = []
    max_tracks = 0
    for fi_str, pts in (point_state or {}).items():
        try:
            fi = int(fi_str)
        except Exception:
            continue
        if not isinstance(pts, list) or not pts:
            continue
        items.append((fi, pts))
        max_tracks = max(max_tracks, len(pts))

    if not items or max_tracks <= 0:
        return None

    items = sorted(items, key=lambda x: x[0])

    def _safe_xy(v: Any) -> Optional[Tuple[float, float]]:
        if not isinstance(v, (list, tuple)) or len(v) < 2:
            return None
        try:
            return float(v[0]), float(v[1])
        except Exception:
            return None

    out: List[List[float]] = []
    for track_idx in range(max_tracks):
        series: List[Tuple[int, float, float]] = []
        for fi, pts in items:
            if track_idx >= len(pts):
                continue
            xy = _safe_xy(pts[track_idx])
            if xy is None:
                continue
            x, y = xy
            series.append((fi, x, y))

        if not series:
            # No data for this index anywhere; default to (0,0) to keep shape stable.
            out.append([0.0, 0.0])
            continue

        series = sorted(series, key=lambda t: t[0])
        if frame_idx <= series[0][0]:
            _, x, y = series[0]
            out.append([x, y])
            continue
        if frame_idx >= series[-1][0]:
            _, x, y = series[-1]
            out.append([x, y])
            continue

        for j in range(len(series) - 1):
            f0, x0, y0 = series[j]
            f1, x1, y1 = series[j + 1]
            if f0 <= frame_idx <= f1:
                span = max(1, f1 - f0)
                t = (frame_idx - f0) / span
                out.append([_lerp(x0, x1, t), _lerp(y0, y1, t)])
                break
        else:
            _, x, y = series[-1]
            out.append([x, y])

    return out


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

        if op == "bbox.set":
            f = _cap_frame(item.get("frame", 0), 0)
            fi = str(int(f))
            space = str(item.get("space", "norm")).strip().lower()
            bb = item.get("bbox")
            if not isinstance(bb, (list, tuple)) or len(bb) != 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in bb]
            if space in {"px", "pixel", "pixels"}:
                x1, x2 = x1 / max(1.0, float(w)), x2 / max(1.0, float(w))
                y1, y2 = y1 / max(1.0, float(h)), y2 / max(1.0, float(h))
            x1, y1, x2, y2 = _clamp01(x1), _clamp01(y1), _clamp01(x2), _clamp01(y2)
            if x2 <= x1:
                x2 = _clamp01(x1 + 1e-4)
            if y2 <= y1:
                y2 = _clamp01(y1 + 1e-4)
            bbox_state[fi] = [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]
            continue

        if op == "points.set":
            f = _cap_frame(item.get("frame", 0), 0)
            fi = str(int(f))
            space = str(item.get("space", "norm")).strip().lower()
            pts = item.get("points")
            if not isinstance(pts, list) or not pts:
                continue
            out_pts: List[List[float]] = []
            for xy in pts:
                if not isinstance(xy, (list, tuple)) or len(xy) < 2:
                    continue
                x, y = float(xy[0]), float(xy[1])
                if space in {"px", "pixel", "pixels"}:
                    x = x / max(1.0, float(w))
                    y = y / max(1.0, float(h))
                out_pts.append([round(_clamp01(x), 4), round(_clamp01(y), 4)])
            if out_pts:
                point_state[fi] = out_pts
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
            transition = str(item.get("transition", "step")).strip().lower()
            dx = float(item.get("dx", 0.0))
            dy = float(item.get("dy", 0.0))
            if space in {"px", "pixel", "pixels"}:
                dx = dx / max(1.0, float(w))
                dy = dy / max(1.0, float(h))

            sf = _cap_frame(item.get("start_frame", 0), 0)
            ef = _cap_frame(item.get("end_frame", nf - 1), nf - 1)
            if ef < sf:
                sf, ef = ef, sf

            # Smooth transition: ramp from 0 at start_frame to full dx/dy at end_frame,
            # and keep full shift after end_frame (so no discontinuity at end).
            if transition in {"linear", "smooth", "ramp"}:
                orig_bbox_state = dict(bbox_state or {})
                existing_frames = {str(int(k)) for k in orig_bbox_state.keys()}
                span = max(1, ef - sf)

                sfi = str(int(sf))
                efi = str(int(ef))

                start_bbox = _interp_bbox_norm_for_frame(orig_bbox_state, sf)
                end_bbox = _interp_bbox_norm_for_frame(orig_bbox_state, ef)
                if start_bbox is None or end_bbox is None:
                    continue

                if sfi not in existing_frames:
                    bbox_state[sfi] = [round(float(start_bbox[0]), 4), round(float(start_bbox[1]), 4), round(float(start_bbox[2]), 4), round(float(start_bbox[3]), 4)]
                if efi not in existing_frames:
                    # end frame should be fully shifted
                    bx = list(end_bbox)
                    x1, y1, x2, y2 = [float(x) for x in bx]
                    x1, x2 = _clamp01(x1 + dx), _clamp01(x2 + dx)
                    y1, y2 = _clamp01(y1 + dy), _clamp01(y2 + dy)
                    if x2 <= x1:
                        x2 = _clamp01(x1 + 1e-4)
                    if y2 <= y1:
                        y2 = _clamp01(y1 + 1e-4)
                    bbox_state[efi] = [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]

                def _factor(fi: int) -> float:
                    if fi < sf:
                        return 0.0
                    if fi >= ef:
                        return 1.0
                    return (fi - sf) / float(span)

                def _shift(bb, t: float):
                    x1, y1, x2, y2 = [float(x) for x in bb]
                    x1, x2 = _clamp01(x1 + dx * t), _clamp01(x2 + dx * t)
                    y1, y2 = _clamp01(y1 + dy * t), _clamp01(y2 + dy * t)
                    if x2 <= x1:
                        x2 = _clamp01(x1 + 1e-4)
                    if y2 <= y1:
                        y2 = _clamp01(y1 + 1e-4)
                    return [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]

                # Shift all existing keyframes with time-dependent factor
                for fi_str, bb in list(bbox_state.items()):
                    try:
                        fi = int(fi_str)
                    except Exception:
                        continue
                    t = _factor(fi)
                    if t <= 0.0:
                        continue
                    bbox_state[fi_str] = _shift(bb, t)
                continue

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
            transition = str(item.get("transition", "step")).strip().lower()
            dx = float(item.get("dx", 0.0))
            dy = float(item.get("dy", 0.0))
            if space in {"px", "pixel", "pixels"}:
                dx = dx / max(1.0, float(w))
                dy = dy / max(1.0, float(h))

            sf = _cap_frame(item.get("start_frame", 0), 0)
            ef = _cap_frame(item.get("end_frame", nf - 1), nf - 1)
            if ef < sf:
                sf, ef = ef, sf

            if transition in {"linear", "smooth", "ramp"}:
                span = max(1, ef - sf)

                def _factor(fi: int) -> float:
                    if fi < sf:
                        return 0.0
                    if fi >= ef:
                        return 1.0
                    return (fi - sf) / float(span)

                existing_frames = {str(int(k)) for k in (point_state or {}).keys()}
                sfi = str(int(sf))
                efi = str(int(ef))
                base_start = _interp_points_norm_for_frame(point_state, sf)
                base_end = _interp_points_norm_for_frame(point_state, ef)

                if base_start is not None and sfi not in existing_frames:
                    point_state[sfi] = [[round(_clamp01(float(xy[0])), 4), round(_clamp01(float(xy[1])), 4)] for xy in base_start]
                if base_end is not None and efi not in existing_frames:
                    shifted_end = []
                    for xy in base_end:
                        x, y = float(xy[0]), float(xy[1])
                        shifted_end.append([round(_clamp01(x + dx), 4), round(_clamp01(y + dy), 4)])
                    point_state[efi] = shifted_end

                for fi_str, pts in list(point_state.items()):
                    try:
                        fi = int(fi_str)
                    except Exception:
                        continue
                    t = _factor(fi)
                    if t <= 0.0:
                        continue
                    if not isinstance(pts, list):
                        continue
                    new_pts = []
                    for xy in pts:
                        if not isinstance(xy, (list, tuple)) or len(xy) < 2:
                            continue
                        x, y = float(xy[0]), float(xy[1])
                        new_pts.append([round(_clamp01(x + dx * t), 4), round(_clamp01(y + dy * t), 4)])
                    point_state[fi_str] = new_pts
                continue

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
                        "transition": {
                            "type": "string",
                            "enum": ["step", "linear"],
                            "description": "step: apply constant translation inside the range (legacy). linear: ramp from 0 at start_frame to full dx/dy at end_frame, and hold after end_frame.",
                        },
                    },
                    "required": ["start_frame", "end_frame", "dx", "dy"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "bbox_set",
                "description": "Set bbox for a specific frame (norm or px). This writes an exact [x1,y1,x2,y2] box into bbox_json.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "frame": {"type": "integer"},
                        "bbox": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                        "space": {"type": "string", "enum": ["norm", "px"]},
                    },
                    "required": ["frame", "bbox"],
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
                        "transition": {
                            "type": "string",
                            "enum": ["step", "linear"],
                            "description": "step: apply constant translation inside the range (legacy). linear: ramp from 0 at start_frame to full dx/dy at end_frame, and hold after end_frame.",
                        },
                    },
                    "required": ["start_frame", "end_frame", "dx", "dy"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "points_set",
                "description": "Set point(s) for a specific frame (norm or px). This overwrites that frame's points list in point_json.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "frame": {"type": "integer"},
                        "points": {
                            "type": "array",
                            "items": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                            "minItems": 1,
                        },
                        "space": {"type": "string", "enum": ["norm", "px"]},
                    },
                    "required": ["frame", "points"],
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

        # --- Ask user for confirmation ---
        {
            "type": "function",
            "function": {
                "name": "ask_user",
                "description": "Ask the user a clarification/confirmation question and stop. Use when requirements are ambiguous and you must confirm before editing states.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "The question to ask the user (Chinese preferred)."},
                        "choices": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional choices the user can pick from.",
                        },
                        "allow_freeform": {"type": "boolean", "description": "Whether user can answer freely. Default true."},
                    },
                    "required": ["question"],
                },
            },
        },

        # --- On-demand image access ---
        {
            "type": "function",
            "function": {
                "name": "get_input_image",
                "description": "Get the (compressed) input image as a data_url for on-demand visual inspection.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "max_side": {"type": "integer", "description": "Max side length for resizing. Default 384."},
                        "image_format": {"type": "string", "enum": ["JPEG", "PNG"], "description": "Default JPEG."},
                        "jpeg_quality": {"type": "integer", "description": "JPEG quality (1-95). Default 80."},
                    },
                    "required": [],
                },
            },
        },

        # --- Localization tools (GroundingDINO + SAM) ---
        {
            "type": "function",
            "function": {
                "name": "gdino_detect_bbox",
                "description": "Use GroundingDINO to detect a target bbox from text query and write it into bbox_json for a target frame.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "frame": {"type": "integer", "description": "Target frame index. If omitted, use current_frame_idx."},
                        "query": {"type": "string", "description": "Text query describing the target (e.g., 'a red car', 'the cup', 'logo', 'left headlight')."},
                        "box_threshold": {"type": "number", "description": "Box threshold, default 0.3."},
                        "text_threshold": {"type": "number", "description": "Text threshold, default 0.25."},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "gdino_sam_detect_point",
                "description": "Use GroundingDINO to get a bbox then SAM to segment and compute a precise point (centroid/top_center) and write it into point_json for a target frame.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "frame": {"type": "integer", "description": "Target frame index. If omitted, use current_frame_idx."},
                        "query": {"type": "string", "description": "Text query describing the target region/part whose point you want (e.g., 'cup rim', 'car logo', 'left headlight corner')."},
                        "point_method": {"type": "string", "enum": ["centroid", "top_center"], "description": "How to convert SAM mask to a point. Default centroid."},
                        "box_threshold": {"type": "number", "description": "DINO box threshold, default 0.3."},
                        "text_threshold": {"type": "number", "description": "DINO text threshold, default 0.25."},
                    },
                    "required": ["query"],
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
    input_image: Optional[Image.Image],
    current_frame_idx: int,
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
) -> Tuple[
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, Any],
    str,
    str,
    Dict[str, Any],
    List[str],
    List[Dict[str, Any]],
    Optional[Dict[str, Any]],
]:
    msgs: List[str] = []
    tool_results: List[Dict[str, Any]] = []
    ask_payload: Optional[Dict[str, Any]] = None

    def _cap_frame_local(v, default_v=0):
        try:
            vv = int(v)
        except Exception:
            vv = int(default_v)
        nf = int(num_frames)
        return max(0, min(max(0, nf - 1), vv))

    def _add_result_for(call_id: Any, tool_name: str, payload: Any):
        tool_results.append(
            {
                "tool_call_id": str(call_id or ""),
                "name": str(tool_name or ""),
                "content": json.dumps(payload, ensure_ascii=False),
            }
        )

    # If the model asks the user a question, we short-circuit: do not apply other tools.
    # This prevents accidental state edits before the user confirms.
    ask_call = None
    for c in (tool_calls or []):
        fn0 = (c.get("function") or {})
        nm0 = str(fn0.get("name") or "").strip()
        if nm0 == "ask_user":
            ask_call = c
            break

    if ask_call is not None:
        fn = (ask_call.get("function") or {})
        call_id = ask_call.get("id") or ""
        args_raw = fn.get("arguments")
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
        except Exception:
            args = {}
        question = str((args or {}).get("question") or "").strip()
        choices = (args or {}).get("choices")
        allow_freeform = (args or {}).get("allow_freeform")
        if allow_freeform is None:
            allow_freeform = True
        if not question:
            question = "请确认：你希望我如何继续？"
        if not isinstance(choices, list):
            choices = None
        else:
            choices = [str(x) for x in choices if str(x).strip() != ""]
            if not choices:
                choices = None
        ask_payload = {
            "type": "ask",
            "question": question,
            "choices": choices,
            "allow_freeform": bool(allow_freeform),
        }
        msgs.append("ask_user")
        _add_result_for(call_id, "ask_user", {"ok": True, **ask_payload})
        return (
            bbox_state,
            point_state,
            camera_state,
            prompt,
            negative_prompt,
            gen_params,
            msgs,
            tool_results,
            ask_payload,
        )

    for call in tool_calls:
        fn = (call.get("function") or {})
        name = str(fn.get("name") or "").strip()
        call_id = call.get("id") or ""
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
            _add_result_for(call_id, name, {"ok": True, "applied": "camera_set"})
        elif name == "camera_zoom_linear":
            op = {"op": "camera.zoom_linear", **args}
            bbox_state, point_state, camera_state = apply_ops_to_states(
                [op], bbox_state, point_state, camera_state, num_frames, width, height, bbox_json_text=bbox_json_text, point_json_text=point_json_text, camera_json_text=camera_json_text
            )
            msgs.append("camera_zoom_linear")
            _add_result_for(call_id, name, {"ok": True, "applied": "camera_zoom_linear"})
        elif name == "camera_pan_linear":
            op = {"op": "camera.pan_linear", **args}
            bbox_state, point_state, camera_state = apply_ops_to_states(
                [op], bbox_state, point_state, camera_state, num_frames, width, height, bbox_json_text=bbox_json_text, point_json_text=point_json_text, camera_json_text=camera_json_text
            )
            msgs.append("camera_pan_linear")
            _add_result_for(call_id, name, {"ok": True, "applied": "camera_pan_linear"})
        elif name == "camera_rotation_linear":
            op = {"op": "camera.rotation_linear", **args}
            bbox_state, point_state, camera_state = apply_ops_to_states(
                [op], bbox_state, point_state, camera_state, num_frames, width, height, bbox_json_text=bbox_json_text, point_json_text=point_json_text, camera_json_text=camera_json_text
            )
            msgs.append("camera_rotation_linear")
            _add_result_for(call_id, name, {"ok": True, "applied": "camera_rotation_linear"})
        elif name == "bbox_translate":
            op = {"op": "bbox.translate", **args}
            bbox_state, point_state, camera_state = apply_ops_to_states(
                [op], bbox_state, point_state, camera_state, num_frames, width, height, bbox_json_text=bbox_json_text, point_json_text=point_json_text, camera_json_text=camera_json_text
            )
            msgs.append("bbox_translate")
            _add_result_for(call_id, name, {"ok": True, "applied": "bbox_translate"})
        elif name == "bbox_set":
            op = {"op": "bbox.set", "frame": args.get("frame", 0), "bbox": args.get("bbox"), "space": args.get("space", "norm")}
            bbox_state, point_state, camera_state = apply_ops_to_states(
                [op], bbox_state, point_state, camera_state, num_frames, width, height, bbox_json_text=bbox_json_text, point_json_text=point_json_text, camera_json_text=camera_json_text
            )
            msgs.append("bbox_set")
            _add_result_for(call_id, name, {"ok": True, "applied": "bbox_set"})
        elif name == "points_translate":
            op = {"op": "points.translate", **args}
            bbox_state, point_state, camera_state = apply_ops_to_states(
                [op], bbox_state, point_state, camera_state, num_frames, width, height, bbox_json_text=bbox_json_text, point_json_text=point_json_text, camera_json_text=camera_json_text
            )
            msgs.append("points_translate")
            _add_result_for(call_id, name, {"ok": True, "applied": "points_translate"})
        elif name == "points_set":
            op = {"op": "points.set", "frame": args.get("frame", 0), "points": args.get("points"), "space": args.get("space", "norm")}
            bbox_state, point_state, camera_state = apply_ops_to_states(
                [op], bbox_state, point_state, camera_state, num_frames, width, height, bbox_json_text=bbox_json_text, point_json_text=point_json_text, camera_json_text=camera_json_text
            )
            msgs.append("points_set")
            _add_result_for(call_id, name, {"ok": True, "applied": "points_set"})
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
            _add_result_for(call_id, name, {"ok": True, "applied": "set_generation_params"})
        elif name == "get_input_image":
            if input_image is None:
                raise gr.Error("未提供 input_image，无法获取起始帧图像（请上传起始帧图像）")

            max_side = int(args.get("max_side", 384) or 384)
            image_format = str(args.get("image_format", "JPEG") or "JPEG")
            jpeg_quality = int(args.get("jpeg_quality", 80) or 80)

            data_url = pil_to_data_url(
                input_image,
                max_side=max_side,
                image_format=image_format,
                jpeg_quality=jpeg_quality,
            )
            w, h = input_image.size
            msgs.append(f"get_input_image(max_side={max_side})")
            _add_result_for(call_id, name, {"ok": True, "width": int(w), "height": int(h), "data_url": data_url})
        elif name == "gdino_detect_bbox":
            if input_image is None:
                raise gr.Error("未提供 input_image，无法执行 GroundingDINO 定位（请上传起始帧图像）")

            f = _cap_frame_local(args.get("frame", None), default_v=int(current_frame_idx))
            query = str(args.get("query") or "").strip()
            box_th = float(args.get("box_threshold", 0.3) or 0.3)
            text_th = float(args.get("text_threshold", 0.25) or 0.25)

            det = _gdino_best_box_norm(input_image, query=query, box_threshold=box_th, text_threshold=text_th)
            if not det:
                msgs.append("gdino_detect_bbox:not_found")
                _add_result_for(call_id, name, {"ok": True, "found": False, "query": query, "box_threshold": box_th, "text_threshold": text_th})
                continue

            bbox_norm = det.get("bbox_norm")
            if not isinstance(bbox_norm, list) or len(bbox_norm) != 4:
                msgs.append("gdino_detect_bbox:invalid_box")
                _add_result_for(call_id, name, {"ok": False, "error": "invalid_box", "det": det})
                continue

            fi = str(int(f))
            bbox_state[fi] = bbox_norm
            msgs.append(f"gdino_detect_bbox(score={float(det.get('score',0)):.2f},frame={f})")
            _add_result_for(call_id, name, {"ok": True, "found": True, "frame": int(f), "query": query, "score": float(det.get("score", 0.0)), "bbox_norm": bbox_norm})

        elif name == "gdino_sam_detect_point":
            if input_image is None:
                raise gr.Error("未提供 input_image，无法执行 GroundingDINO+SAM 定位（请上传起始帧图像）")

            f = _cap_frame_local(args.get("frame", None), default_v=int(current_frame_idx))
            query = str(args.get("query") or "").strip()
            method = str(args.get("point_method") or "centroid").strip().lower()
            box_th = float(args.get("box_threshold", 0.3) or 0.3)
            text_th = float(args.get("text_threshold", 0.25) or 0.25)

            det = _gdino_best_box_norm(input_image, query=query, box_threshold=box_th, text_threshold=text_th)
            if not det:
                msgs.append("gdino_sam_detect_point:not_found")
                _add_result_for(call_id, name, {"ok": True, "found": False, "query": query, "box_threshold": box_th, "text_threshold": text_th})
                continue

            bbox_norm = det.get("bbox_norm")
            if not isinstance(bbox_norm, list) or len(bbox_norm) != 4:
                msgs.append("gdino_sam_detect_point:invalid_box")
                _add_result_for(call_id, name, {"ok": False, "error": "invalid_box", "det": det})
                continue

            # Try SAM refinement; if SAM is not configured, fall back to bbox center.
            point_norm = None
            sam_used = False
            try:
                point_norm = _sam_point_from_box(input_image, box_norm=bbox_norm, method=method)
                sam_used = point_norm is not None
            except Exception:
                point_norm = None
                sam_used = False

            if point_norm is None:
                cx = (float(bbox_norm[0]) + float(bbox_norm[2])) / 2.0
                cy = (float(bbox_norm[1]) + float(bbox_norm[3])) / 2.0
                point_norm = [round(_clamp01(cx), 4), round(_clamp01(cy), 4)]

            fi = str(int(f))
            point_state[fi] = [[float(point_norm[0]), float(point_norm[1])]]
            msgs.append(f"gdino_sam_detect_point(frame={f},sam={sam_used})")
            _add_result_for(
                call_id,
                name,
                {
                    "ok": True,
                    "found": True,
                    "frame": int(f),
                    "query": query,
                    "point_method": method,
                    "sam_used": bool(sam_used),
                    "score": float(det.get("score", 0.0)),
                    "bbox_norm": bbox_norm,
                    "point_norm": point_state[fi][0],
                },
            )
        else:
            msgs.append(f"unknown_tool:{name}")
            _add_result_for(call_id, name, {"ok": False, "error": f"unknown_tool:{name}"})

    print("\n[MOTIONCANVAS][LLM] tool_results:")
    for tr in (tool_results or []):
        try:
            name = tr.get("name") or ""
            content = tr.get("content") or ""
            print(f"  - {name}: {_truncate_for_log(content, limit=2000)}")
        except Exception:
            pass
    print("[MOTIONCANVAS][LLM] tool_results end\n", flush=True)

    return bbox_state, point_state, camera_state, prompt, negative_prompt, gen_params, msgs, tool_results, ask_payload


# ==================== Public UI callback ====================


def _chatbot_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content

    # Gradio "messages" format: content can be a list of message parts, e.g.
    # [{"type":"text","text":"hi"}] or [{"text":"hi","type":"text"}]
    if isinstance(content, list):
        parts: List[str] = []
        for p in content:
            if p is None:
                continue
            if isinstance(p, str):
                parts.append(p)
                continue
            if isinstance(p, dict):
                if "text" in p and p.get("text") is not None:
                    parts.append(str(p.get("text")))
                    continue
                # best-effort
                if p.get("type") == "text" and p.get("text") is not None:
                    parts.append(str(p.get("text")))
                    continue
        if parts:
            return "".join(parts)
        return str(content)

    if isinstance(content, dict):
        if content.get("type") == "text" and content.get("text") is not None:
            return str(content.get("text"))
        if "text" in content and content.get("text") is not None:
            return str(content.get("text"))

    return str(content)


def _normalize_chat_history(chat_history: Any) -> List[Dict[str, Any]]:
    """Normalize gr.Chatbot history to a list of {role, content(str)} dicts.

    - Gradio 6.x Chatbot uses "messages" format.
    - For backward compatibility, also accepts legacy [(user, assistant), ...] tuples.
    """

    out: List[Dict[str, Any]] = []
    for item in list(chat_history or []):
        if isinstance(item, (list, tuple)) and len(item) == 2:
            u, a = item
            u_text = ("" if u is None else str(u))
            a_text = ("" if a is None else str(a))
            if u_text:
                out.append({"role": "user", "content": u_text})
            if a_text:
                out.append({"role": "assistant", "content": a_text})
            continue

        if isinstance(item, dict) and "role" in item and "content" in item:
            role = str(item.get("role") or "")
            content = _chatbot_content_to_text(item.get("content"))
            if role and content is not None:
                out.append({"role": role, "content": content})
            continue

    return out


def _history_to_openai_messages(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    msgs: List[Dict[str, Any]] = []
    for m in history or []:
        role = str(m.get("role") or "").strip()
        if role not in {"user", "assistant"}:
            continue
        msgs.append({"role": role, "content": _chatbot_content_to_text(m.get("content"))})
    return msgs


def llm_apply_instruction(
    user_message,
    chat_history,
    llm_base_url,
    llm_api_key,
    llm_model,
    llm_timeout,
    gdino_model_dir,
    sam_ckpt,
    sam_type,
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

    # Apply localizer configuration from GUI (no env vars)
    llm_preload_localizers(
        gdino_model_dir=str(gdino_model_dir or ""),
        sam_ckpt=str(sam_ckpt or ""),
        sam_type=str(sam_type or ""),
    )

    history = _normalize_chat_history(chat_history)

    def _camera_values_for_frame(camera_state: Dict[str, Any], frame_idx: int) -> Tuple[float, float, float, float]:
        fi = str(int(frame_idx))
        params = camera_state.get(fi)
        if not isinstance(params, dict):
            return 1.0, 0.0, 0.0, 0.0
        return (
            float(params.get("zoom", 1.0)),
            float(params.get("pan_x", 0.0)),
            float(params.get("pan_y", 0.0)),
            float(params.get("rotation", 0.0)),
        )

    def _editor_canvas_from_input(img: Optional[Image.Image]):
        if img is None:
            return gr.update()
        try:
            return np.array(img)
        except Exception:
            return gr.update()

    # Defaults for outputs that sync back to motion editor
    default_cam_zoom, default_cam_pan_x, default_cam_pan_y, default_cam_rot = _camera_values_for_frame(
        dict(camera_kf_state or {}),
        int(motion_frame_idx),
    )
    default_bbox_canvas = _editor_canvas_from_input(input_image)
    default_point_canvas = _editor_canvas_from_input(input_image)
    if not user_message:
        # 不抛异常，避免输入框进入错误态导致无法继续操作
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
            default_cam_zoom,
            default_cam_pan_x,
            default_cam_pan_y,
            default_cam_rot,
            default_bbox_canvas,
            default_point_canvas,
            gr.update(),
            "请输入你的要求",
            user_message,
        )

    messages: List[Dict[str, Any]] = [{"role": "system", "content": LLM_SYSTEM_PROMPT}]
    messages.extend(_history_to_openai_messages(history))

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

    sent_image_this_turn = bool(llm_send_image)
    if sent_image_this_turn:
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

    # Allow multi-round tool calling: call -> apply tools -> feed updated state -> call again.
    # Keep it small to avoid accidental loops.
    max_tool_rounds = 4
    tool_names_all: List[str] = []

    # If we sent the image due to checkbox, auto-uncheck after this request.
    send_image_checkbox_update = gr.update(value=False) if sent_image_this_turn else gr.update()

    # Current mutable state during this instruction
    cur_prompt = prompt
    cur_negative_prompt = negative_prompt
    cur_height = int(height)
    cur_width = int(width)
    cur_num_frames = int(num_frames)
    cur_fps = int(fps)
    cur_steps = int(num_inference_steps)
    cur_cfg = float(cfg_scale)
    cur_sigma = float(sigma_shift)
    cur_seed = int(seed)

    tool_role_supported = True
    last_resp: Optional[Dict[str, Any]] = None
    last_raw_content = ""

    for round_idx in range(max_tool_rounds):
        try:
            resp = _openai_chat_complete(
                base_url=llm_base_url,
                api_key=llm_api_key,
                model=(llm_model or "").strip(),
                messages=messages,
                temperature=0.2,
                timeout=float(llm_timeout),
                force_json=False,  # tools 开启时不强制 JSON，有助于提高 tool_calls 命中率
                tools=tools,
                tool_choice="auto",
            )
            _print_llm_debug(resp)
        except Exception as e:
            history.extend(
                [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": f"❌ LLM 调用失败：{e}"},
                ]
            )
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
                default_cam_zoom,
                default_cam_pan_x,
                default_cam_pan_y,
                default_cam_rot,
                default_bbox_canvas,
                default_point_canvas,
                send_image_checkbox_update,
                "LLM 调用失败（见对话）",
                "",
            )

        last_resp = resp
        try:
            last_raw_content = resp["choices"][0]["message"].get("content") or ""
        except Exception:
            last_raw_content = ""

        tool_calls = _tool_calls_from_response(resp)
        if not tool_calls:
            # No tool calls; stop the tool loop and handle content below.
            break

        gen_params: Dict[str, Any] = {
            "height": cur_height,
            "width": cur_width,
            "num_frames": cur_num_frames,
            "fps": cur_fps,
            "num_inference_steps": cur_steps,
            "cfg_scale": cur_cfg,
            "sigma_shift": cur_sigma,
            "seed": cur_seed,
        }

        (
            base_bbox_state,
            base_point_state,
            base_camera_state,
            cur_prompt,
            cur_negative_prompt,
            gen_params,
            tool_names,
            tool_results,
            ask_payload,
        ) = _apply_tool_calls(
            tool_calls,
            input_image=input_image,
            current_frame_idx=int(motion_frame_idx),
            bbox_state=base_bbox_state,
            point_state=base_point_state,
            camera_state=base_camera_state,
            num_frames=int(cur_num_frames),
            width=int(cur_width),
            height=int(cur_height),
            bbox_json_text=bbox_json_text or "",
            point_json_text=point_json_text or "",
            camera_json_text=camera_json_text or "",
            prompt=cur_prompt,
            negative_prompt=cur_negative_prompt,
            gen_params=gen_params,
        )
        tool_names_all.extend(tool_names)

        if ask_payload is not None:
            q = str(ask_payload.get("question") or "").strip()
            choices = ask_payload.get("choices")
            allow_freeform = bool(ask_payload.get("allow_freeform", True))

            parts = ["需要你确认一个问题：", q]
            if isinstance(choices, list) and choices:
                parts.append("可选项：" + " / ".join([str(x) for x in choices]))
            if allow_freeform:
                parts.append("请直接回复你的选择或补充说明。")
            msg = "\n".join([p for p in parts if p and str(p).strip() != ""]) or "请确认：你希望我如何继续？"

            history.extend([
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": msg},
            ])
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
                default_cam_zoom,
                default_cam_pan_x,
                default_cam_pan_y,
                default_cam_rot,
                default_bbox_canvas,
                default_point_canvas,
                send_image_checkbox_update,
                "需要用户确认",
                "",
            )

        # Apply param updates (snap/clip) after each tool round
        if "height" in gen_params:
            snapped = _snap_to_step(gen_params.get("height"), 256, 16, 1280)
            if snapped is not None:
                cur_height = int(snapped)
        if "width" in gen_params:
            snapped = _snap_to_step(gen_params.get("width"), 256, 16, 1280)
            if snapped is not None:
                cur_width = int(snapped)
        if "num_frames" in gen_params:
            snapped = _snap_to_step(gen_params.get("num_frames"), 5, 4, 121)
            if snapped is not None:
                cur_num_frames = int(snapped)
        if "fps" in gen_params:
            snapped = _snap_to_step(gen_params.get("fps"), 8, 1, 30)
            if snapped is not None:
                cur_fps = int(snapped)
        if "num_inference_steps" in gen_params:
            snapped = _snap_to_step(gen_params.get("num_inference_steps"), 10, 1, 100)
            if snapped is not None:
                cur_steps = int(snapped)
        if "cfg_scale" in gen_params and gen_params.get("cfg_scale") is not None:
            cur_cfg = max(1.0, min(15.0, float(gen_params.get("cfg_scale"))))
        if "sigma_shift" in gen_params and gen_params.get("sigma_shift") is not None:
            cur_sigma = max(1.0, min(15.0, float(gen_params.get("sigma_shift"))))
        if "seed" in gen_params and gen_params.get("seed") is not None:
            cur_seed = int(gen_params.get("seed"))

        # Recompute JSON texts for the next round (so the model sees updated state)
        bbox_json_text = _bbox_state_to_json(base_bbox_state)
        point_json_text = _point_state_to_json(base_point_state)
        camera_json_text = _camera_state_to_json(base_camera_state)

        # Feed tool results back to the model. Some OpenAI-compatible backends don't
        # accept role="tool" messages; fall back to a plain user message when needed.
        if tool_role_supported:
            try:
                assistant_msg = (resp.get("choices") or [{}])[0].get("message") or {}
                messages.append(
                    {
                        "role": "assistant",
                        "content": assistant_msg.get("content") or "",
                        "tool_calls": assistant_msg.get("tool_calls") or tool_calls,
                    }
                )
                print("[MOTIONCANVAS][LLM] sending tool results to LLM:", flush=True)
                for tr in (tool_results or []):
                    try:
                        name = tr.get("name") or ""
                        content = tr.get("content") or ""
                        print(f"  - {name}: {_truncate_for_log(content, limit=2000)}")
                    except Exception:
                        pass
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tr.get("tool_call_id") or "",
                            "content": tr.get("content") or "{}",
                        }
                    )
                print("[MOTIONCANVAS][LLM] sending tool results end\n", flush=True)
            except Exception:
                tool_role_supported = False

        state_blob = {
            "prompt": cur_prompt,
            "negative_prompt": cur_negative_prompt,
            "height": int(cur_height),
            "width": int(cur_width),
            "num_frames": int(cur_num_frames),
            "fps": int(cur_fps),
            "num_inference_steps": int(cur_steps),
            "cfg_scale": float(cur_cfg),
            "sigma_shift": float(cur_sigma),
            "seed": int(cur_seed),
            "bbox_json": bbox_json_text or "",
            "camera_json": camera_json_text or "",
            "point_json": point_json_text or "",
            "current_frame_idx": int(motion_frame_idx),
        }

        tool_results_blob = []
        try:
            for tr in (tool_results or []):
                tool_results_blob.append(
                    {
                        "name": tr.get("name"),
                        "content": tr.get("content"),
                    }
                )
        except Exception:
            tool_results_blob = []

        cont_payload = (
            f"已执行第 {round_idx + 1} 轮 tools：{', '.join(tool_names) if tool_names else '(none)'}\n"
            "tool 返回（如有）：\n"
            + json.dumps(tool_results_blob, ensure_ascii=False)
            + "\n\n"
            "请基于下面的最新状态继续完成用户需求：如果还需要进一步修改，请继续调用 tools；"
            "如果已经完成，请给出简短说明。\n\n"
            + json.dumps(state_blob, ensure_ascii=False)
        )
        messages.append({"role": "user", "content": cont_payload})

    # End tool loop

    resp = last_resp or {"choices": [{"message": {"content": ""}}]}
    raw_content = last_raw_content
    tool_calls = _tool_calls_from_response(resp)

    # New outputs (initialized from current values; may already be updated by tool loop)
    new_bbox_json = bbox_json_text
    new_point_json = point_json_text
    new_camera_json = camera_json_text
    new_prompt = cur_prompt
    new_negative_prompt = cur_negative_prompt
    new_height = cur_height
    new_width = cur_width
    new_num_frames = cur_num_frames
    new_fps = cur_fps
    new_steps = cur_steps
    new_cfg = cur_cfg
    new_sigma = cur_sigma
    new_seed = cur_seed

    assistant_msg = None

    try:
        if tool_names_all:
            # Tools were applied in this instruction (possibly across multiple rounds).
            new_bbox_json = _bbox_state_to_json(base_bbox_state)
            new_point_json = _point_state_to_json(base_point_state)
            new_camera_json = _camera_state_to_json(base_camera_state)
            new_bbox_state = base_bbox_state
            new_point_state = base_point_state
            new_camera_state = base_camera_state

            # If the final model message is JSON ops/updates, apply it; otherwise treat as plain text.
            content = ""
            try:
                content = resp["choices"][0]["message"].get("content")
            except Exception:
                content = ""
            obj = None
            if content and isinstance(content, str):
                try:
                    obj = _extract_json_object(content)
                except Exception:
                    obj = None

            if isinstance(obj, dict) and (obj.get("ops") is not None or obj.get("updates") is not None):
                updates = obj.get("updates", {}) if isinstance(obj, dict) else {}
                ops = obj.get("ops", []) if isinstance(obj, dict) else []
                assistant_msg = obj.get("assistant_message") if isinstance(obj, dict) else None

                if ops and isinstance(ops, list):
                    base_bbox_state, base_point_state, base_camera_state = apply_ops_to_states(
                        ops,
                        base_bbox_state,
                        base_point_state,
                        base_camera_state,
                        num_frames=int(new_num_frames),
                        width=int(new_width),
                        height=int(new_height),
                        bbox_json_text=new_bbox_json or "",
                        point_json_text=new_point_json or "",
                        camera_json_text=new_camera_json or "",
                    )
                    new_bbox_json = _bbox_state_to_json(base_bbox_state)
                    new_point_json = _point_state_to_json(base_point_state)
                    new_camera_json = _camera_state_to_json(base_camera_state)

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
                if "cfg_scale" in updates and updates.get("cfg_scale") is not None:
                    new_cfg = max(1.0, min(15.0, float(updates.get("cfg_scale"))))
                if "sigma_shift" in updates and updates.get("sigma_shift") is not None:
                    new_sigma = max(1.0, min(15.0, float(updates.get("sigma_shift"))))
                if "seed" in updates and updates.get("seed") is not None:
                    new_seed = int(updates.get("seed"))

                new_bbox_state = _bbox_state_from_json_text(new_bbox_json)
                new_point_state = _point_state_from_json_text(new_point_json)
                new_camera_state = _camera_state_from_json_text(new_camera_json)
            else:
                # Plain text final message: use it if present; otherwise show applied tools summary.
                s = ("" if content is None else str(content)).strip()
                assistant_msg = s if s else f"✅ 已通过 tools 应用：{', '.join(tool_names_all)}"

        else:
            # Fallback: parse as JSON output (ops/updates). If not JSON, show plain text and apply nothing.
            content = ""
            try:
                content = resp["choices"][0]["message"].get("content")
            except Exception:
                content = ""
            obj = None
            if content and isinstance(content, str):
                try:
                    obj = _extract_json_object(content)
                except Exception:
                    obj = None

            if obj is None:
                assistant_msg = ("" if content is None else str(content)).strip() or "✅ 已应用更新"
                new_bbox_state = base_bbox_state
                new_point_state = base_point_state
                new_camera_state = base_camera_state
            else:
                obj = obj

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
        history.extend([
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": f"❌ 解析/应用更新失败：{e}"},
        ])
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
            default_cam_zoom,
            default_cam_pan_x,
            default_cam_pan_y,
            default_cam_rot,
            default_bbox_canvas,
            default_point_canvas,
            send_image_checkbox_update,
            "LLM 输出不合法（未应用）",
            "",
        )

    if assistant_msg is not None and str(assistant_msg).strip() != "":
        msg = str(assistant_msg).strip()
    else:
        # 如果模型没提供 assistant_message，就把原始 content（截断+脱敏）展示出来，避免永远只有“✅ 已应用更新”
        raw_s = _redact_data_urls(_truncate_for_log(raw_content, limit=800)).strip()
        msg = raw_s if raw_s else "✅ 已应用更新"

    history.extend([
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": msg},
    ])

    nf = int(new_num_frames)
    max_frame = max(0, nf - 1)
    cur_frame = int(motion_frame_idx)
    new_frame_val = min(cur_frame, max_frame)
    frame_update = gr.update(minimum=0, maximum=max_frame, value=new_frame_val)

    # Sync camera sliders + reset editor canvases so motion editor reflects new state immediately.
    cam_zoom_v, cam_pan_x_v, cam_pan_y_v, cam_rot_v = _camera_values_for_frame(new_camera_state, new_frame_val)
    bbox_canvas_v = _editor_canvas_from_input(input_image)
    point_canvas_v = _editor_canvas_from_input(input_image)

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
        cam_zoom_v,
        cam_pan_x_v,
        cam_pan_y_v,
        cam_rot_v,
        bbox_canvas_v,
        point_canvas_v,
        send_image_checkbox_update,
        "✅ 已应用 LLM 更新",
        "",
    )


def llm_clear_chat():
    return [], ""
