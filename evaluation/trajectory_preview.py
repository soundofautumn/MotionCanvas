"""
轨迹预览视频生成 —— 在评估视频质量的同时输出轨迹可视化预览。

使用 CoTracker 对生成的视频进行点轨迹追踪，然后将轨迹以点+轨迹线
的形式叠加渲染到视频帧上，输出一个带轨迹叠加的预览视频。

用法（独立）:
    from evaluation.trajectory_preview import render_trajectory_preview
    frames = load_video_frames("output.mp4")
    preview = render_trajectory_preview(frames, device="cuda")
    preview[0].save("trajectory_preview.mp4", save_all=True, ...)

依赖:
    - torch (必需)
    - CoTracker3 (通过 torch.hub 加载)
    - decord (读取视频)
    - imageio (保存视频)
    - numpy / Pillow
"""

import math
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw


def load_cotracker(
    device: str = "cuda",
    dtype: torch.dtype = None,
) -> Optional[torch.nn.Module]:
    """
    加载 CoTracker3 模型。

    优先使用本地缓存（通过 COTRACKER_HUB_DIR 环境变量指定），
    否则从 GitHub torch hub 下载。

    Returns:
        CoTracker 模型（eval 模式）或 None（如果加载失败）。
    """
    import torch

    if dtype is None:
        dtype = torch.bfloat16

    try:
        cotracker_local = os.environ.get("COTRACKER_HUB_DIR")
        if cotracker_local and os.path.isdir(
            os.path.join(cotracker_local, "facebookresearch_co-tracker_main")
        ):
            torch.hub.set_dir(cotracker_local)
            model = torch.hub.load(
                os.path.join(cotracker_local, "facebookresearch_co-tracker_main"),
                "cotracker3_offline",
                source="local",
            ).to(device, dtype=dtype)
        else:
            model = torch.hub.load(
                "facebookresearch/co-tracker",
                "cotracker3_offline",
                trust_repo=True,
            ).to(device, dtype=dtype)
        model.requires_grad_(False)
        model.eval()
        return model
    except Exception as e:
        print(f"  [WARN] CoTracker 加载失败: {e}")
        return None


def compute_tracks(
    cotracker: torch.nn.Module,
    frames: List[Image.Image],
    device: str = "cuda",
    grid_size: int = 12,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用 CoTracker 计算视频帧的点轨迹。

    Args:
        cotracker: CoTracker 模型。
        frames: RGB PIL Image 列表。
        device: 计算设备。
        grid_size: 点网格大小（grid_size × grid_size 个追踪点）。

    Returns:
        tracks: (T, N, 2) 数组，每个点为 (x, y) 像素坐标。
        visibility: (T, N) bool 数组，表示点是否可见。
    """
    import torch

    # 拼接视频帧张量: (1, T, 3, H, W), 值域 [0, 1]
    video_tensor = (
        torch.stack(
            [
                torch.from_numpy(np.array(f, dtype=np.float32)).permute(2, 0, 1) / 255.0
                for f in frames
            ]
        )
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        pred_tracks, pred_visibility = cotracker(
            video_tensor, grid_size=grid_size, backward_tracking=False
        )

    # pred_tracks: (1, T, N, 2) 像素坐标
    # pred_visibility: (1, T, N, 1)
    tracks = pred_tracks[0].cpu().numpy()  # (T, N, 2)
    visibility = pred_visibility[0, :, :, 0].cpu().numpy()  # (T, N)

    return tracks, visibility


def _generate_colors(n: int, seed: int = 42) -> List[Tuple[int, int, int]]:
    """生成 n 个视觉上可区分的颜色。"""
    import random
    rng = random.Random(seed)
    colors = []
    for i in range(n):
        # 使用 HSV-like 方法生成颜色
        hue = i * (360.0 / max(n, 1))
        # 将 hue 转为 RGB，保持高饱和度 and 中等明度
        h = hue / 60.0
        x = 1 - abs(h % 2 - 1)
        rgb_vals = {
            0: (1, x, 0),
            1: (x, 1, 0),
            2: (0, 1, x),
            3: (0, x, 1),
            4: (x, 0, 1),
            5: (1, 0, x),
        }
        r, g, b = rgb_vals[int(h) % 6]
        # 加一点随机偏移让颜色更自然
        r = int((r * 0.7 + rng.random() * 0.3) * 255)
        g = int((g * 0.7 + rng.random() * 0.3) * 255)
        b = int((b * 0.7 + rng.random() * 0.3) * 255)
        colors.append((r, g, b))
    return colors


def render_trajectory_preview(
    frames: List[Image.Image],
    tracks: Optional[np.ndarray] = None,
    visibility: Optional[np.ndarray] = None,
    trail_length: int = 8,
    point_radius: int = 3,
    line_width: int = 2,
    subsample_tracks: int = 1,
    device: str = "cuda",
    cotracker: Optional[torch.nn.Module] = None,
) -> List[Image.Image]:
    """
    在视频帧上渲染轨迹预览。

    如果未提供 tracks/visibility，会自动加载 CoTracker 并计算。

    Args:
        frames: 原始视频帧 (RGB PIL Image 列表)。
        tracks: 可选，预计算轨迹 (T, N, 2) 像素坐标。
        visibility: 可选，预计算可见性 (T, N) bool。
        trail_length: 轨迹拖尾长度（帧数）。
        point_radius: 点半径（像素）。
        line_width: 轨迹线宽度（像素）。
        subsample_tracks: 轨迹点下采样（1=全部，2=每隔一个取一个...）。
        device: CoTracker 计算设备。
        cotracker: 可选，预加载的 CoTracker 模型。

    Returns:
        Rendered RGB PIL Image 列表。
    """
    import torch

    T = len(frames)

    # ---- 计算轨迹 (如果未提供) ----
    if tracks is None or visibility is None:
        if cotracker is None:
            cotracker = load_cotracker(device=device)
        if cotracker is None:
            print("  [WARN] CoTracker 不可用，跳过轨迹预览渲染")
            return frames
        tracks, visibility = compute_tracks(cotracker, frames, device=device)

    # ---- 下采样轨迹点 ----
    if subsample_tracks > 1:
        tracks = tracks[:, ::subsample_tracks, :]
        visibility = visibility[:, ::subsample_tracks]

    T, N, _ = tracks.shape

    # ---- 生成颜色 ----
    colors = _generate_colors(N)

    # ---- 渲染每一帧 ----
    rendered: List[Image.Image] = []
    for t in range(T):
        frame = frames[t].convert("RGB")
        draw = ImageDraw.Draw(frame)

        start_t = max(0, t - trail_length + 1)

        for n in range(N):
            if not visibility[t, n]:
                continue
            color = colors[n % len(colors)]

            # 绘制轨迹线 (从 start_t 到 t)
            for pt in range(start_t, t):
                if visibility[pt, n] and visibility[pt + 1, n]:
                    x1, y1 = float(tracks[pt, n, 0]), float(tracks[pt, n, 1])
                    x2, y2 = float(tracks[pt + 1, n, 0]), float(tracks[pt + 1, n, 1])
                    if all(v >= 0 for v in [x1, y1, x2, y2]):
                        # 越老的轨迹线越淡
                        alpha = 0.3 + 0.7 * (pt - start_t) / max(t - start_t, 1)
                        faded_color = tuple(int(c * alpha) for c in color)
                        draw.line([(x1, y1), (x2, y2)], fill=faded_color, width=line_width)

            # 绘制当前帧的点
            x, y = float(tracks[t, n, 0]), float(tracks[t, n, 1])
            if x >= 0 and y >= 0:
                draw.ellipse(
                    [
                        (x - point_radius, y - point_radius),
                        (x + point_radius, y + point_radius),
                    ],
                    fill=color,
                    outline="white",
                    width=1,
                )

        rendered.append(frame)

    return rendered


def save_trajectory_preview_video(
    frames: List[Image.Image],
    output_path: str,
    fps: int = 15,
    quality: int = 8,
) -> str:
    """
    将轨迹预览帧保存为视频文件。

    Args:
        frames: 渲染后的帧列表 (RGB PIL Image)。
        output_path: 输出视频路径。
        fps: 帧率。
        quality: 视频质量 (0-10, 越高越好)。

    Returns:
        输出路径。
    """
    from diffsynth.data.video import save_video

    output_path = str(output_path)
    save_video(frames, output_path, fps=fps, quality=quality)
    print(f"  轨迹预览视频已保存: {output_path}")
    return output_path


# ═══════════════════════════════════════════════════════════════
#  从保存的 bbox_mask.pt / track_video.pt 生成预览
# ═══════════════════════════════════════════════════════════════

_BBOX_COLORS = [
    (255, 80, 80),    # 红
    (80, 255, 80),    # 绿
    (80, 130, 255),   # 蓝
]


def _find_bbox_from_mask(mask_slice: torch.Tensor) -> Optional[Tuple[int, int, int, int]]:
    """从单帧单通道的 mask (H, W) 中提取非零区域的 bounding box。

    Returns:
        (x1, y1, x2, y2) 或 None（无有效区域）。
    """
    non_zero = torch.nonzero(mask_slice > 0)
    if non_zero.shape[0] == 0:
        return None
    y_min = int(non_zero[:, 0].min().item())
    y_max = int(non_zero[:, 0].max().item())
    x_min = int(non_zero[:, 1].min().item())
    x_max = int(non_zero[:, 1].max().item())
    return x_min, y_min, x_max, y_max


def render_from_bbox_mask(
    frames: List[Image.Image],
    bbox_mask: torch.Tensor,
    colors: Optional[List[Tuple[int, int, int]]] = None,
    line_width: int = 3,
    fill_alpha: float = 0.15,
    show_label: bool = True,
) -> List[Image.Image]:
    """在视频帧上叠加 bbox_mask 中物体区域的边框可视化。

    `bbox_mask.pt` 是 `build_bbox_mask_from_json_str` 的输出，
    形状为 `(1, 3, T, H, W)`，值域约 [-1, 1]（>0 表示物体区域）。
    每个 channel 对应一个物体，逐帧插值得到平滑的 bbox 运动。

    Args:
        frames: 原始视频帧 (RGB PIL Image 列表)。
        bbox_mask: (1, 3, T, H, W) 张量。
        colors: 每个物体的颜色列表，默认使用红/绿/蓝。
        line_width: 边框线宽（像素）。
        fill_alpha: 半透明填充的 alpha 值。
        show_label: 是否在边框左上角显示物体编号。

    Returns:
        渲染后的 RGB PIL Image 列表。
    """
    if colors is None:
        colors = _BBOX_COLORS

    T = len(frames)
    # bbox_mask: (1, 3, T, H, W) -> (3, T, H, W)
    mask = bbox_mask.squeeze(0)
    n_channels = min(mask.shape[0], len(colors))

    # 预计算每帧每个物体的 bbox
    per_frame_bboxes: List[List[Optional[Tuple[int, int, int, int]]]] = []
    for t in range(T):
        frame_bboxes: List[Optional[Tuple[int, int, int, int]]] = []
        for c in range(n_channels):
            bbox = _find_bbox_from_mask(mask[c, t])
            frame_bboxes.append(bbox)
        per_frame_bboxes.append(frame_bboxes)

    rendered: List[Image.Image] = []
    for t in range(T):
        # 转 RGBA 以支持半透明绘制
        frame_copy = frames[t].convert("RGBA").copy()
        draw = ImageDraw.Draw(frame_copy)

        for obj_idx, bbox in enumerate(per_frame_bboxes[t]):
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            color = colors[obj_idx % len(colors)]
            fill_color = color + (int(255 * fill_alpha),)

            # 半透明填充
            draw.rectangle([x1, y1, x2, y2], fill=fill_color, outline=None)
            # 实线边框
            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

            # 标签
            if show_label:
                label = f"Obj {obj_idx + 1}"
                left, top, right, bottom = draw.textbbox((0, 0), label)
                tw = right - left
                th = bottom - top
                label_bg_color = color + (200,)
                draw.rectangle(
                    [x1, y1 - th - 4, x1 + tw + 6, y1],
                    fill=label_bg_color,
                )
                draw.text((x1 + 3, y1 - th - 2), label, fill="white")

        # 转回 RGB
        rendered.append(frame_copy.convert("RGB"))

    return rendered


def render_track_video_heatmap(
    frames: List[Image.Image],
    track_video: torch.Tensor,
    alpha: float = 0.35,
    colormap: str = "viridis",
) -> List[Image.Image]:
    """将 track_video 特征图的激活强度可视化为热力图叠加层。

    `track_video.pt` 是 `build_track_video_from_tracks` 输出的特征图，
    形状为 `(1, C, T', H', W')`（C=64, T'≈T/4, H'=H/8, W'=W/8）。
    通过对 channel 维度求 norm 得到激活强度，再上采样到原始分辨率。

    Args:
        frames: 原始视频帧 (RGB PIL Image 列表)。
        track_video: (1, C, T', H', W') 张量。
        alpha: 热力图叠加透明度。
        colormap: matplotlib colormap 名称（'viridis', 'jet', 'plasma' 等）。

    Returns:
        渲染后的 RGB PIL Image 列表。
    """
    import torch.nn.functional as F

    T = len(frames)
    H, W = frames[0].height, frames[0].width

    # (1, C, T', H', W') -> 沿 channel 求 norm -> (T', H', W')
    activation = track_video.squeeze(0).norm(dim=0)  # (T', H', W')

    # 确保 float32（bfloat16 不支持 CPU interpolate）
    if activation.dtype != torch.float32:
        activation = activation.to(torch.float32)

    # 归一化到 [0, 1]
    act_min = activation.min()
    act_max = activation.max()
    if act_max > act_min:
        activation = (activation - act_min) / (act_max - act_min)
    else:
        activation = torch.zeros_like(activation)

    # Resample 到原始时间/空间分辨率
    T_prime, H_prime, W_prime = activation.shape
    act_5d = activation.unsqueeze(0).unsqueeze(0)  # (1, 1, T', H', W')
    act_5d = F.interpolate(
        act_5d, size=(T, H, W), mode="trilinear", align_corners=False,
    )
    act_map = act_5d.squeeze().cpu().numpy()  # (T, H, W)

    # 加载 matplotlib colormap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm

    cmap = cm.get_cmap(colormap)

    rendered: List[Image.Image] = []
    for t in range(T):
        frame = frames[t].convert("RGBA").copy()

        # 当前帧的激活图 (H, W)
        heat = act_map[t]
        heat_rgba = (cmap(heat, alpha=alpha) * 255).astype(np.uint8)  # (H, W, 4)
        heat_img = Image.fromarray(heat_rgba, "RGBA")

        # 叠加到原帧（heat_img 的 alpha 通道控制透明度）
        frame.alpha_composite(heat_img)
        rendered.append(frame.convert("RGB"))

    return rendered


def render_signals_preview(
    frames: List[Image.Image],
    bbox_mask_path: Optional[str] = None,
    track_video_path: Optional[str] = None,
    output_path: Optional[str] = None,
    fps: int = 15,
    quality: int = 8,
    show_bbox: bool = True,
    show_heatmap: bool = True,
    bbox_line_width: int = 3,
    heatmap_alpha: float = 0.35,
    heatmap_colormap: str = "viridis",
) -> List[Image.Image]:
    """从保存的 bbox_mask.pt / track_video.pt 加载信号并渲染预览。

    这是评估流程中替代 CoTracker 轨迹追踪的轻量预览方式：
    - bbox_mask.pt → 绘制物体边界框（反映 bbox 关键帧的运动）
    - track_video.pt → 绘制特征激活热力图（反映轨迹信号强度）

    Args:
        frames: 原始视频帧列表。
        bbox_mask_path: bbox_mask.pt 路径（可选）。
        track_video_path: track_video.pt 路径（可选）。
        output_path: 可选，预览视频保存路径。
        fps: 预览视频帧率。
        quality: 视频质量。
        show_bbox: 是否显示 bbox 框。
        show_heatmap: 是否显示热力图。
        bbox_line_width: bbox 边框宽度。
        heatmap_alpha: 热力图透明度。
        heatmap_colormap: 热力图 colormap。

    Returns:
        渲染后的帧列表。
    """
    rendered = [f.convert("RGB").copy() for f in frames]

    # 1. bbox_mask 可视化
    if show_bbox and bbox_mask_path and os.path.exists(bbox_mask_path):
        print(f"  加载 bbox_mask: {bbox_mask_path}")
        try:
            bbox_mask = torch.load(bbox_mask_path, map_location="cpu", weights_only=True)
            if isinstance(bbox_mask, dict):
                # 某些旧版本保存格式可能是 dict
                bbox_mask = bbox_mask.get("bbox_mask", list(bbox_mask.values())[0])
            bbox_frames = render_from_bbox_mask(
                rendered, bbox_mask, line_width=bbox_line_width,
            )
            rendered = bbox_frames
            print(f"  ✓ bbox 框可视化完成")
        except Exception as e:
            print(f"  [WARN] bbox_mask 可视化失败: {e}")

    # 2. track_video 热力图
    if show_heatmap and track_video_path and os.path.exists(track_video_path):
        print(f"  加载 track_video: {track_video_path}")
        try:
            track_video = torch.load(track_video_path, map_location="cpu", weights_only=True)
            if isinstance(track_video, dict):
                track_video = track_video.get("track_video", list(track_video.values())[0])
            heatmap_frames = render_track_video_heatmap(
                rendered, track_video, alpha=heatmap_alpha, colormap=heatmap_colormap,
            )
            rendered = heatmap_frames
            print(f"  ✓ track_video 热力图完成")
        except Exception as e:
            print(f"  [WARN] track_video 热力图失败: {e}")

    # 3. 保存视频
    if output_path:
        save_trajectory_preview_video(rendered, output_path, fps=fps, quality=quality)

    return rendered
