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
import json as _json
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
    # pred_visibility: (1, T, N) 或 (1, T, N, 1)
    tracks = pred_tracks[0].cpu().numpy()  # (T, N, 2)
    if pred_visibility.ndim == 4:
        visibility = pred_visibility[0, :, :, 0].cpu().numpy()
    else:
        visibility = pred_visibility[0].cpu().numpy()  # (T, N)

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
    每个追踪点的轨迹从起点（第一帧可见位置）到当前位置绘制一条直线，
    并在当前位置画一个带白色描边的圆点。

    Args:
        frames: 原始视频帧 (RGB PIL Image 列表)。
        tracks: 可选，预计算轨迹 (T, N, 2) 像素坐标。
        visibility: 可选，预计算可见性 (T, N) bool。
        trail_length: 保留（仅用于向后兼容）。
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

    # ---- 预计算每根轨迹的起点（第一个可见帧的位置） ----
    start_positions = []
    for n in range(N):
        start_idx = None
        for t in range(T):
            if visibility[t, n]:
                start_idx = t
                break
        if start_idx is not None:
            start_positions.append(tracks[start_idx, n])
        else:
            start_positions.append(None)

    # ---- 渲染每一帧 ----
    rendered: List[Image.Image] = []
    for t in range(T):
        frame = frames[t].convert("RGB")
        draw = ImageDraw.Draw(frame)

        for n in range(N):
            if not visibility[t, n]:
                continue
            color = colors[n % len(colors)]

            # 起点 → 当前点 直线
            start_pos = start_positions[n]
            curr_x, curr_y = float(tracks[t, n, 0]), float(tracks[t, n, 1])
            if start_pos is not None and curr_x >= 0 and curr_y >= 0:
                sx, sy = float(start_pos[0]), float(start_pos[1])
                if sx >= 0 and sy >= 0:
                    draw.line([(sx, sy), (curr_x, curr_y)], fill=color, width=line_width)

            # 绘制当前帧的点（带白色描边）
            if curr_x >= 0 and curr_y >= 0:
                draw.ellipse(
                    [
                        (curr_x - point_radius, curr_y - point_radius),
                        (curr_x + point_radius, curr_y + point_radius),
                    ],
                    fill=color,
                    outline=(255, 255, 255),
                    width=2,
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


def _find_connected_bboxes(mask_slice: torch.Tensor) -> List[Tuple[int, int, int, int]]:
    """从单帧 mask (H, W) 中找出所有连通区域的外接矩形。

    使用简单的行扫描算法分离不同的物体区域。
    """
    binary = (mask_slice > 0).cpu().numpy()
    if not binary.any():
        return []

    from scipy import ndimage
    labeled, num_features = ndimage.label(binary)
    bboxes = []
    for i in range(1, num_features + 1):
        ys, xs = np.where(labeled == i)
        if len(ys) < 5:  # 过滤噪声点
            continue
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        if x2 > x1 and y2 > y1:
            bboxes.append((x1, y1, x2, y2))
    return bboxes


def render_from_bbox_mask(
    frames: List[Image.Image],
    bbox_mask: torch.Tensor,
    colors: Optional[List[Tuple[int, int, int]]] = None,
    line_width: int = 3,
    fill_alpha: float = 0.2,
    show_label: bool = False,
) -> List[Image.Image]:
    """在视频帧上叠加 bbox_mask 中物体区域的边框可视化——与 Gradio GUI 风格一致。

    `bbox_mask.pt` 是 `build_bbox_mask_from_json_str` 的输出，
    形状为 `(1, 3, T, H, W)`，值域约 [-1, 1]（>0 表示物体区域）。

    渲染风格（完全对齐 GUI `preview_control_video`）：
    - 红色半透明填充 + 红色实线边框
    - 橙色中心轨迹拖尾（渐隐效果）
    - 使用连通域分析分离不同物体

    Args:
        frames: 原始视频帧 (RGB PIL Image 列表)。
        bbox_mask: (1, 3, T, H, W) 张量。
        colors: 每个物体的颜色列表，默认红色 (255,80,80)。
        line_width: 边框线宽（像素），GUI 默认 3。
        fill_alpha: 半透明填充的 alpha 值，GUI 默认 0.2（RGBA 50/255）。
        show_label: 是否显示物体编号标签（GUI 默认不显示）。
        draw_trajectory: 是否绘制中心轨迹拖尾。
        trail_length: 轨迹拖尾最大帧数，GUI 默认 15。
        trajectory_color: 轨迹线颜色，GUI 使用橙黄色 (255, 200, 80)。

    Returns:
        渲染后的 RGB PIL Image 列表。
    """
    if colors is None:
        colors = [(255, 80, 80)]

    T = len(frames)
    # bbox_mask: (1, 3, T, H, W) — 所有 3 个 channel 内容相同（merged）
    # 使用 channel 0 做连通域分析分离物体
    mask_2d = bbox_mask[0, 0]  # (T, H, W)

    # 预计算每帧每个连通域的 bbox
    per_frame_bboxes: List[List[Tuple[int, int, int, int]]] = []
    per_frame_centers: List[List[Optional[Tuple[float, float]]]] = []
    for t in range(T):
        bboxes = _find_connected_bboxes(mask_2d[t])
        per_frame_bboxes.append(bboxes)
        centers = []
        for x1, y1, x2, y2 in bboxes:
            centers.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))
        per_frame_centers.append(centers)

    # 跨帧匹配物体（基于最近中心距离）
    # 简单策略: 按第一帧的物体数量确定 N_objects，逐帧按最近距离匹配
    n_objects = max(len(bboxes) for bboxes in per_frame_bboxes) if per_frame_bboxes else 0
    if n_objects == 0:
        return [f.convert("RGB") for f in frames]

    # 为每个物体构建跨帧的 center 轨迹
    obj_centers: List[List[Optional[Tuple[float, float]]]] = [
        [None] * T for _ in range(n_objects)
    ]

    # 第一帧直接分配
    for oi in range(min(n_objects, len(per_frame_centers[0]))):
        obj_centers[oi][0] = per_frame_centers[0][oi]

    # 后续帧按最近距离匹配
    for t in range(1, T):
        prev_centers = [obj_centers[oi][t - 1] for oi in range(n_objects)
                       if obj_centers[oi][t - 1] is not None]
        curr_bboxes = per_frame_bboxes[t]
        curr_centers = per_frame_centers[t]
        assigned_curr = set()
        assigned_obj = set()

        # 对每个有前一帧位置的物体，找最近的当前帧中心
        candidates = []
        for oi in range(n_objects):
            prev = obj_centers[oi][t - 1]
            if prev is None:
                continue
            for ci, cc in enumerate(curr_centers):
                if ci in assigned_curr:
                    continue
                d = (prev[0] - cc[0]) ** 2 + (prev[1] - cc[1]) ** 2
                candidates.append((d, oi, ci))
        candidates.sort()
        for d, oi, ci in candidates:
            if oi in assigned_obj or ci in assigned_curr:
                continue
            obj_centers[oi][t] = curr_centers[ci]
            assigned_obj.add(oi)
            assigned_curr.add(ci)

        # 未匹配的当前帧中心 -> 新物体（补到后面）
        next_oi = n_objects
        for ci, cc in enumerate(curr_centers):
            if ci in assigned_curr:
                continue
            if next_oi >= len(obj_centers):
                obj_centers.append([None] * T)
            obj_centers[next_oi][t] = cc
            next_oi += 1

    # 渲染
    rendered: List[Image.Image] = []
    for t in range(T):
        # 背景层（原始帧）
        bg = frames[t].convert("RGBA").copy()
        # 叠加层（全透明，在上面画填充和线条）
        overlay = Image.new("RGBA", bg.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        bboxes = per_frame_bboxes[t]

        for obj_idx in range(min(len(bboxes), len(colors))):
            x1, y1, x2, y2 = bboxes[obj_idx]
            color = colors[obj_idx % len(colors)]
            fill_color = color + (int(255 * fill_alpha),)

            # 半透明填充（画在叠加层上）
            draw.rectangle([x1, y1, x2, y2], fill=fill_color, outline=None)

        # 合成：叠加层在背景之上
        bg.alpha_composite(overlay)

        # 边框直接画在合成后的图像上（无 alpha，清晰可见）
        draw_final = ImageDraw.Draw(bg)
        for obj_idx in range(min(len(bboxes), len(colors))):
            x1, y1, x2, y2 = bboxes[obj_idx]
            color = colors[obj_idx % len(colors)]
            draw_final.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        rendered.append(bg.convert("RGB"))

    return rendered


def render_track_video_grid(
    frames: List[Image.Image],
    track_video: torch.Tensor,
    num_points: int = 200,
    point_radius: int = 2,
    color: Tuple[int, int, int] = (80, 160, 255),
    min_activation: float = 0.1,
) -> List[Image.Image]:
    """将 track_video 特征图的激活强度渲染为离散轨迹点网格。

    与 GUI `preview_control_video` 中的蓝色背景网格点效果一致：
    从特征图中提取激活最强的 N 个点位置，绘制为彩色圆点。
    如果包含相机运动，这些点会随镜头参数移动，形成"网格跟随相机"的视觉效果。

    Args:
        frames: 原始视频帧 (RGB PIL Image 列表)。
        track_video: (1, C, T', H', W') 张量。
        num_points: 每帧显示的轨迹点数量，GUI 网格约 200-400 个。
        point_radius: 点半径（像素），GUI 使用 2px。
        color: 点颜色，GUI 使用蓝色 (80, 160, 255)。
        min_activation: 最小激活阈值，低于此值的点不显示。

    Returns:
        渲染后的 RGB PIL Image 列表。
    """
    import torch.nn.functional as F

    T = len(frames)
    H, W = frames[0].height, frames[0].width

    # (1, C, T', H', W') -> 沿 channel 求 norm -> (T', H', W')
    activation = track_video.squeeze(0).norm(dim=0)  # (T', H', W')
    if activation.dtype != torch.float32:
        activation = activation.to(torch.float32)

    # 归一化到 [0, 1]
    act_min = activation.min()
    act_max = activation.max()
    if act_max > act_min:
        activation = (activation - act_min) / (act_max - act_min)
    else:
        return [f.convert("RGB") for f in frames]

    # 在低分辨率格点上直接采样（避免上采样模糊导致点聚集）
    # track_video 特征图尺寸为 (1, C, T', H', W')，格点步长 = 原图 / 特征图
    _, C, Tp, Hp, Wp = track_video.shape
    act_low = activation.cpu().numpy()  # (Tp, Hp, Wp)
    # 每个格点覆盖的像素区域
    scale_y = H / Hp
    scale_x = W / Wp

    rendered: List[Image.Image] = []
    for t in range(T):
        # 找到当前时间步对应的特征帧（线性映射）
        tf = min(int(round(t * (Tp - 1) / max(T - 1, 1))), Tp - 1)

        frame = frames[t].convert("RGB").copy()
        draw = ImageDraw.Draw(frame)

        # 在 Hp x Wp 的格点中，找到激活 > 阈值的格点中心坐标
        mask = act_low[tf] > min_activation
        if not mask.any():
            # 保底：取最强激活的一定数量格点
            flat = act_low[tf].flatten()
            threshold = np.sort(flat)[::-1][min(num_points, len(flat) - 1)]
            mask = act_low[tf] > threshold

        count = 0
        for hy in range(Hp):
            for wx in range(Wp):
                if not mask[hy, wx]:
                    continue
                # 格点中心在原图中的像素坐标
                cx = int((wx + 0.5) * scale_x)
                cy = int((hy + 0.5) * scale_y)
                draw.ellipse(
                    [(cx - point_radius, cy - point_radius),
                     (cx + point_radius, cy + point_radius)],
                    fill=color,
                )
                count += 1
                if count >= num_points:
                    break
            if count >= num_points:
                break

        rendered.append(frame)

    return rendered


def render_signals_preview(
    frames: List[Image.Image],
    bbox_mask_path: Optional[str] = None,
    track_video_path: Optional[str] = None,
    output_path: Optional[str] = None,
    fps: int = 15,
    quality: int = 8,
    show_bbox: bool = True,
    show_heatmap: bool = False,
    bbox_line_width: int = 3,
    heatmap_alpha: float = 0.35,
    heatmap_colormap: str = "viridis",
) -> List[Image.Image]:
    """从保存的 bbox_mask.pt / track_video.pt 加载信号并渲染预览。

    Bbox 渲染风格完全对齐 Gradio GUI 的 `preview_control_video`：
    - 红色 (255,80,80) 半透明填充 + 红色实线边框 (width=3)
    - 连通域分离多物体
    - 蓝色 (80,160,255) 轨迹点网格从 track_video 特征图提取激活强度，
      效果等同 GUI 的相机运动背景网格点

    Args:
        frames: 原始视频帧列表。
        bbox_mask_path: bbox_mask.pt 路径（可选）。
        track_video_path: track_video.pt 路径（可选）。
        output_path: 可选，预览视频保存路径。
        fps: 预览视频帧率。
        quality: 视频质量。
        show_bbox: 是否显示 bbox 框（GUI 默认显示）。
        show_heatmap: 是否显示轨迹点网格（GUI 默认显示背景网格）。
        bbox_line_width: bbox 边框宽度，GUI 默认 3。
        heatmap_alpha: 热力图透明度。
        heatmap_colormap: 热力图 colormap。

    Returns:
        渲染后的帧列表。
    """
    rendered = [f.convert("RGB").copy() for f in frames]

    # 1. bbox_mask 可视化（GUI 风格：红色填充+边框+橙色中心轨迹）
    if show_bbox and bbox_mask_path and os.path.exists(bbox_mask_path):
        print(f"  加载 bbox_mask: {bbox_mask_path}")
        try:
            bbox_mask = torch.load(bbox_mask_path, map_location="cpu", weights_only=True)
            if isinstance(bbox_mask, dict):
                bbox_mask = bbox_mask.get("bbox_mask", list(bbox_mask.values())[0])
            bbox_frames = render_from_bbox_mask(
                rendered, bbox_mask,
                line_width=bbox_line_width,
            )
            rendered = bbox_frames
            print(f"  ✓ bbox 框可视化完成")
        except Exception as e:
            print(f"  [WARN] bbox_mask 可视化失败: {e}")

    # 2. track_video 轨迹点网格（GUI 风格的蓝色网格点，反映相机运动）
    if show_heatmap and track_video_path and os.path.exists(track_video_path):
        print(f"  加载 track_video: {track_video_path}")
        try:
            track_video = torch.load(track_video_path, map_location="cpu", weights_only=True)
            if isinstance(track_video, dict):
                track_video = track_video.get("track_video", list(track_video.values())[0])
            grid_frames = render_track_video_grid(
                rendered, track_video, num_points=250, point_radius=2,
            )
            rendered = grid_frames
            print(f"  ✓ track_video 轨迹点网格完成")
        except Exception as e:
            print(f"  [WARN] track_video 轨迹点网格失败: {e}")

    # 3. 保存视频
    if output_path:
        save_trajectory_preview_video(rendered, output_path, fps=fps, quality=quality)

    return rendered


def _build_point_tracks_from_json(json_str: str, num_frames: int, height: int, width: int) -> Optional[List]:
    """从 point_json 构建插值后的逐帧轨迹（与 GUI 逻辑一致）。"""
    if not json_str or not json_str.strip():
        return None
    data = _json.loads(json_str)
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
    return tracks if tracks else None


def _build_camera_params(json_str: str, num_frames: int) -> Optional[List[dict]]:
    """从 camera_json 构建插值后的逐帧相机参数（与 GUI 逻辑一致）。"""
    if not json_str or not json_str.strip():
        return None
    try:
        data = _json.loads(json_str)
        kfs = data.get("camera", {}).get("keyframes", [])
    except Exception:
        return None
    if not kfs:
        return None
    kf_dict = {}
    for kf in kfs:
        fi = int(kf.get("frame", 0))
        kf_dict[fi] = {
            "zoom": float(kf.get("zoom", 1.0)),
            "pan_x": float(kf.get("pan", [0, 0])[0]),
            "pan_y": float(kf.get("pan", [0, 0])[1]),
            "rotation": float(kf.get("rotation", 0)),
        }
    idxs = sorted(kf_dict.keys())
    params = []
    for fi in range(num_frames):
        prev_idx = next_idx = idxs[0]
        for idx in idxs:
            if idx <= fi:
                prev_idx = idx
            if idx >= fi and (next_idx == idxs[0] or idx < next_idx):
                next_idx = idx
        if prev_idx == next_idx:
            params.append(kf_dict[prev_idx])
        else:
            t = (fi - prev_idx) / max(1, next_idx - prev_idx)
            p = kf_dict[prev_idx]
            n = kf_dict[next_idx]
            params.append({
                "zoom": p["zoom"] * (1 - t) + n["zoom"] * t,
                "pan_x": p["pan_x"] * (1 - t) + n["pan_x"] * t,
                "pan_y": p["pan_y"] * (1 - t) + n["pan_y"] * t,
                "rotation": p["rotation"] * (1 - t) + n["rotation"] * t,
            })
    return params


def _apply_camera_to_point(x: float, y: float, width: int, height: int,
                            zoom: float, pan_x: float, pan_y: float, rotation: float) -> Tuple[float, float]:
    """对单个点应用相机变换（与 GUI 逻辑一致）。"""
    cx, cy = width / 2.0, height / 2.0
    dx = (x - cx) * zoom
    dy = (y - cy) * zoom
    theta = math.radians(rotation)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    rx = dx * cos_t - dy * sin_t
    ry = dx * sin_t + dy * cos_t
    return rx + cx + pan_x, ry + cy + pan_y


def render_point_tracks_from_config(
    frames: List[Image.Image],
    point_json_str: str,
    camera_json_str: str = "",
    num_frames: int = 49,
    height: int = 480,
    width: int = 832,
    line_width: int = 3,
    point_color: Tuple[int, int, int] = (80, 160, 255),
    outline_color: Tuple[int, int, int] = (255, 255, 255),
) -> List[Image.Image]:
    """从 config.json 的原始点轨迹数据渲染轨迹预览（与 GUI 效果一致）。

    解析 point_json 中的点关键帧，插值得到逐帧位置，再应用相机变换，
    绘制起点→当前点直线 + 白色描边圆。

    Args:
        frames: 视频帧列表。
        point_json_str: point_json 字符串（含 points 数组）。
        camera_json_str: camera_json 字符串（含 camera.keyframes）。
        num_frames: 帧数。
        height: 画面高度。
        width: 画面宽度。
        line_width: 轨迹线宽度。
        point_color: 轨迹点填充颜色。
        outline_color: 轨迹点描边颜色。

    Returns:
        渲染后的帧列表。
    """
    # 构建相机参数和点轨迹
    camera_params = _build_camera_params(camera_json_str, num_frames)
    local_tracks = _build_point_tracks_from_json(point_json_str, num_frames, height, width)

    if not local_tracks:
        return frames

    rendered = [f.convert("RGB").copy() for f in frames]

    for t in range(num_frames):
        if t >= len(rendered):
            break
        draw = ImageDraw.Draw(rendered[t])

        for track in local_tracks:
            if len(track) < 2 or t < 1:
                continue
            start_p = track[0]
            curr_p = track[min(t, len(track) - 1)]

            cp_start = camera_params[0] if camera_params else None
            cp_curr = camera_params[min(t, len(camera_params) - 1)] if camera_params else None

            if cp_start and cp_curr:
                sx, sy = _apply_camera_to_point(
                    start_p[0], start_p[1], width, height,
                    cp_start["zoom"], cp_start["pan_x"], cp_start["pan_y"], cp_start["rotation"],
                )
                cx, cy = _apply_camera_to_point(
                    curr_p[0], curr_p[1], width, height,
                    cp_curr["zoom"], cp_curr["pan_x"], cp_curr["pan_y"], cp_curr["rotation"],
                )
                draw.line([(sx, sy), (cx, cy)], fill=point_color, width=line_width)
                draw.ellipse(
                    [cx - 5, cy - 5, cx + 5, cy + 5],
                    fill=point_color, outline=outline_color, width=2,
                )

    return rendered
