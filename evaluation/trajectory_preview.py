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
