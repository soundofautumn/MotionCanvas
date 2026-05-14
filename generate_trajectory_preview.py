#!/usr/bin/env python3
"""
直接从已保存的 bbox_mask.pt / track_video.pt 生成轨迹预览视频。

不需要 GPU、不需要 CoTracker，在 CPU 上即可快速运行。

用法:
  # 扫描 ablation_results/ 下所有实验，生成预览
  python generate_trajectory_preview.py --dir ablation_results

  # 指定单个实验目录
  python generate_trajectory_preview.py --dir ablation_results/05_cat_llm

  # 指定输出 fps
  python generate_trajectory_preview.py --dir ablation_results --fps 15

  # 关闭热力图 / 关闭 bbox 框
  python generate_trajectory_preview.py --dir ablation_results --no-heatmap
  python generate_trajectory_preview.py --dir ablation_results --no-bbox
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from PIL import Image

# 确保项目根目录在路径中
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from evaluation.trajectory_preview import render_signals_preview


def load_video_frames(video_path: str) -> list:
    """加载视频全部帧为 PIL Image 列表。"""
    import decord
    decord.bridge.set_bridge("torch")
    vr = decord.VideoReader(video_path)
    frames = vr.get_batch(list(range(len(vr)))).numpy()
    return [Image.fromarray(f) for f in frames]


def process_experiment(
    exp_dir: Path,
    fps: int = 15,
    quality: int = 8,
    show_bbox: bool = True,
    show_heatmap: bool = True,
    bbox_line_width: int = 3,
    heatmap_alpha: float = 0.35,
    force: bool = False,
) -> bool:
    """处理单个实验目录，生成轨迹预览视频。

    Returns:
        True 表示成功处理，False 表示跳过或失败。
    """
    name = exp_dir.name

    # 检查必要文件
    video_path = exp_dir / "output.mp4"
    bbox_pt = exp_dir / "bbox_mask.pt"
    track_pt = exp_dir / "track_video.pt"

    if not video_path.exists():
        print(f"  [SKIP] {name}: 缺少 output.mp4")
        return False

    if not bbox_pt.exists() and not track_pt.exists():
        print(f"  [SKIP] {name}: 缺少 bbox_mask.pt 和 track_video.pt")
        return False

    output_path = exp_dir / "trajectory_preview.mp4"
    if output_path.exists() and not force:
        print(f"  [SKIP] {name}: trajectory_preview.mp4 已存在 (加 --force 覆盖)")
        return False

    print(f"\n{'='*50}")
    print(f"  Experiment: {name}")
    print(f"  Video:      {video_path}")
    print(f"  bbox_mask:  {'✓' if bbox_pt.exists() else '✗'}  "
          f"({bbox_pt.stat().st_size / 1024 / 1024:.0f} MB)" if bbox_pt.exists() else "")
    print(f"  track_video: {'✓' if track_pt.exists() else '✗'}  "
          f"({track_pt.stat().st_size / 1024 / 1024:.1f} MB)" if track_pt.exists() else "")
    print(f"{'='*50}")

    # 加载视频帧
    print(f"  Loading video frames...")
    frames = load_video_frames(str(video_path))
    print(f"  Frames: {len(frames)}")

    # 生成预览
    print(f"  Generating preview...")
    render_signals_preview(
        frames,
        bbox_mask_path=str(bbox_pt) if bbox_pt.exists() else None,
        track_video_path=str(track_pt) if track_pt.exists() else None,
        output_path=str(output_path),
        fps=fps,
        quality=quality,
        show_bbox=show_bbox,
        show_heatmap=show_heatmap,
        bbox_line_width=bbox_line_width,
        heatmap_alpha=heatmap_alpha,
    )
    print(f"  Done: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="从已保存的 bbox_mask.pt / track_video.pt 生成轨迹预览视频"
    )
    parser.add_argument(
        "--dir", "-d", required=True,
        help="实验根目录（扫描其下所有子目录）或单个实验目录的路径",
    )
    parser.add_argument(
        "--fps", type=int, default=15,
        help="预览视频帧率 (默认: 15)",
    )
    parser.add_argument(
        "--quality", type=int, default=8,
        help="视频质量 0-10 (默认: 8)",
    )
    parser.add_argument(
        "--no-bbox", action="store_true",
        help="不显示 bbox 框",
    )
    parser.add_argument(
        "--no-heatmap", action="store_true",
        help="不显示热力图",
    )
    parser.add_argument(
        "--bbox-line-width", type=int, default=3,
        help="bbox 边框宽度 (默认: 3)",
    )
    parser.add_argument(
        "--heatmap-alpha", type=float, default=0.35,
        help="热力图透明度 (默认: 0.35)",
    )
    parser.add_argument(
        "--force", "-f", action="store_true",
        help="覆盖已存在的 trajectory_preview.mp4",
    )
    args = parser.parse_args()

    target = Path(args.dir)
    if not target.exists():
        print(f"错误: 路径不存在 {target}")
        sys.exit(1)

    # 判断是单个实验目录还是根目录
    has_video = list(target.glob("output.mp4"))
    has_pt = list(target.glob("bbox_mask.pt")) or list(target.glob("track_video.pt"))

    if has_video and has_pt:
        # 单个实验目录
        exp_dirs = [target]
    else:
        # 根目录, 扫描子目录
        exp_dirs = sorted([
            d for d in target.iterdir()
            if d.is_dir()
            and (d / "output.mp4").exists()
            and ((d / "bbox_mask.pt").exists() or (d / "track_video.pt").exists())
        ])
        if not exp_dirs:
            # 放宽条件，只要有 output.mp4 就尝试
            exp_dirs = sorted([
                d for d in target.iterdir()
                if d.is_dir() and (d / "output.mp4").exists()
            ])
            if not exp_dirs:
                print(f"错误: 在 {target} 中未找到包含 output.mp4 的实验目录")
                sys.exit(1)

    print(f"找到 {len(exp_dirs)} 个实验:")
    for d in exp_dirs:
        print(f"  - {d.name}")

    success = 0
    for exp_dir in exp_dirs:
        ok = process_experiment(
            exp_dir,
            fps=args.fps,
            quality=args.quality,
            show_bbox=not args.no_bbox,
            show_heatmap=not args.no_heatmap,
            bbox_line_width=args.bbox_line_width,
            heatmap_alpha=args.heatmap_alpha,
            force=args.force,
        )
        if ok:
            success += 1

    print(f"\n完成: {success}/{len(exp_dirs)} 个实验生成了轨迹预览视频")


if __name__ == "__main__":
    main()
