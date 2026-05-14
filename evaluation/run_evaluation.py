import argparse
import json
import os
import sys
from pathlib import Path
from PIL import Image
from typing import List, Optional

import torch

from .image_quality_metrics import ImageQualityEvaluator, uniform_sample
from .reference_metrics import ReferenceMetrics


def load_video_frames(video_path: str, max_frames: int = -1) -> List[Image.Image]:
    import decord
    decord.bridge.set_bridge("torch")
    vr = decord.VideoReader(video_path)
    total = len(vr)
    if max_frames > 0:
        indices = [int(i * (total - 1) / (max_frames - 1)) for i in range(max_frames)]
    else:
        indices = list(range(total))
    frames = vr.get_batch(indices).numpy()
    return [Image.fromarray(f) for f in frames]


def evaluate_video(
    video_path: str,
    prompt: str,
    reference_path: str = None,
    sample_n: int = 8,
    device: str = "cuda",
    quality_models: List[str] = None,
    trajectory_preview: Optional[str] = None,
    fps: int = 15,
    trail_length: int = 8,
    subsample_tracks: int = 3,
    bbox_mask_path: Optional[str] = None,
    track_video_path: Optional[str] = None,
) -> dict:
    frames = load_video_frames(video_path, max_frames=-1)
    sampled = uniform_sample(frames, n=sample_n)

    results = {}

    if quality_models is not None:
        print(f"Loading quality models: {quality_models}")
        evaluator = ImageQualityEvaluator(model_names=quality_models, device=device)
        results["quality"] = evaluator.score_all(sampled, prompt)
        for name, score in results["quality"].items():
            print(f"  {name}: {score:.4f}")

    if reference_path:
        ref = Image.open(reference_path).convert("RGB")
        print(f"Loading reference metrics (SSIM/LPIPS/PSNR)...")
        metrics = ReferenceMetrics(device=device)
        results["reference"] = metrics.all(frames, ref)
        for name, score in results["reference"].items():
            print(f"  {name}: {score:.4f}")

    # ── 轨迹预览视频 ──
    if trajectory_preview:
        print(f"Generating trajectory preview video...")
        try:
            from .trajectory_preview import (
                render_signals_preview,
                load_cotracker,
                compute_tracks,
                render_trajectory_preview,
                save_trajectory_preview_video,
            )

            use_saved_pt = (bbox_mask_path and os.path.exists(bbox_mask_path)) or \
                           (track_video_path and os.path.exists(track_video_path))

            if use_saved_pt:
                print(f"  Using saved signals (bbox_mask.pt / track_video.pt)...")
                render_signals_preview(
                    frames,
                    bbox_mask_path=bbox_mask_path if bbox_mask_path and os.path.exists(bbox_mask_path) else None,
                    track_video_path=track_video_path if track_video_path and os.path.exists(track_video_path) else None,
                    output_path=trajectory_preview,
                    fps=fps,
                    show_bbox=True,
                    show_heatmap=True,
                )
                results["trajectory_preview"] = str(trajectory_preview)
                results["trajectory_preview_mode"] = "saved_signals"
            else:
                cotracker = load_cotracker(device=device)
                if cotracker is not None:
                    tracks, visibility = compute_tracks(cotracker, frames, device=device)
                    preview_frames = render_trajectory_preview(
                        frames,
                        tracks=tracks,
                        visibility=visibility,
                        trail_length=trail_length,
                        subsample_tracks=subsample_tracks,
                    )
                    save_trajectory_preview_video(
                        preview_frames, trajectory_preview, fps=fps
                    )
                    results["trajectory_preview"] = str(trajectory_preview)
                    results["trajectory_preview_mode"] = "cotracker"
                    del cotracker
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    print("  [SKIP] CoTracker 不可用，跳过轨迹预览")
        except Exception as e:
            print(f"  [ERROR] 轨迹预览生成失败: {e}")
            import traceback
            traceback.print_exc()

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate video generation quality")
    parser.add_argument("--video", required=True, help="Path to generated video")
    parser.add_argument("--prompt", required=True, help="Text prompt")
    parser.add_argument("--reference", help="Path to reference image (for SSIM/LPIPS/PSNR)")
    parser.add_argument("--sample", type=int, default=8, help="Number of frames to sample for quality metrics")
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("--models", nargs="+", choices=["ImageReward", "Aesthetic", "PickScore", "CLIP", "HPSv2", "HPSv2.1", "MPS"],
                        help="Quality models to use (default: all)")
    parser.add_argument("--output", help="Path to save JSON results")
    parser.add_argument("--trajectory_preview", help="Path to save trajectory preview video (e.g. trajectory_preview.mp4)")
    parser.add_argument("--fps", type=int, default=15, help="FPS for trajectory preview video (default: 15)")
    parser.add_argument("--trail_length", type=int, default=8, help="Trajectory trail length in frames (default: 8)")
    parser.add_argument("--subsample_tracks", type=int, default=3, help="Subsample tracks by this factor (default: 3)")
    parser.add_argument("--bbox_mask", help="Path to bbox_mask.pt (saved signal, overrides CoTracker)")
    parser.add_argument("--track_video", help="Path to track_video.pt (saved signal, overrides CoTracker)")
    args = parser.parse_args()

    results = evaluate_video(
        video_path=args.video,
        prompt=args.prompt,
        reference_path=args.reference,
        sample_n=args.sample,
        device=args.device,
        quality_models=args.models,
        trajectory_preview=args.trajectory_preview,
        fps=args.fps,
        trail_length=args.trail_length,
        subsample_tracks=args.subsample_tracks,
        bbox_mask_path=args.bbox_mask,
        track_video_path=args.track_video,
    )

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
