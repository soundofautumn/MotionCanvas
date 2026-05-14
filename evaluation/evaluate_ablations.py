#!/usr/bin/env python3
"""
批量评估 ablations/ 下所有消融实验的视频质量。

用法：
  # 使用所有质量模型（默认）
  python evaluation/evaluate_ablations.py

  # 指定部分模型
  python evaluation/evaluate_ablations.py --models ImageReward CLIP

  # 同时计算参考指标（SSIM/LPIPS/PSNR），需提供参考图目录
  python evaluation/evaluate_ablations.py --reference_dir data/input_img

  # 指定 GPU
  python evaluation/evaluate_ablations.py --device cuda:0

  # 输出到指定汇总文件
  python evaluation/evaluate_ablations.py --summary results.json

  # 启用轨迹预览视频（使用 CoTracker 追踪并渲染轨迹叠加）
  python evaluation/evaluate_ablations.py --trajectory_preview --trail_length 10 --subsample_tracks 3
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

# 确保项目根目录在路径中
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from evaluation.image_quality_metrics import ImageQualityEvaluator, uniform_sample, ALL_MODELS
from evaluation.reference_metrics import ReferenceMetrics
from evaluation.run_evaluation import load_video_frames


def load_config(exp_dir: Path) -> dict:
    """读取消融实验的 config.json，返回 {prompt, image, ...}"""
    cfg_path = exp_dir / "config.json"
    if not cfg_path.exists():
        print(f"  [SKIP] {exp_dir.name}: 缺少 config.json")
        return None
    cfg = json.loads(cfg_path.read_text())
    return cfg


def load_log(exp_dir: Path) -> dict:
    """读取 log.json 获取额外信息"""
    log_path = exp_dir / "log.json"
    if log_path.exists():
        return json.loads(log_path.read_text())
    return {}


def evaluate_ablation(
    exp_dir: Path,
    device: str = "cuda",
    quality_models: list = None,
    reference_dir: str = None,
    sample_n: int = 8,
    trajectory_preview: bool = False,
    trail_length: int = 8,
    subsample_tracks: int = 3,
    fps: int = 15,
) -> dict:
    """评估单个消融实验目录"""
    name = exp_dir.name
    video_path = exp_dir / "output.mp4"
    if not video_path.exists():
        print(f"  [SKIP] {name}: 缺少 output.mp4")
        return None

    # 读取配置
    cfg = load_config(exp_dir)
    if cfg is None:
        return None
    log = load_log(exp_dir)

    prompt = cfg.get("prompt", "") or log.get("prompt", "")
    image_rel = log.get("image", "")
    seed = cfg.get("gen_params", {}).get("seed", log.get("seed", "?"))

    if not prompt:
        print(f"  [SKIP] {name}: prompt 为空")
        return None

    print(f"\n{'='*60}")
    print(f"Evaluating: {name}")
    print(f"  video:    {video_path}")
    print(f"  prompt:   {prompt[:80]}...")
    print(f"  seed:     {seed}")
    print(f"  mode:     {log.get('mode', '?')}")
    print(f"{'='*60}")

    # 加载视频帧
    print(f"  Loading video frames...")
    frames = load_video_frames(str(video_path), max_frames=-1)
    sampled = uniform_sample(frames, n=sample_n)
    print(f"  Total frames: {len(frames)}, sampled: {len(sampled)}")

    results = {
        "experiment": name,
        "prompt": prompt,
        "seed": seed,
        "mode": log.get("mode", "?"),
        "image": image_rel,
    }

    # ── 质量模型评分 ──
    if quality_models is not None:
        print(f"  Loading quality models: {quality_models}")
        t0 = time.time()
        evaluator = ImageQualityEvaluator(model_names=quality_models, device=device)
        quality_scores = evaluator.score_all(sampled, prompt)
        results["quality"] = {k: round(v, 6) for k, v in quality_scores.items()}
        print(f"  Quality scores ({time.time()-t0:.1f}s):")
        for name, score in quality_scores.items():
            print(f"    {name}: {score:.4f}")

    # ── 参考指标（SSIM/LPIPS/PSNR） ──
    if reference_dir:
        ref_path = Path(reference_dir) / (image_rel.split("/")[-1] if image_rel else "")
        if not ref_path.exists() and image_rel:
            ref_path = project_root / image_rel
        if ref_path.exists():
            from PIL import Image
            ref = Image.open(str(ref_path)).convert("RGB")
            print(f"  Reference: {ref_path}")
            t0 = time.time()
            metrics = ReferenceMetrics(device=device)
            ref_scores = metrics.all(frames, ref)
            results["reference"] = {k: round(v, 6) for k, v in ref_scores.items()}
            print(f"  Reference metrics ({time.time()-t0:.1f}s):")
            for k, v in ref_scores.items():
                print(f"    {k}: {v:.4f}")
        else:
            print(f"  [WARN] 参考图未找到: {ref_path}，跳过参考指标")

    # ── 轨迹预览视频 ──
    if trajectory_preview:
        traj_out = exp_dir / "trajectory_preview.mp4"
        print(f"  Generating trajectory preview video...")
        t0 = time.time()
        try:
            from evaluation.trajectory_preview import (
                render_signals_preview,
                load_cotracker,
                compute_tracks,
                render_trajectory_preview,
                save_trajectory_preview_video,
            )
            import torch

            # 优先使用已保存的 track_video.pt / bbox_mask.pt
            bbox_mask_pt = exp_dir / "bbox_mask.pt"
            track_video_pt = exp_dir / "track_video.pt"
            use_saved_pt = bbox_mask_pt.exists() or track_video_pt.exists()

            if use_saved_pt:
                print(f"  Using saved signals (bbox_mask.pt / track_video.pt)...")
                render_signals_preview(
                    frames,
                    bbox_mask_path=str(bbox_mask_pt) if bbox_mask_pt.exists() else None,
                    track_video_path=str(track_video_pt) if track_video_pt.exists() else None,
                    output_path=str(traj_out),
                    fps=fps,
                    show_bbox=True,
                    show_heatmap=True,
                    bbox_line_width=3,
                    heatmap_alpha=0.35,
                )
                results["trajectory_preview"] = str(traj_out)
                results["trajectory_preview_mode"] = "saved_signals"
                print(f"  Trajectory preview done ({time.time()-t0:.1f}s)")
            else:
                # Fallback: CoTracker 轨迹追踪
                print(f"  未找到已保存的 .pt 信号文件，尝试 CoTracker...")
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
                    save_trajectory_preview_video(preview_frames, str(traj_out), fps=fps)
                    results["trajectory_preview"] = str(traj_out)
                    results["trajectory_preview_mode"] = "cotracker"
                    del cotracker
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print(f"  Trajectory preview done ({time.time()-t0:.1f}s)")
                else:
                    print(f"  [SKIP] CoTracker 不可用，跳过轨迹预览")
        except Exception as e:
            print(f"  [ERROR] 轨迹预览生成失败: {e}")
            import traceback
            traceback.print_exc()

    return results


def main():
    parser = argparse.ArgumentParser(description="批量评估 ablations 下的视频质量")
    parser.add_argument(
        "--ablations_dir",
        default=str(project_root / "ablation_results"),
        help="ablations 根目录 (默认: ./ablation_results)",
    )
    parser.add_argument(
        "--models", nargs="+",
        choices=["ImageReward", "Aesthetic", "PickScore", "CLIP", "HPSv2", "HPSv2.1", "MPS"],
        help="使用的质量模型 (默认: 全部)",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="推理设备 (默认: cuda)",
    )
    parser.add_argument(
        "--sample", type=int, default=8,
        help="采样的帧数 (默认: 8)",
    )
    parser.add_argument(
        "--reference_dir",
        help="参考图目录，提供后额外计算 SSIM/LPIPS/PSNR",
    )
    parser.add_argument(
        "--summary",
        default=str(project_root / "ablation_results" / "evaluation_results.json"),
        help="汇总结果 JSON 路径 (默认: ablation_results/evaluation_results.json)",
    )
    parser.add_argument(
        "--summary_csv",
        default=str(project_root / "ablation_results" / "evaluation_results.csv"),
        help="汇总结果 CSV 路径 (默认: ablation_results/evaluation_results.csv)",
    )
    parser.add_argument(
        "--trajectory_preview", action="store_true",
        help="启用轨迹预览视频生成（使用 CoTracker 追踪点轨迹并叠加渲染）",
    )
    parser.add_argument(
        "--trail_length", type=int, default=8,
        help="轨迹拖尾长度（帧数，默认: 8）",
    )
    parser.add_argument(
        "--subsample_tracks", type=int, default=3,
        help="轨迹点下采样因子（默认: 3，每 3 个点取 1 个）",
    )
    parser.add_argument(
        "--traj_fps", type=int, default=15,
        help="轨迹预览视频帧率（默认: 15）",
    )
    args = parser.parse_args()

    ablations_dir = Path(args.ablations_dir)
    if not ablations_dir.exists():
        print(f"错误: 目录不存在 {ablations_dir}")
        sys.exit(1)

    # 找到所有消融实验子目录（按名称排序）
    exp_dirs = sorted([
        d for d in ablations_dir.iterdir()
        if d.is_dir() and d.name[0].isdigit()
    ])
    if not exp_dirs:
        print(f"错误: 在 {ablations_dir} 中未找到消融实验子目录")
        sys.exit(1)

    # 默认使用全部质量模型
    models_to_use = args.models if args.models is not None else ALL_MODELS
    print(f"找到 {len(exp_dirs)} 个消融实验:\n  " + "\n  ".join(d.name for d in exp_dirs))
    print(f"质量模型: {models_to_use}")
    if args.reference_dir:
        print(f"参考指标: 开启 (参考图目录: {args.reference_dir})")
    else:
        print(f"参考指标: 关闭")
    if args.trajectory_preview:
        print(f"轨迹预览: 开启 (拖尾={args.trail_length}, 下采样={args.subsample_tracks}, fps={args.traj_fps})")
    else:
        print(f"轨迹预览: 关闭")
    print()

    # 逐个评估
    all_results = []
    for exp_dir in exp_dirs:
        result = evaluate_ablation(
            exp_dir,
            device=args.device,
            quality_models=models_to_use,
            reference_dir=args.reference_dir,
            sample_n=args.sample,
            trajectory_preview=args.trajectory_preview,
            trail_length=args.trail_length,
            subsample_tracks=args.subsample_tracks,
            fps=args.traj_fps,
        )
        if result is not None:
            all_results.append(result)

            # 每个实验单独保存一份结果到 ablation_results
            results_out_dir = project_root / "ablation_results"
            results_out_dir.mkdir(parents=True, exist_ok=True)
            result_path = results_out_dir / f"{result['experiment']}_evaluation.json"
            result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
            print(f"  结果已保存: {result_path}")

    # ── 汇总 JSON ──
    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "device": args.device,
        "models": args.models or "all",
        "sample_n": args.sample,
        "has_reference": args.reference_dir is not None,
        "has_trajectory_preview": args.trajectory_preview,
        "trail_length": args.trail_length,
        "subsample_tracks": args.subsample_tracks,
        "results": all_results,
    }
    summary_path = Path(args.summary)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n汇总 JSON 已保存: {summary_path}")

    # ── 汇总 CSV ──
    csv_path = Path(args.summary_csv)
    with open(csv_path, "w", newline="") as f:
        # 动态构建表头
        quality_names = sorted(set(
            k for r in all_results if "quality" in r
            for k in r["quality"]
        ))
        ref_names = sorted(set(
            k for r in all_results if "reference" in r
            for k in r["reference"]
        ))
        header = ["experiment", "mode", "seed", "prompt", "trajectory_preview"] + quality_names + ref_names
        w = csv.writer(f)
        w.writerow(header)
        for r in all_results:
            row = [
                r["experiment"],
                r.get("mode", ""),
                r.get("seed", ""),
                r.get("prompt", ""),
                r.get("trajectory_preview", ""),
            ]
            row += [r.get("quality", {}).get(n, "") for n in quality_names]
            row += [r.get("reference", {}).get(n, "") for n in ref_names]
            w.writerow(row)
    print(f"汇总 CSV 已保存: {csv_path}")

    # ── 打印汇总表格 ──
    print(f"\n{'='*80}")
    print(f"{'实验':<25} {'模式':<8} {'seed':<6}", end="")
    if all_results and "quality" in all_results[0]:
        for qn in sorted(all_results[0]["quality"].keys()):
            print(f" {qn:<14}", end="")
    print()
    print("-" * 80)
    for r in all_results:
        print(f"{r['experiment']:<25} {r.get('mode',''):<8} {str(r.get('seed','')):<6}", end="")
        if "quality" in r:
            for qn in sorted(r["quality"].keys()):
                print(f" {r['quality'][qn]:<14.4f}", end="")
        print()
    print("=" * 80)


if __name__ == "__main__":
    main()
