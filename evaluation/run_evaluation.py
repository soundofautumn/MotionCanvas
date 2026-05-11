import argparse
import json
import sys
from pathlib import Path
from PIL import Image
from typing import List

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
    args = parser.parse_args()

    results = evaluate_video(
        video_path=args.video,
        prompt=args.prompt,
        reference_path=args.reference,
        sample_n=args.sample,
        device=args.device,
        quality_models=args.models,
    )

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
