#!/usr/bin/env python3
"""
Convert UI motion intent into model parameters.

Pipeline summary:
1) UI inputs: input image + camera motion + global bbox + local point tracks.
2) Convert to 2D control signals:
   - camera motion -> background point tracks (2D)
   - global motion -> interpolated bbox sequence (2D)
   - local motion -> point tracks (2D), combined with camera motion
3) Encode for model:
   - point tracks -> DCT coefficients (for logging/inspection)
   - bbox sequence -> rasterized mask (bbox_mask.pt)
   - point tracks -> pos feature map (track_video.pt)

Note: This script does NOT modify model code or perform training.
"""

import argparse
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from diffsynth.pipelines.tracker_utils import create_pos_feature_map


DEFAULT_DOWNSAMPLE_RATIOS = [4, 8, 8]
DEFAULT_POS_EMB_DIM = 16


def read_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def write_json(path: str, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def lerp(a: float, b: float, t: float) -> float:
    return a * (1.0 - t) + b * t


def build_camera_params_from_json(json_str: str, num_frames: int) -> Optional[List[dict]]:
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


def apply_camera_transform_to_point(x: float, y: float, width: int, height: int, zoom: float, pan_x: float, pan_y: float, rotation: float) -> Tuple[float, float]:
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

    out_x = rx + cx + pan_x
    out_y = ry + cy + pan_y
    return out_x, out_y


def build_bbox_mask_from_json(bbox_json: dict, num_frames: int, height: int, width: int) -> torch.Tensor:
    mask = torch.zeros(1, 3, num_frames, height, width)
    for obj in bbox_json.get("objects", []):
        frames = obj.get("frames", {})
        for fi_str, bbox in frames.items():
            fi = int(fi_str)
            if fi >= num_frames:
                continue
            x1, y1, x2, y2 = bbox
            if all(0.0 <= v <= 1.0 for v in [x1, y1, x2, y2]):
                x1, x2 = int(x1 * width), int(x2 * width)
                y1, y2 = int(y1 * height), int(y2 * height)
            else:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            x1, x2 = max(0, x1), min(width, x2)
            y1, y2 = max(0, y1), min(height, y2)
            if x2 > x1 and y2 > y1:
                mask[:, :, fi, y1:y2, x1:x2] = 1.0
    return mask * 2.0 - 1.0


def interpolate_bbox_frames(bbox_json: dict, num_frames: int, height: int, width: int) -> dict:
    objects_out = []
    for obj in bbox_json.get("objects", []):
        frames = obj.get("frames", {})
        kf_items = sorted([(int(k), v) for k, v in frames.items()], key=lambda x: x[0])
        if not kf_items:
            continue
        full_frames = {}
        for f in range(num_frames):
            prev_kf = kf_items[0]
            next_kf = kf_items[-1]
            for kf in kf_items:
                if kf[0] <= f:
                    prev_kf = kf
                if kf[0] >= f:
                    next_kf = kf
                    break
            f0, v0 = prev_kf
            f1, v1 = next_kf
            t = 0.0 if f0 == f1 else (f - f0) / max(1, f1 - f0)
            x1 = lerp(v0[0], v1[0], t)
            y1 = lerp(v0[1], v1[1], t)
            x2 = lerp(v0[2], v1[2], t)
            y2 = lerp(v0[3], v1[3], t)
            full_frames[str(f)] = [x1, y1, x2, y2]
        objects_out.append({"frames": full_frames})
    return {"objects": objects_out}


def build_point_tracks_from_json(json_str: str, num_frames: int, height: int, width: int) -> Optional[List[List[Tuple[float, float]]]]:
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
        kf_items = sorted([(int(k), v) for k, v in frames.items()], key=lambda x: x[0])
        if not kf_items:
            continue

        per_frame = []
        for f in range(num_frames):
            prev_kf = kf_items[0]
            next_kf = kf_items[-1]
            for kf in kf_items:
                if kf[0] <= f:
                    prev_kf = kf
                if kf[0] >= f:
                    next_kf = kf
                    break
            f0, v0 = prev_kf
            f1, v1 = next_kf
            t = 0.0 if f0 == f1 else (f - f0) / max(1, f1 - f0)
            x = lerp(v0[0], v1[0], t)
            y = lerp(v0[1], v1[1], t)
            if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                x = x * width
                y = y * height
            per_frame.append((float(x), float(y)))
        tracks.append(per_frame)

    if not tracks:
        return None
    return tracks


def generate_background_tracks(
    camera_params: List[dict],
    num_frames: int,
    height: int,
    width: int,
    bbox_mask: Optional[torch.Tensor],
    grid_size: int,
) -> List[List[Tuple[float, float]]]:
    xs = np.linspace(0, width - 1, grid_size)
    ys = np.linspace(0, height - 1, grid_size)
    points = [(float(x), float(y)) for y in ys for x in xs]

    if bbox_mask is not None:
        mask = (bbox_mask[0, :, 0] > 0).any(dim=0)
        mask_np = mask.cpu().numpy()
        points = [p for p in points if not mask_np[int(round(p[1])), int(round(p[0]))]]

    tracks = []
    for x, y in points:
        track = []
        for f in range(num_frames):
            params = camera_params[f]
            tx, ty = apply_camera_transform_to_point(
                x, y, width, height, params["zoom"], params["pan_x"], params["pan_y"], params["rotation"]
            )
            track.append((tx, ty))
        tracks.append(track)
    return tracks


def build_track_video_from_tracks(tracks: List[List[Tuple[float, float]]], num_frames: int, height: int, width: int) -> torch.Tensor:
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


def dct_1d(x: np.ndarray, k: int) -> np.ndarray:
    n = x.shape[0]
    coeffs = np.zeros(k, dtype=np.float32)
    for i in range(k):
        s = 0.0
        for t in range(n):
            s += x[t] * math.cos(math.pi * i * (2 * t + 1) / (2 * n))
        coeffs[i] = s * (2.0 / n) ** 0.5
    return coeffs


def encode_tracks_dct(tracks: List[List[Tuple[float, float]]], k: int, width: int, height: int) -> List[dict]:
    encoded = []
    for track in tracks:
        xs = np.array([p[0] / max(1.0, width) for p in track], dtype=np.float32)
        ys = np.array([p[1] / max(1.0, height) for p in track], dtype=np.float32)
        coeff_x = dct_1d(xs, k).tolist()
        coeff_y = dct_1d(ys, k).tolist()
        encoded.append({"dct_x": coeff_x, "dct_y": coeff_y})
    return encoded


def main() -> None:
    parser = argparse.ArgumentParser(description="MotionCanvas UI intent to params")
    parser.add_argument("--input_image", type=str, required=True, help="input image path")
    parser.add_argument("--camera_json", type=str, required=True, help="camera JSON from UI")
    parser.add_argument("--bbox_json", type=str, required=True, help="bbox JSON from UI")
    parser.add_argument("--points_json", type=str, required=True, help="point JSON from UI")
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--grid_size", type=int, default=14)
    parser.add_argument("--dct_coeffs", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="outputs/intent")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    _ = Image.open(args.input_image).convert("RGB")
    camera_json_text = read_json(args.camera_json)
    bbox_json = read_json(args.bbox_json)
    points_json_text = read_json(args.points_json)

    camera_params = build_camera_params_from_json(json.dumps(camera_json_text), args.num_frames)
    if camera_params is None:
        camera_params = [
            {"zoom": 1.0, "pan_x": 0.0, "pan_y": 0.0, "rotation": 0.0}
            for _ in range(args.num_frames)
        ]

    bbox_2d = interpolate_bbox_frames(bbox_json, args.num_frames, args.height, args.width)
    bbox_mask = build_bbox_mask_from_json(bbox_2d, args.num_frames, args.height, args.width)

    local_tracks = build_point_tracks_from_json(json.dumps(points_json_text), args.num_frames, args.height, args.width)
    if local_tracks is None:
        local_tracks = []

    # Apply camera motion to local tracks
    camera_applied_tracks = []
    for track in local_tracks:
        cam_track = []
        for f, (x, y) in enumerate(track):
            params = camera_params[f]
            tx, ty = apply_camera_transform_to_point(
                x, y, args.width, args.height,
                params["zoom"], params["pan_x"], params["pan_y"], params["rotation"]
            )
            cam_track.append((tx, ty))
        camera_applied_tracks.append(cam_track)

    background_tracks = generate_background_tracks(
        camera_params,
        args.num_frames,
        args.height,
        args.width,
        bbox_mask,
        args.grid_size,
    )

    all_tracks = background_tracks + camera_applied_tracks
    track_video = build_track_video_from_tracks(all_tracks, args.num_frames, args.height, args.width)
    dct_tracks = encode_tracks_dct(all_tracks, args.dct_coeffs, args.width, args.height)

    bbox_json_path = os.path.join(args.output_dir, "bbox_2d.json")
    points_json_path = os.path.join(args.output_dir, "points_2d.json")
    dct_json_path = os.path.join(args.output_dir, "points_dct.json")
    bbox_mask_path = os.path.join(args.output_dir, "bbox_mask.pt")
    track_video_path = os.path.join(args.output_dir, "track_video.pt")

    write_json(bbox_json_path, bbox_2d)
    points_out = []
    for track in all_tracks:
        frames = {}
        for f, (x, y) in enumerate(track):
            frames[str(f)] = [x / max(1.0, args.width), y / max(1.0, args.height)]
        points_out.append({"frames": frames})
    write_json(points_json_path, {"points": points_out})
    write_json(dct_json_path, {"tracks": dct_tracks})
    torch.save(bbox_mask, bbox_mask_path)
    torch.save(track_video, track_video_path)

    print("Done.")
    print("bbox_2d.json:", bbox_json_path)
    print("points_2d.json:", points_json_path)
    print("points_dct.json:", dct_json_path)
    print("bbox_mask.pt:", bbox_mask_path)
    print("track_video.pt:", track_video_path)


if __name__ == "__main__":
    main()
