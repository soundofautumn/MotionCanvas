#!/usr/bin/env python3
"""
Convert 3D user intent (camera path, global object motion, local object points)
into 2D MotionCanvas control parameters without changing model code.

Outputs:
- bbox_2d.json (normalized 2D bboxes)
- points_2d.json (normalized 2D point tracks)
- bbox_mask.pt (tensor for model)
- track_video.pt (pos feature map for model)
"""

import argparse
import json
import math
import os
from typing import Dict, List, Tuple, Optional

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


def euler_yaw_pitch_roll_to_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)

    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)

    # Yaw (Y), Pitch (X), Roll (Z)
    r_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    r_x = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float32)
    r_z = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    return r_z @ r_x @ r_y


def interpolate_keyframes(keyframes: List[dict], num_frames: int, fields: List[str]) -> List[dict]:
    if not keyframes:
        return []

    keyframes_sorted = sorted(keyframes, key=lambda k: int(k.get("frame", 0)))
    frames_out = []
    for frame_idx in range(num_frames):
        prev_kf = keyframes_sorted[0]
        next_kf = keyframes_sorted[-1]
        for kf in keyframes_sorted:
            if int(kf.get("frame", 0)) <= frame_idx:
                prev_kf = kf
            if int(kf.get("frame", 0)) >= frame_idx:
                next_kf = kf
                break

        f0 = int(prev_kf.get("frame", 0))
        f1 = int(next_kf.get("frame", 0))
        if f0 == f1:
            t = 0.0
        else:
            t = (frame_idx - f0) / max(1, f1 - f0)

        out = {"frame": frame_idx}
        for field in fields:
            v0 = prev_kf.get(field)
            v1 = next_kf.get(field)
            if isinstance(v0, list) and isinstance(v1, list):
                out[field] = [lerp(v0[i], v1[i], t) for i in range(len(v0))]
            else:
                out[field] = lerp(float(v0), float(v1), t)
        frames_out.append(out)

    return frames_out


def build_intrinsics(width: int, height: int, fov_deg: float) -> Tuple[float, float, float, float]:
    fov = math.radians(fov_deg)
    fx = 0.5 * width / math.tan(0.5 * fov)
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    return fx, fy, cx, cy


def project_world_to_pixel(
    point_w: np.ndarray,
    cam_pos: np.ndarray,
    cam_rot: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> Optional[Tuple[float, float, float]]:
    # camera coordinates: R^T (p - c)
    p_cam = cam_rot.T @ (point_w - cam_pos)
    z = p_cam[2]
    if z <= 1e-6:
        return None
    x = (p_cam[0] / z) * fx + cx
    y = (p_cam[1] / z) * fy + cy
    return x, y, z


def compute_camera_poses(camera_json: dict, num_frames: int) -> List[dict]:
    keyframes = camera_json.get("camera", {}).get("keyframes", [])
    fields = ["pos", "rot"]
    interpolated = interpolate_keyframes(keyframes, num_frames, fields)

    poses = []
    for kf in interpolated:
        pos = np.array(kf.get("pos", [0.0, 0.0, 0.0]), dtype=np.float32)
        rot = kf.get("rot", [0.0, 0.0, 0.0])
        rot_m = euler_yaw_pitch_roll_to_matrix(rot[0], rot[1], rot[2])
        poses.append({"pos": pos, "rot_m": rot_m})
    return poses


def interpolate_object_frames(obj: dict, num_frames: int) -> List[dict]:
    keyframes = obj.get("keyframes", [])
    fields = ["center", "size"]
    return interpolate_keyframes(keyframes, num_frames, fields)


def project_object_bbox(
    obj_frames: List[dict],
    camera_poses: List[dict],
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> Dict[str, List[float]]:
    frames_out: Dict[str, List[float]] = {}
    for frame in obj_frames:
        f = int(frame["frame"])
        center = np.array(frame.get("center", [0.0, 0.0, 0.0]), dtype=np.float32)
        size = np.array(frame.get("size", [1.0, 1.0, 1.0]), dtype=np.float32)
        half = size * 0.5

        corners = []
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    corners.append(center + half * np.array([sx, sy, sz], dtype=np.float32))

        cam = camera_poses[f]
        pts_2d = []
        for c in corners:
            proj = project_world_to_pixel(c, cam["pos"], cam["rot_m"], fx, fy, cx, cy)
            if proj is not None:
                pts_2d.append((proj[0], proj[1]))

        if not pts_2d:
            continue

        xs = [p[0] for p in pts_2d]
        ys = [p[1] for p in pts_2d]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        x1 = max(0.0, min(width - 1.0, x1))
        y1 = max(0.0, min(height - 1.0, y1))
        x2 = max(0.0, min(width - 1.0, x2))
        y2 = max(0.0, min(height - 1.0, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        frames_out[str(f)] = [x1 / width, y1 / height, x2 / width, y2 / height]

    return frames_out


def build_bbox_json(objects_json: dict, camera_poses: List[dict], width: int, height: int, fx: float, fy: float, cx: float, cy: float) -> dict:
    objects_out = []
    for obj in objects_json.get("objects", []):
        obj_frames = interpolate_object_frames(obj, len(camera_poses))
        frames_out = project_object_bbox(obj_frames, camera_poses, width, height, fx, fy, cx, cy)
        if frames_out:
            objects_out.append({"frames": frames_out, "id": obj.get("id", "")})
    return {"objects": objects_out}


def interpolate_point_keyframes(keyframes: Dict[str, List[float]], num_frames: int) -> List[List[float]]:
    kf_items = sorted([(int(k), v) for k, v in keyframes.items()], key=lambda x: x[0])
    if not kf_items:
        return []

    tracks = []
    for frame_idx in range(num_frames):
        prev_kf = kf_items[0]
        next_kf = kf_items[-1]
        for kf in kf_items:
            if kf[0] <= frame_idx:
                prev_kf = kf
            if kf[0] >= frame_idx:
                next_kf = kf
                break

        f0 = prev_kf[0]
        f1 = next_kf[0]
        if f0 == f1:
            t = 0.0
        else:
            t = (frame_idx - f0) / max(1, f1 - f0)

        v0 = prev_kf[1]
        v1 = next_kf[1]
        tracks.append([lerp(v0[i], v1[i], t) for i in range(len(v0))])

    return tracks


def object_frame_lookup(objects_json: dict, num_frames: int) -> Dict[str, List[dict]]:
    lookup = {}
    for obj in objects_json.get("objects", []):
        obj_id = obj.get("id")
        if not obj_id:
            continue
        lookup[obj_id] = interpolate_object_frames(obj, num_frames)
    return lookup


def point_world_from_object_local(local_xyz: List[float], obj_frame: dict) -> np.ndarray:
    center = np.array(obj_frame.get("center", [0.0, 0.0, 0.0]), dtype=np.float32)
    size = np.array(obj_frame.get("size", [1.0, 1.0, 1.0]), dtype=np.float32)
    # local coords are normalized [0, 1], convert to [-0.5, 0.5]
    local = np.array(local_xyz, dtype=np.float32) - 0.5
    return center + local * size


def build_points_json(
    points_json: dict,
    objects_json: dict,
    camera_poses: List[dict],
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> dict:
    points_out = []
    obj_lookup = object_frame_lookup(objects_json, len(camera_poses))

    for point in points_json.get("points", []):
        space = point.get("space", "world")
        obj_id = point.get("object_id", "")
        keyframes = point.get("frames", {})
        tracks_3d = interpolate_point_keyframes(keyframes, len(camera_poses))
        if not tracks_3d:
            continue

        frames_out = {}
        for f, pos in enumerate(tracks_3d):
            if space == "object":
                obj_frames = obj_lookup.get(obj_id)
                if not obj_frames:
                    continue
                world = point_world_from_object_local(pos, obj_frames[f])
            else:
                world = np.array(pos, dtype=np.float32)

            cam = camera_poses[f]
            proj = project_world_to_pixel(world, cam["pos"], cam["rot_m"], fx, fy, cx, cy)
            if proj is None:
                continue
            x, y, _ = proj
            if x < 0 or x >= width or y < 0 or y >= height:
                continue
            frames_out[str(f)] = [x / width, y / height]

        if frames_out:
            points_out.append({"frames": frames_out})

    return {"points": points_out}


def build_bbox_mask_from_json(bbox_json: dict, num_frames: int, height: int, width: int) -> torch.Tensor:
    mask = torch.zeros(1, 3, num_frames, height, width)
    for obj in bbox_json.get("objects", []):
        for fi_str, bbox in obj.get("frames", {}).items():
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


def build_track_video_from_points(points_json: dict, num_frames: int, height: int, width: int) -> Optional[torch.Tensor]:
    points = points_json.get("points", [])
    if not points:
        return None

    tracks = []
    for pt in points:
        frames = pt.get("frames", {})
        if not frames:
            continue
        tracks.append(frames)

    if not tracks:
        return None

    n = len(tracks)
    pred_tracks = torch.full((1, num_frames, n, 2), -1.0, dtype=torch.float32)
    pred_visibility = torch.zeros((1, num_frames, n), dtype=torch.bool)

    for i, frames in enumerate(tracks):
        for fi_str, xy in frames.items():
            fi = int(fi_str)
            if fi >= num_frames:
                continue
            x, y = xy
            if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                x = x * width
                y = y * height
            pred_tracks[0, fi, i, 0] = float(x)
            pred_tracks[0, fi, i, 1] = float(y)
            pred_visibility[0, fi, i] = True

        # Fill missing frames by interpolation in 2D
        kf_items = sorted([(int(k), v) for k, v in frames.items()], key=lambda x: x[0])
        if len(kf_items) >= 2:
            for j in range(len(kf_items) - 1):
                f0, v0 = kf_items[j]
                f1, v1 = kf_items[j + 1]
                for f in range(f0, f1 + 1):
                    t = 0.0 if f0 == f1 else (f - f0) / max(1, f1 - f0)
                    x = lerp(v0[0], v1[0], t)
                    y = lerp(v0[1], v1[1], t)
                    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                        x = x * width
                        y = y * height
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


def main():
    parser = argparse.ArgumentParser(description="MotionCanvas intent to params")
    parser.add_argument("--input_image", type=str, required=True, help="input image path")
    parser.add_argument("--camera_json", type=str, required=True, help="3D camera json")
    parser.add_argument("--objects_json", type=str, required=True, help="3D objects json")
    parser.add_argument("--points_json", type=str, required=True, help="3D points json")
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fov", type=float, default=60.0)
    parser.add_argument("--output_dir", type=str, default="outputs/intent")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    _ = Image.open(args.input_image).convert("RGB")

    camera_json = read_json(args.camera_json)
    objects_json = read_json(args.objects_json)
    points_json = read_json(args.points_json)

    fx, fy, cx, cy = build_intrinsics(args.width, args.height, args.fov)
    cam_poses = compute_camera_poses(camera_json, args.num_frames)

    bbox_2d = build_bbox_json(objects_json, cam_poses, args.width, args.height, fx, fy, cx, cy)
    points_2d = build_points_json(points_json, objects_json, cam_poses, args.width, args.height, fx, fy, cx, cy)

    bbox_json_path = os.path.join(args.output_dir, "bbox_2d.json")
    points_json_path = os.path.join(args.output_dir, "points_2d.json")
    write_json(bbox_json_path, bbox_2d)
    write_json(points_json_path, points_2d)

    bbox_mask = build_bbox_mask_from_json(bbox_2d, args.num_frames, args.height, args.width)
    bbox_mask_path = os.path.join(args.output_dir, "bbox_mask.pt")
    torch.save(bbox_mask, bbox_mask_path)

    track_video = build_track_video_from_points(points_2d, args.num_frames, args.height, args.width)
    if track_video is not None:
        track_video_path = os.path.join(args.output_dir, "track_video.pt")
        torch.save(track_video, track_video_path)

    print("Done.")
    print("bbox_2d.json:", bbox_json_path)
    print("points_2d.json:", points_json_path)
    print("bbox_mask.pt:", bbox_mask_path)
    if track_video is not None:
        print("track_video.pt:", track_video_path)
    else:
        print("track_video.pt: not generated (no points)")


if __name__ == "__main__":
    main()
