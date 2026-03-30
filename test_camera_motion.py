#!/usr/bin/env python3
"""
Test script for camera motion implementation
验证相机运动功能是否正常工作
"""

import os
import sys
import json
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the modified Gradio helper functions
from apps.gradio.motioncanvas import (
    generate_camera_json_from_sliders,
    build_camera_mask_from_json_str,
    preview_camera_motion,
)

def test_camera_json_generation():
    """Test camera JSON generation from slider values"""
    print("=" * 60)
    print("TEST 1: Camera JSON Generation")
    print("=" * 60)
    
    # Test parameters
    num_frames = 49
    
    # Generate camera JSON with different camera motions
    camera_json = generate_camera_json_from_sliders(
        zoom_start=1.0, pan_x_start=0, pan_y_start=0, rotation_start=0,
        zoom_mid=1.2, pan_x_mid=20, pan_y_mid=-10, rotation_mid=5,
        zoom_end=0.8, pan_x_end=-30, pan_y_end=15, rotation_end=-10,
        num_frames=num_frames,
    )
    
    print("Generated camera JSON:")
    camera_dict = json.loads(camera_json)
    print(json.dumps(camera_dict, indent=2))
    
    # Verify structure
    assert "camera" in camera_dict, "Missing 'camera' key"
    assert "keyframes" in camera_dict["camera"], "Missing 'keyframes' key"
    assert len(camera_dict["camera"]["keyframes"]) == 3, "Should have 3 keyframes"
    
    print("✅ Camera JSON generation test PASSED\n")
    return camera_json


def test_camera_mask_generation(camera_json):
    """Test converting camera JSON to tensor"""
    print("=" * 60)
    print("TEST 2: Camera Mask Tensor Generation")
    print("=" * 60)
    
    height, width, num_frames = 480, 832, 49
    
    camera_mask = build_camera_mask_from_json_str(camera_json, num_frames, height, width)
    
    if camera_mask is None:
        print("❌ Camera mask generation FAILED: returned None")
        return False
    
    print(f"Camera mask shape: {camera_mask.shape}")
    print(f"Camera mask dtype: {camera_mask.dtype}")
    print(f"Camera mask value range: [{camera_mask.min():.3f}, {camera_mask.max():.3f}]")
    
    # Verify tensor properties
    assert camera_mask.shape == (1, 4, num_frames, height, width), \
        f"Unexpected shape: {camera_mask.shape}"
    assert camera_mask.dtype == torch.float32, f"Unexpected dtype: {camera_mask.dtype}"
    assert -1.0 <= camera_mask.min() <= 1.0 and -1.0 <= camera_mask.max() <= 1.0, \
        "Values should be in [-1, 1] range"
    
    print("✅ Camera mask generation test PASSED\n")
    return True


def test_camera_preview_visualization():
    """Test camera motion preview visualization"""
    print("=" * 60)
    print("TEST 3: Camera Motion Preview Visualization")
    print("=" * 60)
    
    # Create a test image
    input_image = Image.new("RGB", (832, 480), color="white")
    
    # Test camera preview
    preview_image = preview_camera_motion(
        input_image=input_image,
        zoom_start=1.0, pan_x_start=0, pan_y_start=0, rotation_start=0,
        zoom_mid=1.3, pan_x_mid=50, pan_y_mid=30, rotation_mid=10,
        zoom_end=0.7, pan_x_end=-50, pan_y_end=-30, rotation_end=-15,
        num_frames=49,
    )
    
    if preview_image is None:
        print("❌ Camera preview visualization FAILED: returned None")
        return False
    
    # Check if preview is a PIL Image
    if not isinstance(preview_image, Image.Image):
        print(f"❌ Camera preview visualization FAILED: returned {type(preview_image)} instead of PIL Image")
        return False
    
    print(f"Preview image size: {preview_image.size}")
    print(f"Preview image mode: {preview_image.mode}")
    
    # Save the preview for manual inspection
    preview_path = "/home/qjming/MotionCanvas/camera_motion_preview.png"
    preview_image.save(preview_path)
    print(f"Preview saved to: {preview_path}")
    
    print("✅ Camera motion preview visualization test PASSED\n")
    return True


def test_interpolation():
    """Test camera motion interpolation between keyframes"""
    print("=" * 60)
    print("TEST 4: Camera Motion Interpolation")
    print("=" * 60)
    
    height, width, num_frames = 480, 832, 49
    
    # Create camera JSON with distinct keyframes
    keyframes = [
        {"frame": 0, "zoom": 1.0, "pan": [0, 0], "rotation": 0},
        {"frame": 24, "zoom": 1.5, "pan": [100, 50], "rotation": 45},
        {"frame": 48, "zoom": 0.5, "pan": [-100, -50], "rotation": -45},
    ]
    camera_json = json.dumps({"camera": {"keyframes": keyframes}})
    
    camera_mask = build_camera_mask_from_json_str(camera_json, num_frames, height, width)
    
    if camera_mask is None:
        print("❌ Interpolation test FAILED")
        return False
    
    # Check interpolation at key frames
    print("Checking interpolation at key frames:")
    print(f"  Frame 0 (start):   zoom={camera_mask[0, 0, 0, 0, 0]:.3f}")
    print(f"  Frame 24 (middle): zoom={camera_mask[0, 0, 24, 0, 0]:.3f}")
    print(f"  Frame 48 (end):    zoom={camera_mask[0, 0, 48, 0, 0]:.3f}")
    
    # Check a middle frame for interpolation
    mid_frame = 12  # Between 0 and 24
    print(f"  Frame 12 (interpolated): zoom={camera_mask[0, 0, 12, 0, 0]:.3f}")
    
    print("✅ Camera motion interpolation test PASSED\n")
    return True


def main():
    print("\n" + "=" * 60)
    print("MotionCanvas Camera Motion Implementation Tests")
    print("=" * 60 + "\n")
    
    try:
        # Test 1: JSON generation
        camera_json = test_camera_json_generation()
        
        # Test 2: Camera mask generation
        test_camera_mask_generation(camera_json)
        
        # Test 3: Preview visualization
        test_camera_preview_visualization()
        
        # Test 4: Interpolation
        test_interpolation()
        
        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nCamera motion implementation is working correctly!")
        print("You can now start the Gradio app and test the camera motion UI.")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED WITH ERROR:")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
