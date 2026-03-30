# MotionCanvas Camera Motion - User Guide

## Overview

You now have **full camera motion control** in the MotionCanvas Gradio UI! This guide explains how to use the new camera motion features to create dynamic cinematic shots.

---

## Quick Start

### 1. Launch the Gradio App

```bash
cd /home/qjming/MotionCanvas
python3 apps/gradio/motioncanvas.py
```

Then open `http://localhost:6006` in your browser.

### 2. Navigate to the Camera Motion Tab

1. Scroll down to **"Motion Control"** section
2. Click on the **"相机运动" (Camera Motion)** tab
3. You'll see sliders for camera parameters at three keyframes

### 3. Set Camera Motion

For each keyframe (Start/Mid/End), adjust:
- **缩放 (Zoom)**: How close/far (0.5 = 2x zoomed out, 2.0 = 2x zoomed in)
- **平移 X (Pan X)**: Move camera left (-100) or right (+100)
- **平移 Y (Pan Y)**: Move camera up (+100) or down (-100)
- **旋转 (°)**: Rotate view (-45° to +45°)

### 4. Preview and Generate

- Click **"预览相机轨迹"** to see the camera path visualized on your image
- Click **"生成相机 JSON"** to create the motion data
- Click **"生成视频"** to generate the video with camera motion!

---

## Understanding Camera Motion Parameters

### Zoom (缩放)

| Value | Effect | Use Case |
|-------|--------|----------|
| 0.5 | Zoomed OUT 2x | Reveal wide scene |
| 1.0 | Original (no zoom) | Neutral, normal framing |
| 2.0 | Zoomed IN 2x | Focus on subject |

**Example: Dolly In**
- Start: Zoom = 0.5
- Mid: Zoom = 1.0
- End: Zoom = 1.5

### Pan (平移)

Moves the camera left/right (X) or up/down (Y).

| Pan X | Effect | Pan Y | Effect |
|-------|--------|-------|--------|
| -100 | Far left | -100 | Far up |
| 0 | Center (no pan) | 0 | Center |
| +100 | Far right | +100 | Far down |

**Example: Pan Left**
- Start: Pan X = 0
- Mid: Pan X = -50
- End: Pan X = -100

### Rotation (旋转)

Tilts or rolls the view.

| Rotation | Effect |
|----------|--------|
| -45° | Rotate counter-clockwise |
| 0° | Level (no rotation) |
| +45° | Rotate clockwise |

---

## Preset Camera Movements

### 1. Dolly In
*Camera moves closer to subject*

```
Start:  Zoom=0.7, Pan=[0,0], Rotation=0
Mid:    Zoom=1.0, Pan=[0,0], Rotation=0
End:    Zoom=1.5, Pan=[0,0], Rotation=0
```

### 2. Dolly Out
*Camera pulls away from subject*

```
Start:  Zoom=1.5, Pan=[0,0], Rotation=0
Mid:    Zoom=1.0, Pan=[0,0], Rotation=0
End:    Zoom=0.5, Pan=[0,0], Rotation=0
```

### 3. Pan Left
*Sweep camera from right to left*

```
Start:  Zoom=1.0, Pan=[+100,0], Rotation=0
Mid:    Zoom=1.0, Pan=[0,0], Rotation=0
End:    Zoom=1.0, Pan=[-100,0], Rotation=0
```

### 4. Pan Right
*Sweep camera from left to right*

```
Start:  Zoom=1.0, Pan=[-100,0], Rotation=0
Mid:    Zoom=1.0, Pan=[0,0], Rotation=0
End:    Zoom=1.0, Pan=[+100,0], Rotation=0
```

### 5. Orbit Around Subject
*Circle around the subject*

```
Start:  Zoom=1.0, Pan=[+50,+50], Rotation=0
Mid:    Zoom=1.0, Pan=[+0,-70], Rotation=0
End:    Zoom=1.0, Pan=[-50,+50], Rotation=0
```

### 6. Zoom and Pan (Dynamic Focus)
*Get closer while moving to subject*

```
Start:  Zoom=0.8, Pan=[50,50], Rotation=0
Mid:    Zoom=1.2, Pan=[20,20], Rotation=0
End:    Zoom=1.8, Pan=[0,0], Rotation=0
```

---

## Using with Object Motion

You can combine **camera motion** + **object motion** for powerful effects!

### Example: Follow-Cam + Object Motion

**Camera Motion:**
```
Zoom: 1.0 → 1.2 → 1.0 (slight zoom)
Pan:  [0,0] → [30,0] → [0,0] (follow subject)
```

**Object Motion** (in "可视化选区" tab):
*Draw bbox at start, mid, end to track object movement*

Result: Camera follows the moving object while adjusting zoom dynamically!

---

## Advanced: Manual JSON Editing

If you want precise control, edit the JSON directly:

### JSON Structure

```json
{
  "camera": {
    "keyframes": [
      {
        "frame": 0,
        "zoom": 1.0,
        "pan": [0, 0],
        "rotation": 0
      },
      {
        "frame": 24,
        "zoom": 1.2,
        "pan": [20, -10],
        "rotation": 5
      },
      {
        "frame": 48,
        "zoom": 0.9,
        "pan": [-30, 15],
        "rotation": -5
      }
    ]
  }
}
```

### Tips for Manual JSON Editing

1. **Frame numbers** should be: 0, mid_frame, num_frames-1
2. **Zoom** should be in range [0.5, 2.0]
3. **Pan** values should be in range [-100, 100]
4. **Rotation** should be in range [-45, 45]
5. Values are **interpolated linearly** between keyframes

---

## Preview Visualization

When you click **"预览相机轨迹"**, you'll see:

- **Blue/Yellow/Red rectangles**: Camera viewport at each keyframe
  - Rectangle size shows zoom level (smaller = zoomed in)
  - Rectangle position shows pan offset
- **White line**: Motion trajectory connecting the three keyframes
- **Color intensity**: Keyframe importance (brighter = more distinct motion)

---

## Troubleshooting

### Camera motion not appearing in generated video

**Possible causes:**
1. Model doesn't have camera_zeroconv layer yet
   - **Solution**: Will work transparently without it (graceful degradation)
2. Camera values at defaults
   - **Solution**: Adjust zoom/pan/rotation away from neutral values

### JSON parsing error

**Possible causes:**
1. Syntax error in manual JSON
   - **Solution**: Check quotes, brackets, commas
2. Value out of range
   - **Solution**: Ensure zoom ∈ [0.5, 2.0], pan ∈ [-100, 100], rotation ∈ [-45, 45]

### Preview looks wrong

1. Zoom too extreme (< 0.5 or > 2.0)
   - **Solution**: Use moderate values for best results
2. Pan way off screen
   - **Solution**: Start with smaller pan values like ±50

---

## Tips for Best Results

### ✅ DO:

- **Start subtle**: Begin with small pan/rotation values (±20-30)
- **Use gradual changes**: Smooth transitions between keyframes look better
- **Combine with prompts**: Match camera motion to content
  - "moving closer to a person" + dolly-in camera
  - "panning across a landscape" + pan-right camera
- **Test with fewer frames**: Start with 25-30 frames before full 49-frame renders
- **Preview first**: Always preview trajectory before full generation

### ❌ DON'T:

- **Extreme zoom**: Avoid zoom < 0.6 or > 1.8 (can cause artifacts)
- **Conflicting motions**: Don't have camera and object moving in opposite directions (usually)
- **Rapid changes**: Avoid sudden jumps between keyframes (interpolation works better with gradual change)
- **Multiple parameters at max**: Having zoom=2.0, pan=[100,100], rotation=45 simultaneously is chaotic

---

## Saving Your Work

### Save Camera Motion

1. Click **"生成相机 JSON"** to create the motion data
2. Copy the JSON from the text box
3. Save to a `.json` file for later reuse:
   ```
   camera_dolly_in.json
   camera_pan_left.json
   etc.
   ```

### Save Preview Image

The preview image automatically saves to:
```
/tmp/motioncanvas_preview.png
```

Keep it for reference or documentation!

---

## FAQ

**Q: Can I keyframe more than 3 frames?**
A: Not yet through the UI. But you can edit the JSON manually to add more keyframes! (Advanced)

**Q: Does camera motion affect the generated video quality?**
A: No! It's a conditioning signal to the model. Video quality depends on prompts and inference steps.

**Q: Can I do circular motion (360° rotation)?**
A: Not with linear interpolation (-45° to +45°). Use Pan instead for circular motion around subject.

**Q: What happens if I set both Camera Motion AND Object Motion?**
A: Both are applied! Camera and object motions are independent and stack together.

---

## Examples

### Example 1: Cinematic Reveal

**Prompt**: "A beautiful woman standing in a garden"

**Camera Motion:**
- Start: Zoom=0.6, Pan=[-50, 50] (far away, top-left)
- Mid: Zoom=1.0, Pan=[0, 0] (getting closer, centering)
- End: Zoom=1.3, Pan=[20, -10] (close-up, slightly right)

→ Creates a smooth "reveal" effect, pulling the viewer in

---

### Example 2: Dramatic Pan

**Prompt**: "Epic landscape with mountains, 4K cinematic"

**Camera Motion:**
- Start: Zoom=1.0, Pan=[100, 0] (at right edge)
- Mid: Zoom=1.0, Pan=[0, 0] (move to center)
- End: Zoom=1.0, Pan=[-100, 0] (pan to left edge)

→ Creates a left-to-right sweep showing the entire landscape

---

### Example 3: Follow-Focus

**Prompt**: "Person walking through city street"

**Camera Motion:**
- Start: Zoom=1.0, Pan=[0, 0]
- Mid: Zoom=1.1, Pan=[30, 0]
- End: Zoom=1.2, Pan=[60, 0]

(Also draw object motion boxes matching the path)

→ Combines dolly-in with forward pan, mimicking a tracking shot

---

## Support & Feedback

For issues or feature requests:
1. Check the test script: `python3 test_camera_motion.py`
2. Review implementation notes in `/memories/session/plan.md`
3. Check console output for detailed error messages

---

Enjoy creating cinematic shots with MotionCanvas! 🎬✨
