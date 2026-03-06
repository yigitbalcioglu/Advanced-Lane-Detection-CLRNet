# CLRNet Video Lane Pipeline 

This repository is a practical lane-detection project built around CLRNet, focused on running ONNX inference on videos and producing stable, interpretable lane outputs.

## What This Project Does

- Detects lane lines frame-by-frame from road videos.
- Stabilizes lane interpretation over time.
- Computes geometry signals (curvature and heading-related cues).
- Builds optional bird's-eye-view (BEV) transforms.
- Renders an annotated output video (lanes, ego corridor, optional dashboard).

## Current Main Entry

Use this script for inference:
- `demo_onnx.py`

This is the active path in the current project structure.

## Project Architecture

### 1) CLI Layer
- `demo_onnx.py`
- Parses runtime arguments (video, model path, thresholds, visualization flags) and starts the pipeline.

### 2) Deployment Pipeline Layer
- `clrnet/deploy/onnx_pipeline.py`
- Core OOP runtime:
  - model session setup (ONNX Runtime)
  - frame preprocessing
  - inference
  - decode + lane filtering
  - geometry/BEV integration
  - overlay rendering
  - video read/write loop

### 3) Lane Geometry/Control Utilities
- `clrnet/utils/advanced_lane_pipeline.py`
- Contains helpers for:
  - polynomial lane fitting
  - curvature estimation
  - BEV-based transforms
  - extra sliding-window angle estimation
  - temporal smoothing components

### 4) Training/Evaluation Stack (separate from demo runtime)
- `main.py`
- `clrnet/models/`, `clrnet/datasets/`, `clrnet/engine/`, `configs/`
- Used when training or validating `.pth` models.

## Algorithm Flow (Per Frame)

1. Read frame from input video.
2. Crop the upper region (`cut_height`) and resize to network input size.
3. Run ONNX inference.
4. Decode lane candidates and apply lane NMS/filtering.
5. Convert lane points into image-space polylines.
6. Fit lane curves (`numpy.polyfit`) for geometric consistency.
7. Estimate curvature and heading-related values from fitted lanes.
8. Optionally apply BEV perspective transform.
9. Optionally compute sliding-window angle in BEV as an extra cue.
10. Smooth temporal signals to reduce frame-to-frame jitter.
11. Render overlays and write the output frame.

## Weights And Model Files

In this workspace, model files are at the repository root:
- `tusimple_r18.onnx`
- `culane_r18.onnx`
- `tusimple_r18.pth`
- `culane_r18.pth`
- `culane_dla34.pth`

Usage convention:
- `*.onnx` for `demo_onnx.py` inference.
- `*.pth` for training/validation via `main.py`.

## Quick Usage

Run ONNX inference on a video from `videos/`:

```bash
python demo_onnx.py 2 --videos-dir ./videos --onnx ./tusimple_r18.onnx
```

Useful options:
- `--output <path>`: save annotated video.
- `--no-show`: run without live display.
- `--allow-cpu`: allow CPU fallback.
- `--disable-bev`: disable BEV branch.
- `--no-dashboard`: disable side dashboard.
- `--cut-height <int>`: adjust top crop.

Example:

```bash
python demo_onnx.py 2 --videos-dir ./videos --onnx ./tusimple_r18.onnx --output ./output_demo.mp4 --no-show
```

## File Roles (Top Level)

- `demo_onnx.py`: active production-style ONNX video pipeline entry.
- `main.py`: training/validation/testing entry for config-based experiments.
- `torch2onnx.py`: conversion/export helper from PyTorch checkpoints.
- `demo_video.py`: optional PyTorch inference demo.
- `demo_onnx_new.py`, `demo_trt.py`: legacy single-image demo scripts.

## Notes

- Best lane quality is sensitive to `--cut-height`; tune it per camera setup.
- GPU is recommended for stable real-time throughput.
- If you only need one reliable runtime path, keep using `demo_onnx.py`.
