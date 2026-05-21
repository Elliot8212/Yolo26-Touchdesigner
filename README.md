# TouchDesigner + YOLO26 (Script TOP)

Python Script TOP for TouchDesigner that runs YOLO26 models (detection, segmentation, pose) with prompt filtering, native YOLO overlay or mask output, and optional tracking (ByteTrack / BoT-SORT) for stable IDs and colors.
![Detection](Assets/detect.gif) ![Segmentation](Assets/seg.gif) ![Pose](Assets/pose.gif)

## Features
- Preset model menu (det / seg / pose variants of YOLO26).
- Prompt-based class filtering (names or numeric IDs).
- Output modes:
  - **Overlay (YOLO draw)** — native YOLO annotated overlay.
  - **Binary Mask** — mask-only output (for compositing).
  - **Data only (passthrough)** — source frame unchanged; overlays drawn in TD via DAT data (fastest, recommended for max FPS).
- SegmentationID selector:
  - `-2` → all masks, each colored by track ID (stable across frames).
  - `-1` → combined binary mask.
  - `>=0` → only the mask whose track ID matches the value.
- Optional tracking: ByteTrack or BoT-SORT (keeps IDs/colors stable). Falls back to no tracking if tracker deps are missing.
- Frame skip option for performance scaling.
- Per-frame data export to DATs: bounding boxes (`detections` DAT) and pose keypoints (`pose_points` DAT, when using pose models).
- Device status display (read-only) showing the active GPU/CPU.
- Auto-detected inference size from input buffer (no manual `imgsz` to tune).
- Model cache and Reload pulse to avoid repeated loads.

## Requirements
- TouchDesigner 2023+ with embedded **Python 3.11**.
- Windows 10/11 (the installer is PowerShell-based).
- NVIDIA GPU recommended (CUDA 11.8 → 12.8 supported, auto-detected). CPU fallback supported but slow.
- Python packages (see `requirements.txt`):
  - `ultralytics`
  - `opencv-python`
  - `numpy`
  - `lapx` (needed for ByteTrack / BoT-SORT tracking)

## Installation

### Windows (recommended)

Double-click **`install_td.bat`**, or right-click **`install_td.ps1`** → Run with PowerShell.

The installer:
1. Locates TouchDesigner's embedded Python automatically (versioned or non-versioned install paths).
2. Detects your NVIDIA GPU's compute capability via `nvidia-smi` and picks the matching PyTorch wheel:
   - RTX 50XX (Blackwell, sm_12.x) → `cu128`
   - RTX 30XX / 40XX (sm_8.x / 9.x) → `cu124`
   - RTX 20XX / 16XX (sm_7.x) → `cu121`
   - GTX 10XX and older → `cu118`
   - No NVIDIA GPU → CPU-only PyTorch
3. Installs `torch` + `torchvision` + `torchaudio` from the matching index.
4. Installs `requirements.txt` (`ultralytics`, `opencv-python`, `numpy`, `lapx`).
5. Verifies the install and prints versions.

No admin rights required — packages install to the user site-packages (`%APPDATA%\Python\Python311\site-packages`), accessible from TouchDesigner's Python automatically.

### Manual install

If you prefer to handle Python yourself, install requirements with TouchDesigner's Python directly:

```powershell
& "C:\Program Files\Derivative\TouchDesigner\bin\python.exe" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
& "C:\Program Files\Derivative\TouchDesigner\bin\python.exe" -m pip install -r requirements.txt --prefer-binary
```

Adjust the `cu124` index URL to match your GPU (see table above).

## Setup in TouchDesigner

### Basic setup

1. Place `td_yolo26.py` as a **Text DAT** in your project.
2. Add a **Script TOP** and set its `DAT` parameter to that Text DAT.
3. Click **Setup Parameters** on the Script TOP to create the custom parameters.
4. Connect a video/image TOP to the Script TOP input.
5. (Optional) Create two Table DATs named `detections` and `pose_points` to receive per-frame tabular outputs.

### Recommended setup for max FPS

To match the performance of similar projects (60+ FPS achievable), add two TOPs **before** the Script TOP — this offloads the flip + resize work to TD's GPU instead of doing it on CPU in Python:

```
videodevin1 → flip1 (Flip Y) → resolution1 (640×640 Best fit) → script1
```

- **Flip TOP** with **Flip Y = On** — handles the vertical orientation flip on GPU (~0.3 ms vs ~1–2 ms in Python).
- **Resolution TOP** with **Resolution = 640×640** and **Common → Fit = Best** — letterboxes the input to 640×640 on GPU, dramatically reducing the cost of `numpyArray()`/`copyNumpyArray()` transfers.

If you need a 1280×720 display downstream, add another Resolution TOP after the Script TOP to upscale back.

The script auto-detects the inference resolution from the input buffer size (rounded up to the nearest multiple of 32, as required by YOLO), so there's nothing to configure in Python — just choose your buffer size via the Resolution TOP.

## Parameters (Script TOP)
- **Model** — choose a YOLO26 weight (det/seg/pose variants).
- **CustomPath** — path to a custom `.pt` model (only used when `Model` = `custom`).
- **PromptClasses** — comma/semicolon list of class names or IDs to keep.
- **Confidence** — detection confidence threshold (0–1).
- **Output** — `Overlay (YOLO draw)`, `Binary Mask`, or `Data only (passthrough)`.
- **Tracker** — `None`, `ByteTrack`, or `BoT-SORT` (needs `lapx`).
- **SegmentationID** — mask selection (`-2` all colored, `-1` combined, specific ID).
- **Skip Frames** — number of frames to skip between inference runs (`0` = every frame, `1` = every other, etc.). Reuses the last result on skipped frames.
- **ReloadModel** — pulse to clear the model cache (use after changing weights on disk).
- **Device** (read-only) — shows the active inference device, e.g. `cuda:0 (NVIDIA GeForce RTX 4070 Laptop GPU)` or `cpu`.

## Outputs
- **Script TOP output:** RGBA uint8 at the input resolution.
  - Overlay mode: YOLO native annotations.
  - Mask mode: according to `SegmentationID` rules above.
  - Data only mode: source frame passthrough (draw overlays in TD using the DAT data).
- **DATs (if present):**
  - `detections`: columns `id, Object Type, Confidence, X_Center, Y_Center, Width, Height` (IDs are tracker-stable when tracking is on).
  - `pose_points`: columns `det_id, kp_id, x, y` (for pose models).

## Notes
- Use `*-seg.pt` models for masks; `*-pose.pt` for keypoints.
- If tracking reports a missing `lap` module, either install `lapx` or set `Tracker` to `None`.
- Masks and IDs remain stable across frames when tracking is enabled.
- The script reads the `delayed` argument default of TouchDesigner for `numpyArray()` — no extra latency knobs to tune.

## Performance tips
- The biggest single FPS win is **adding a Flip TOP + Resolution TOP upstream** of the Script TOP (see "Recommended setup for max FPS" above). Without it, `numpyArray()` transfer cost at full-res input is unavoidable from Python.
- `Output = Data only (passthrough)` skips `result.plot()` (the most expensive non-inference step). Combined with TD-side overlay drawing, this is the fastest mode.
- Lower the **Resolution TOP** size (e.g., 480×480) for faster inference; raise it (e.g., 1280×1280) for more precision. The script's `imgsz` follows automatically.
- Set **Skip Frames = 1** to nearly double the apparent FPS while keeping detection responsiveness reasonable.

## Troubleshooting
- **No output:** ensure an input TOP is connected and the selected model matches the task (e.g., seg model for masks).
- **Missing Python modules:** rerun `install_td.bat`.
- **Device shows `cpu` but you have an NVIDIA GPU:** check that the installer ran successfully and that `nvidia-smi` is available in your PATH. Re-run the installer; CUDA selection is automatic.
- **Tracking slowdown or errors:** switch `Tracker` to `None` if `lapx` is unavailable or performance is critical.
- **Detection box positions look offset:** make sure the Flip TOP is set to **Flip Y = On** only (not X). The Script TOP expects standard image orientation after the flip.
