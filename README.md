# AI Intelligent Safety Mirror

A computer-vision POC for visualising industrial PPE on a live camera feed.

## Development roadmap

1. **Stage 1 – Pose detection:** OpenCV webcam + MediaPipe Pose Landmarker.
2. **Stage 2 – 2D PPE overlays:** transparent PNG assets anchored to body landmarks.
3. **Stage 3 – Tracking improvements:** smoothing, orientation and tracking-loss recovery.
4. **Stage 4 – 3D PPE:** GLB/glTF assets rendered with Three.js.
5. **Stage 5 – Kiosk UI:** PPE selection and approved safety guidance.

The initial POC is for visual awareness and demonstration only. It is not a PPE fit-validation or safety certification system.

## Requirements

- Python 3.10–3.12 recommended (use Python 3.11 if you encounter package compatibility issues)
- Webcam
- Windows, macOS or Linux

## Setup (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Download the **Pose Landmarker Lite** task model from the official MediaPipe documentation:
https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker

Save it as:

`models/pose_landmarker_lite.task`

## Run

From the repository root:

```powershell
python src/pose_detection.py
```

The app automatically creates a `recordings/` folder and saves each session as an MP4 file in that folder. You can override the folder with:

```powershell
python src/pose_detection.py --record-dir recordings
```

Press **Q** or **Ctrl+Q** to quit.

## Project structure

```text
AI-Intelligent-Safety-Mirror/
├── assets/
│   └── ppe/                 # Transparent PNG PPE assets (Stage 2)
├── models/
│   └── pose_landmarker_lite.task  # Download separately
├── src/
│   └── pose_detection.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Stage 1 acceptance checklist

- [ ] Webcam opens.
- [ ] A single person is detected.
- [ ] Body landmarks and skeleton are visible.
- [ ] Skeleton follows movement.
- [ ] Tracking recovers after temporary loss.
- [ ] FPS is displayed.

## Notes

- The model file is not committed to Git because it is a downloaded binary asset.
- Keep the camera view well lit and ensure the user's full body is visible.
- MediaPipe landmark coordinates are estimates. They are not precise anatomical measurements.
