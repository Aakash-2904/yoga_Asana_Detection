# Yoga Asana Detection

Real-time yoga posture recognition from an ordinary webcam.
The system detects the body with MediaPipe Pose and classifies the posture with a BiLSTM + Attention model.

**Recognised asanas:** Extended Side Angle Pose, High Lunge, Low Lunge, Pyramid Pose

---

## Package contents

```
yoga_Asana_Detection/
├── run.py                              Ready-to-run application (webcam, video or image)
├── requirements.txt                    Python dependencies (pinned versions)
└── models/
    ├── bilstm_attention.keras          Trained model
    └── bilstm_attention_labels.npy     Class labels
```

## Requirements

- **Python 3.11** (64-bit). Other Python versions are not supported by the pinned TensorFlow / MediaPipe versions.
- Windows 10/11, macOS or Linux
- A webcam (for live mode)
- About 1.5 GB of free disk space for dependencies
- No GPU required, runs in real time on a laptop CPU

Check your Python version:

```bash
python --version        # should print Python 3.11.x
```

On Windows, if you have several Python versions installed, use `py -3.11` in place of `python` in Step 2.
Python 3.11 can be downloaded from https://www.python.org/downloads/release/python-3119/

---

## Installation

### Step 1: Download the project

```bash
git clone https://github.com/Aakash-2904/yoga_Asana_Detection.git
cd yoga_Asana_Detection
```

(Or download the ZIP from GitHub using **Code → Download ZIP**, extract it, and open a terminal in the extracted folder.)

### Step 2: Create a virtual environment

**Windows (PowerShell)**

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation script, run this once and try again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**Windows (Command Prompt)**

```cmd
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
```

**macOS / Linux**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Your prompt should now start with `(.venv)`.

### Step 3: Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This downloads about 1 GB (mostly TensorFlow) and may take a few minutes.

### Step 4: Verify the installation

```bash
python -c "import tensorflow as tf, mediapipe as mp; print(tf.__version__, mp.__version__)"
```

Expected output: `2.15.1 0.10.14`

---

## Running

Make sure the virtual environment is active (`(.venv)` in your prompt), then:

| Mode | Command |
|---|---|
| Live webcam | `python run.py` |
| Another webcam | `python run.py --camera 1` |
| Video file | `python run.py --video path/to/video.mp4` |
| Single image | `python run.py --image path/to/photo.jpg` |

Press **Esc** or **Q** to close the window.

### How to use the live mode

1. Run `python run.py`. A window opens showing the camera feed.
2. Step back until your **whole body, including both wrists and both ankles, is in the frame**.
3. Hold one of the four asanas.
4. The detected asana and its confidence are shown at the top of the window.

| On-screen message | Meaning |
|---|---|
| `High Lunge (97%)` (green) | Asana recognised with confidence above 75% |
| `Make sure your full body is visible` | Wrists or ankles are out of frame. Step back. |
| `Pose not recognised` | Body is visible but the posture doesn't match a trained asana closely enough |
| `No person detected` | No body found in the frame |

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `Could not open video source: 0` / no camera image | Close other apps using the camera (Zoom, Teams, Camera app). On Windows, allow camera access under **Settings → Privacy & security → Camera**. Try `--camera 1`. |
| `No module named tensorflow` (or `cv2`, `mediapipe`) | The virtual environment isn't active. Repeat the activate command from Step 2. |
| Windows: `Could not install packages due to an OSError ... Long Path support`, then `No module named 'tensorflow.python'` | The folder path is too long for Windows. Move the project to a short path such as `C:\yoga_Asana_Detection`, delete `.venv` and repeat Steps 2–3. (Alternatively, enable long paths: https://pip.pypa.io/warnings/enable-long-paths) |
| `pip` can't find `tensorflow==2.15.1` | You're not using Python 3.11. Recreate the virtual environment with Python 3.11. |
| `AttributeError: module 'mediapipe' has no attribute 'solutions'` | A newer MediaPipe is installed. Run `pip install mediapipe==0.10.14`. |
| Predictions flicker or stay at "Pose not recognised" | Improve lighting, wear clothing that contrasts with the background, and keep the whole body in view. |

---

## Technical details

| | |
|---|---|
| Pose estimation | MediaPipe Pose, 33 body landmarks per frame |
| Model input | 33 × (x, y) landmarks, relative to the nose and normalised by torso length |
| Architecture | Joint embedding → BiLSTM (64) → additive attention over joints → softmax |
| Output | One of 4 asanas plus a confidence score |
| Confidence threshold | 0.75 (set by `CONFIDENCE_THRESHOLD` in `run.py`) |

### Using the model in your own code

`run.py` can be imported directly:

```python
import cv2
from run import AsanaClassifier

clf = AsanaClassifier()
frame = cv2.imread("photo.jpg")
landmarks, label, confidence, message = clf.predict(frame)
print(label or message, confidence)
```

The model uses a custom `Attention` layer defined in `run.py`. Importing `run` registers it, so the model can also be loaded with `keras.models.load_model("models/bilstm_attention.keras")` afterwards.
