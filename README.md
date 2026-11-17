# Yoga Asana Detection

Real-time yoga posture recognition from an ordinary webcam.
The system detects the body with MediaPipe Pose and classifies the posture with a trained neural network.
You choose an asana and a hold time with hand gestures, and a timer runs only while you hold the asana correctly.

**Recognised asanas:** Extended Side Angle Pose, High Lunge, Low Lunge, Pyramid Pose

---

## Package contents

```
yoga_Asana_Detection/
├── run.py                              Ready-to-run application (live webcam)
├── requirements.txt                    Python dependencies (pinned versions)
├── data/
│   └── poses/*.npy                     Recorded landmarks, used to draw the reference pose
└── models/
    ├── model_001.h5                    Model used by run.py
    ├── labels_001.npy                  Class labels used by run.py
    ├── bilstm_attention.keras          BiLSTM + Attention model (from the paper)
    └── bilstm_attention_labels.npy     Its class labels
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

```bash
python run.py
```

The window opens on the default webcam. Press **Esc** to quit.

### How to use it

1. **Choose an asana.** Show 1–4 fingers on one hand and hold them still for about 1.5 seconds. The panel on the right shows which number picks which asana. You can also press keys 1–4.
2. **Choose a hold time.** Lower your hand, then show fingers for the time: 1 = 10 s, 2 = 20 s, 3 = 30 s, 4 = 45 s, 5 = 60 s (or press 1–5).
3. **Get into position.** A 5-second countdown starts. Step back until your **whole body, including both wrists and both ankles, is in the frame**.
4. **Hold the asana.** The right-hand panel shows the target pose. The timer only counts while the chosen asana is recognised with more than 75% confidence. If you lose the pose, it pauses and shows `Perform the asana correctly`.
5. When the hold time is reached, `Asana Completed!` is shown and you go back to the menu after 3 seconds.

**Going back a step:** press **B** or **Backspace**, or hold up both hands for 1.5 seconds. The two-hands gesture is ignored while your full body is in view during the pose, so raising your arms in an asana won't trigger it.

| On-screen message | Meaning |
|---|---|
| `Timer: 12s / 30s` (green) | Correct asana held, timer running |
| `Perform the asana correctly` | The chosen asana isn't recognised with enough confidence. The timer is paused. |
| `Make sure your full body is visible` | Wrists or ankles are out of frame. Step back. |

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `Could not read a frame from the webcam` / no camera image | Close other apps using the camera (Zoom, Teams, Camera app). On Windows, allow camera access under **Settings → Privacy & security → Camera**. To use another camera, change `cv2.VideoCapture(0)` in `run.py` to `cv2.VideoCapture(1)`. |
| Fingers not detected | Hold your hand up, palm facing the camera, fingers spread, in good light. Or use the number keys. |
| `No module named tensorflow` (or `cv2`, `mediapipe`) | The virtual environment isn't active. Repeat the activate command from Step 2. |
| Windows: `Could not install packages due to an OSError ... Long Path support`, then `No module named 'tensorflow.python'` | The folder path is too long for Windows. Move the project to a short path such as `C:\yoga_Asana_Detection`, delete `.venv` and repeat Steps 2–3. (Alternatively, enable long paths: https://pip.pypa.io/warnings/enable-long-paths) |
| `pip` can't find `tensorflow==2.15.1` | You're not using Python 3.11. Recreate the virtual environment with Python 3.11. |
| `AttributeError: module 'mediapipe' has no attribute 'solutions'` | A newer MediaPipe is installed. Run `pip install mediapipe==0.10.14`. |
| Timer stays paused at "Perform the asana correctly" | Improve lighting, wear clothing that contrasts with the background, and keep the whole body in view. |

---

## Technical details

| | |
|---|---|
| Pose estimation | MediaPipe Pose, 33 body landmarks per frame |
| Gesture input | MediaPipe Hands, finger count held steady for 1.5 s |
| Model | `models/model_001.h5` with `models/labels_001.npy` |
| Model input | 33 × (x, y) landmarks, relative to the nose |
| Output | One of 4 asanas plus a confidence score |
| Timer threshold | Confidence above 0.75 for the chosen asana |
| Reference pose | Median of the recorded landmarks in `data/poses/*.npy` |
