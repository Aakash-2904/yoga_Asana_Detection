# Yoga Asana Detection

BiLSTM + Attention model for classifying yoga postures from MediaPipe Pose landmarks.

Trained asanas: Extended Side Angle Pose, High Lunge, Low Lunge, Pyramid Pose.

## Files

| File | Description |
|---|---|
| `models/bilstm_attention.keras` | Trained BiLSTM + Attention model |
| `models/bilstm_attention_labels.npy` | Class labels |
| `requirements.txt` | Pinned dependencies |

## Setup

Requires **Python 3.11**.

```bash
git clone https://github.com/Aakash-2904/yoga_Asana_Detection.git
cd yoga_Asana_Detection

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

pip install --upgrade pip
pip install -r requirements.txt
```

Check the install (should print `2.15.1 0.10.14`):

```bash
python -c "import tensorflow as tf, mediapipe as mp; print(tf.__version__, mp.__version__)"
```

Keep the pinned versions. Later MediaPipe releases remove the `mp.solutions` API.

## Usage

The model uses a custom `Attention` layer, which must be defined before loading.
Input is the 33 MediaPipe Pose landmarks as (x, y) relative to the nose, normalised by torso length: shape `(N, 33, 2)`.

```python
import numpy as np
import tensorflow as tf
import keras
from keras import layers

@keras.utils.register_keras_serializable(package="yoga")
class Attention(layers.Layer):
    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        d = input_shape[-1]
        self.W = self.add_weight(name="W", shape=(d, self.units), initializer="glorot_uniform")
        self.b = self.add_weight(name="b", shape=(self.units,), initializer="zeros")
        self.v = self.add_weight(name="v", shape=(self.units, 1), initializer="glorot_uniform")
        super().build(input_shape)

    def call(self, h):
        score = tf.matmul(tf.tanh(tf.matmul(h, self.W) + self.b), self.v)
        alpha = tf.nn.softmax(score, axis=1)
        return tf.reduce_sum(alpha * h, axis=1), tf.squeeze(alpha, -1)

    def get_config(self):
        return {**super().get_config(), "units": self.units}


def to_sequence(landmarks):
    """MediaPipe pose landmarks -> (1, 33, 2), nose-relative and torso-normalised."""
    seq = np.array([[lm.x - landmarks[0].x, lm.y - landmarks[0].y] for lm in landmarks])
    torso = np.linalg.norm(seq[[11, 12]].mean(0) - seq[[23, 24]].mean(0))
    return (seq / max(torso, 1e-6))[None]


model = keras.models.load_model("models/bilstm_attention.keras")
labels = np.load("models/bilstm_attention_labels.npy")

# landmarks = results.pose_landmarks.landmark  (from mediapipe.solutions.pose.Pose)
probs = model.predict(to_sequence(landmarks), verbose=0)[0]
if probs.max() > 0.75:
    print(labels[probs.argmax()], probs.max())
```

Only classify frames where both wrists (15, 16) and both ankles (27, 28) are visible.
