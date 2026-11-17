"""Yoga asana detection: BiLSTM + Attention over MediaPipe Pose landmarks.

Usage:
    python run.py                      # live webcam (default camera 0)
    python run.py --camera 1           # use another webcam
    python run.py --video clip.mp4     # run on a video file
    python run.py --image photo.jpg    # classify a single image

Press Esc or Q to close the window.
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hide TensorFlow info logs

import cv2
import keras
import mediapipe as mp
import numpy as np
import tensorflow as tf
from keras import layers

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "bilstm_attention.keras"
LABELS_PATH = ROOT / "models" / "bilstm_attention_labels.npy"

CONFIDENCE_THRESHOLD = 0.75
# Landmarks that must be visible: left/right wrist and left/right ankle.
REQUIRED_LANDMARKS = (15, 16, 27, 28)
VISIBILITY_THRESHOLD = 0.6


@keras.utils.register_keras_serializable(package="yoga")
class Attention(layers.Layer):
    """Additive attention pooling over the joint sequence (needed to load the model)."""

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


def pretty(label):
    """'02_high lunge' -> 'High Lunge'"""
    return str(label).split("_", 1)[-1].title()


class AsanaClassifier:
    def __init__(self):
        if not MODEL_PATH.exists():
            sys.exit(f"Model not found: {MODEL_PATH}")
        self.model = keras.models.load_model(MODEL_PATH)
        self.labels = [pretty(l) for l in np.load(LABELS_PATH)]
        self.pose = mp.solutions.pose.Pose()

    @staticmethod
    def body_in_frame(landmarks):
        return all(landmarks[i].visibility > VISIBILITY_THRESHOLD for i in REQUIRED_LANDMARKS)

    @staticmethod
    def to_sequence(landmarks):
        """33 landmarks -> (1, 33, 2): nose-relative, normalised by torso length."""
        seq = np.array([[lm.x - landmarks[0].x, lm.y - landmarks[0].y] for lm in landmarks],
                       dtype="float32")
        torso = np.linalg.norm(seq[[11, 12]].mean(0) - seq[[23, 24]].mean(0))
        return (seq / max(torso, 1e-6))[None]

    def predict(self, frame_bgr):
        """Returns (pose_landmarks, label or None, confidence, status message)."""
        res = self.pose.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if not res.pose_landmarks:
            return None, None, 0.0, "No person detected"
        lms = res.pose_landmarks.landmark
        if not self.body_in_frame(lms):
            return res.pose_landmarks, None, 0.0, "Make sure your full body is visible"
        probs = self.model.predict(self.to_sequence(lms), verbose=0)[0]
        idx = int(probs.argmax())
        conf = float(probs[idx])
        if conf < CONFIDENCE_THRESHOLD:
            return res.pose_landmarks, None, conf, "Pose not recognised"
        return res.pose_landmarks, self.labels[idx], conf, ""


def annotate(frame, landmarks, label, conf, message):
    draw = mp.solutions.drawing_utils
    if landmarks:
        draw.draw_landmarks(frame, landmarks, mp.solutions.pose.POSE_CONNECTIONS,
                            draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3),
                            draw.DrawingSpec(color=(255, 255, 255), thickness=3))
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 0), -1)
    if label:
        text, color = f"{label}  ({conf:.0%})", (0, 255, 0)
    else:
        text, color = message, (0, 0, 255)
    cv2.putText(frame, text, (12, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return frame


def run_image(clf, path):
    frame = cv2.imread(str(path))
    if frame is None:
        sys.exit(f"Could not read image: {path}")
    lms, label, conf, msg = clf.predict(frame)
    print(f"{label} ({conf:.1%})" if label else msg)
    cv2.imshow("Yoga Asana Detection", annotate(frame, lms, label, conf, msg))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_stream(clf, source, mirror):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"Could not open video source: {source}")
    print("Running... press Esc or Q in the video window to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            if isinstance(source, int):
                print("Could not read a frame from the webcam. Is another app using it?")
            break
        if mirror:
            frame = cv2.flip(frame, 1)
        lms, label, conf, msg = clf.predict(frame)
        cv2.imshow("Yoga Asana Detection", annotate(frame, lms, label, conf, msg))
        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="Yoga asana detection")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    src.add_argument("--video", type=Path, help="path to a video file")
    src.add_argument("--image", type=Path, help="path to an image file")
    args = ap.parse_args()

    print("Loading model...")
    clf = AsanaClassifier()
    print("Recognised asanas:", ", ".join(clf.labels))

    if args.image:
        run_image(clf, args.image)
    elif args.video:
        run_stream(clf, str(args.video), mirror=False)
    else:
        run_stream(clf, args.camera, mirror=True)


if __name__ == "__main__":
    main()
