import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent  # repository root
POSES_DIR = ROOT / "data" / "poses"
MODELS_DIR = ROOT / "models"


TEXT_FONT = cv2.FONT_HERSHEY_DUPLEX  # one clean font for all on-screen text


def put(img, text, org, font, scale, color, thickness=1):
    """Draw anti-aliased text. `font` is accepted for call compatibility; TEXT_FONT is always used."""
    if scale < 1.0:  # DUPLEX is already double-stroked; thicker strokes smudge small text
        thickness = 1
    cv2.putText(img, text, org, TEXT_FONT, scale, color, thickness, cv2.LINE_AA)


def put_center(img, text, y, scale, color, thickness=1, width=None):
    """Draw text centred horizontally within the first `width` pixels of img."""
    width = img.shape[1] if width is None else width
    w = cv2.getTextSize(text, TEXT_FONT, scale, thickness)[0][0]
    put(img, text, (max((width - w) // 2, 10), y), None, scale, color, thickness)


# Function to check if the user is in the frame
def inFrame(lst):
    if lst[28].visibility > 0.6 and lst[27].visibility > 0.6 and lst[15].visibility > 0.6 and lst[16].visibility > 0.6:
        return True
    return False

# Load the model and labels
model = load_model(MODELS_DIR / "model_001.h5")
labels = np.load(MODELS_DIR / "labels_001.npy")

# Initialize MediaPipe solutions
holistic = mp.solutions.pose
holis = holistic.Pose()
hands = mp.solutions.hands.Hands()
drawing = mp.solutions.drawing_utils

# Function to count fingers
def count_fingers(hand_landmarks):
    count = 0
    if hand_landmarks:
        landmarks = hand_landmarks[0].landmark
        if landmarks[5].y > landmarks[8].y:
            count += 1
        if landmarks[9].y > landmarks[12].y:
            count += 1
        if landmarks[13].y > landmarks[16].y:
            count += 1
        if landmarks[17].y > landmarks[20].y:
            count += 1
        if landmarks[0].x > landmarks[4].x:
            count += 1
    return count

# Display names for each trained asana, keyed by label without the "NN_" prefix
ASANA_INFO = {
    "extended side angle pose": {"name": "Extended Side Angle Pose", "sanskrit": "Utthita Parsvakonasana"},
    "high lunge": {"name": "High Lunge", "sanskrit": "Ashta Chandrasana"},
    "low lunge": {"name": "Low Lunge", "sanskrit": "Anjaneyasana"},
    "pyramid pose": {"name": "Pyramid Pose", "sanskrit": "Parsvottanasana"},
}

PANEL_WIDTH = 460
RETURN_SECONDS = 3  # how long "Asana Completed!" shows before going back to the menu


def asana_key(label):
    """'01_extended side angle pose' -> 'extended side angle pose'"""
    label = str(label)
    return label.split("_", 1)[1] if "_" in label else label


def wrap_text(text, max_width, font, scale, thickness):
    """Split text into lines that fit within max_width pixels."""
    lines, line = [], ""
    for word in text.split():
        candidate = f"{line} {word}".strip()
        if cv2.getTextSize(candidate, TEXT_FONT, scale, thickness)[0][0] <= max_width or not line:
            line = candidate
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def load_reference_poses(labels):
    """Reference skeleton per asana: median of each landmark over the recorded frames in data/poses."""
    refs = {}
    for label in labels:
        path = POSES_DIR / f"{label}.npy"
        if path.exists():
            # (x, y) are relative to the nose, in normalised image units; scale to the 640x480 camera aspect
            refs[label] = np.median(np.load(path), axis=0).reshape(33, 2) * [640, 480]
    return refs


LEFT_COLOR, RIGHT_COLOR, CENTER_COLOR = (255, 170, 60), (60, 170, 255), (235, 235, 235)


def side_color(i, j):
    """MediaPipe numbers left-body landmarks odd and right-body even (from 11 on)."""
    if i >= 11 and j >= 11 and i % 2 == j % 2:
        return LEFT_COLOR if i % 2 else RIGHT_COLOR
    return CENTER_COLOR


def draw_reference(height, label):
    """Side panel: the asana's name and a stick figure of the target pose."""
    panel = np.full((height, PANEL_WIDTH, 3), (40, 30, 25), dtype="uint8")
    font = cv2.FONT_HERSHEY_SIMPLEX
    x = 20
    info = ASANA_INFO.get(asana_key(label), {"name": asana_key(label).title(), "sanskrit": ""})

    y = 50
    for line in wrap_text(info["name"], PANEL_WIDTH - 40, font, 0.95, 2):
        put(panel, line, (x, y), font, 0.95, (0, 215, 255), 2)
        y += 38
    if info["sanskrit"]:
        put(panel, info["sanskrit"], (x, y), font | cv2.FONT_ITALIC, 0.65, (190, 190, 190), 1)
        y += 30
    cv2.line(panel, (x, y), (PANEL_WIDTH - 20, y), (90, 90, 90), 1)
    put(panel, "Match this pose", (x, y + 35), font, 0.75, (255, 255, 255), 2)

    pts = REFERENCE.get(label)
    top, bottom = y + 60, height - 90
    if pts is None:
        put(panel, "No recorded data for this asana.", (x, top + 30), font, 0.6, (200, 200, 200), 1)
        return panel

    # fit the skeleton into the box below the title, keeping its proportions
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    box_w, box_h = PANEL_WIDTH - 60, bottom - top
    scale = min(box_w / max(hi[0] - lo[0], 1e-6), box_h / max(hi[1] - lo[1], 1e-6)) * 0.85
    center = np.array([PANEL_WIDTH / 2, (top + bottom) / 2])
    xy = ((pts - (lo + hi) / 2) * scale + center).astype(int)

    torso = np.linalg.norm((xy[11] + xy[12]) / 2 - (xy[23] + xy[24]) / 2)
    head_r = max(int(torso * 0.2), 10)  # head size from torso length (works for side-on and front-on poses)
    cv2.circle(panel, tuple(xy[0]), head_r, CENTER_COLOR, 3, cv2.LINE_AA)
    for i, j in mp.solutions.pose.POSE_CONNECTIONS:
        if i <= 10 or j <= 10:  # skip the face mesh lines; the head circle stands in for them
            continue
        cv2.line(panel, tuple(xy[i]), tuple(xy[j]), side_color(i, j), 5, cv2.LINE_AA)
    for i in range(11, 33):
        cv2.circle(panel, tuple(xy[i]), 6, (0, 215, 255), -1, cv2.LINE_AA)

    put(panel, "blue = left side   orange = right side", (x, height - 55), font, 0.5, (190, 190, 190), 1)
    put(panel, "Keep wrists and ankles in view.", (x, height - 28), font, 0.55, (160, 160, 160), 1)
    return panel


def asana_name(label):
    info = ASANA_INFO.get(asana_key(label))
    return info["name"] if info else asana_key(label).title()


# Hold times offered in step 2: fingers shown -> seconds
HOLD_TIMES = {1: 10, 2: 20, 3: 30, 4: 45, 5: 60}
CONFIRM_SECONDS = 1.5  # a finger count must stay steady this long to count as a choice


class GestureConfirm:
    """Accept a finger count only after it has been held steady for CONFIRM_SECONDS."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.count, self.since = 0, time.time()

    def update(self, count):
        now = time.time()
        if count != self.count:
            self.count, self.since = count, now
        return min((now - self.since) / CONFIRM_SECONDS, 1.0) if count > 0 else 0.0


def draw_progress(window, x, y, width, progress, color):
    cv2.rectangle(window, (x, y), (x + width, y + 14), (90, 90, 90), 1)
    if progress > 0:
        cv2.rectangle(window, (x + 2, y + 2), (x + 2 + int((width - 4) * progress), y + 12), color, -1)


def draw_asana_menu(height, labels, highlight):
    """Right-hand panel for step 1: which finger count picks which asana."""
    panel = np.full((height, PANEL_WIDTH, 3), (40, 30, 25), dtype="uint8")
    font = cv2.FONT_HERSHEY_SIMPLEX
    put(panel, "Choose an asana", (20, 50), font, 0.95, (0, 215, 255), 2)
    put(panel, "Show this many fingers:", (20, 90), font, 0.6, (200, 200, 200), 1)
    y = 150
    for n, label in enumerate(labels, 1):
        active = n == highlight
        if active:
            cv2.rectangle(panel, (12, y - 40), (PANEL_WIDTH - 12, y + 22), (70, 60, 40), -1)
        cv2.circle(panel, (45, y - 9), 22, (0, 215, 255), -1 if active else 2)
        put(panel, str(n), (37, y), font, 0.8, (40, 30, 25) if active else (0, 215, 255), 2)
        for i, line in enumerate(wrap_text(asana_name(label), PANEL_WIDTH - 110, font, 0.7, 2)):
            put(panel, line, (85, y + i * 28), font, 0.7, (255, 255, 255), 2)
        y += 90
    put(panel, "Keyboard: press 1-%d" % len(labels), (20, height - 60), font, 0.6, (160, 160, 160), 1)
    put(panel, "Esc: quit", (20, height - 30), font, 0.6, (160, 160, 160), 1)
    return panel


def fit_window_to_screen(name, width, height):
    """Create the window at real screen pixels and return the scale that makes the canvas fit the screen.

    Without DPI awareness, Windows display scaling (e.g. 125 %) stretches the window and blurs the text,
    so the process is marked DPI-aware and any shrinking is done with smooth INTER_AREA resampling.
    """
    scale = 1.0
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor DPI aware (Windows 8.1+)
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()
        sw, sh = ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1)
        scale = min(1.0, 0.97 * sw / width, 0.88 * sh / height)
    except Exception:  # not Windows: show at full size
        pass
    cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
    cv2.resizeWindow(name, int(width * scale), int(height * scale))
    return scale


REFERENCE = load_reference_poses(labels)
cap = cv2.VideoCapture(0)
chosen_asana = None
chosen_time = None
start_time = None
timer_running = False
held_time = 0  # seconds already held before the last pause
completed = False
completed_at = 0.0
display_phase = True
time_setting_phase = False
countdown_phase = False
countdown = 5
confirm = GestureConfirm()
hand_lowered = True  # a selection step waits for the hand to drop, so the previous gesture is not re-read
back_since = None  # when the two-hands 'back' gesture started
key = -1
last_fingers = None
DISPLAY_SCALE = fit_window_to_screen("window", 940 + PANEL_WIDTH, 940)
FONT = TEXT_FONT

while True:
    lst = []

    ret, frm = cap.read()
    if not ret:
        raise SystemExit("Could not read a frame from the webcam (cv2.VideoCapture(0)).")
    window = np.zeros((940, 940, 3), dtype="uint8")
    frm = cv2.flip(frm, 1)

    res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
    hand_res = hands.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

    frm = cv2.blur(frm, (4, 4))

    fingers = count_fingers(hand_res.multi_hand_landmarks) if hand_res.multi_hand_landmarks else 0
    if (display_phase or time_setting_phase) and fingers != last_fingers:
        print(f"Detected fingers: {fingers}")
        last_fingers = fingers
    pressed = key - ord("0") if ord("1") <= key <= ord("9") else None  # number key pressed last frame
    n_hands = len(hand_res.multi_hand_landmarks) if hand_res.multi_hand_landmarks else 0

    # ---- Back: press B / Backspace, or hold up both hands for CONFIRM_SECONDS ----
    posing = not (display_phase or time_setting_phase or countdown_phase or completed)
    body_in_view = bool(res.pose_landmarks) and inFrame(res.pose_landmarks.landmark)
    # while posing with the full body in view the gesture is ignored, so arms raised in a pose cannot trigger it
    back_gesture_allowed = time_setting_phase or countdown_phase or (posing and not body_in_view)
    if back_gesture_allowed and n_hands >= 2:
        back_since = back_since or time.time()
        back_progress = min((time.time() - back_since) / CONFIRM_SECONDS, 1.0)
    else:
        back_since, back_progress = None, 0.0
    if (key in (ord("b"), ord("B"), 8) or back_progress >= 1.0) and not display_phase and not completed:
        back_since, back_progress = None, 0.0
        timer_running, held_time, start_time = False, 0, None
        if time_setting_phase:  # hold-time step -> asana menu
            time_setting_phase, display_phase, chosen_asana = False, True, None
            print("Back to the asana menu")
        else:  # countdown or posing -> hold-time step
            countdown_phase, time_setting_phase, chosen_time = False, True, None
            print("Back to the hold-time step")
        confirm.reset()
        hand_lowered = False

    started_in_menu = display_phase  # this frame draws the menu, so no Back hint
    if display_phase:
        put(window, "Step 1 of 2: choose an asana", (50, 50), FONT, 1, (0, 215, 255), 2)
        put(window, f"Show 1-{len(labels)} fingers on one hand and hold them still.", (50, 95), FONT, 0.7, (255, 255, 255), 1)
        put(window, "The list on the right shows which number picks which asana.", (50, 125), FONT, 0.7, (255, 255, 255), 1)
        if not hand_lowered:
            hand_lowered = fingers == 0
        shown = fingers if hand_lowered and 0 < fingers <= len(labels) else 0
        progress = confirm.update(shown)
        if not hand_lowered:
            status = "Lower your hands first, then show fingers."
        elif hand_res.multi_hand_landmarks:
            status = f"Fingers detected: {fingers}"
        else:
            status = "No hand detected: hold your hand up, palm facing the camera"
        put(window, status, (50, 175), FONT, 0.7, (120, 220, 120) if shown else (160, 160, 160), 2)
        draw_progress(window, 50, 190, 400, progress, (0, 215, 255))
        choice = pressed if pressed and pressed <= len(labels) else (shown if progress >= 1.0 else None)
        if choice:
            chosen_asana = labels[choice - 1]
            print(f"Chosen asana: {asana_name(chosen_asana)}")
            display_phase = False
            time_setting_phase = True
            confirm.reset()
            hand_lowered = False
    elif time_setting_phase:
        put(window, f"Step 2 of 2: hold time for {asana_name(chosen_asana)}", (50, 45), FONT, 0.85, (0, 215, 255), 2)
        options = "   ".join(f"{n} = {s}s" for n, s in HOLD_TIMES.items())
        put(window, "Fingers:  " + options, (50, 90), FONT, 0.7, (255, 255, 255), 2)
        if not hand_lowered:
            hand_lowered = fingers == 0
            put(window, "Lower your hand first, then show fingers for the time.", (50, 135), FONT, 0.7, (160, 160, 160), 1)
            progress = 0.0
        else:
            put(window, "Show fingers and hold them still (or press 1-5).", (50, 135), FONT, 0.7, (255, 255, 255), 1)
            progress = confirm.update(fingers if fingers in HOLD_TIMES and n_hands < 2 else 0)
            put(window, f"Fingers detected: {fingers}", (50, 175), FONT, 0.7, (120, 220, 120) if fingers else (160, 160, 160), 2)
            draw_progress(window, 50, 190, 400, progress, (0, 215, 255))
        choice = pressed if pressed in HOLD_TIMES else (confirm.count if hand_lowered and progress >= 1.0 else None)
        if choice:
            chosen_time = HOLD_TIMES[choice]
            print(f"Hold time: {chosen_time}s")
            time_setting_phase = False
            countdown_phase = True
            start_time = time.time()
    elif countdown_phase:
        elapsed = time.time() - start_time

        if elapsed <= countdown:
            put_center(window, f"Starting in {int(countdown - elapsed)}", 190, 1.8, (0, 255, 0), 3, width=940)
        else:
            countdown_phase = False
            start_time = time.time()
            timer_running = True
    elif completed:  # show the result briefly, then go back to the main (asana selection) screen
        remaining = RETURN_SECONDS - (time.time() - completed_at)
        put_center(window, "Asana Completed!", 150, 1.5, (0, 255, 0), 3, width=940)
        put_center(window, f"Back to the menu in {max(int(remaining) + 1, 1)}...", 200, 0.9, (255, 255, 255), 1, width=940)
        if remaining <= 0:
            print("Returning to the asana menu")
            chosen_asana = chosen_time = start_time = None
            timer_running = completed = False
            held_time = 0
            display_phase = True
            confirm.reset()
    else:
        if res.pose_landmarks and inFrame(res.pose_landmarks.landmark):
            for i in res.pose_landmarks.landmark:
                lst.append(i.x - res.pose_landmarks.landmark[0].x)
                lst.append(i.y - res.pose_landmarks.landmark[0].y)

            lst = np.array(lst).reshape(1, -1)
            p = model.predict(lst)
            pred = labels[np.argmax(p)]

            if pred == chosen_asana and p[0][np.argmax(p)] > 0.75:
                if not timer_running and not completed:  # resume after a pause
                    timer_running = True
                    start_time = time.time()
                if timer_running:
                    elapsed_time = int(held_time + time.time() - start_time)
                    if elapsed_time <= chosen_time:
                        put_center(window, f"Timer: {elapsed_time}s / {chosen_time}s", 190, 1.4, (0, 255, 0), 2, width=940)
                    else:
                        completed = True
                        completed_at = time.time()
                        timer_running = False
                        print(f"Completed: {asana_name(chosen_asana)} for {chosen_time}s")
            else:
                put_center(window, "Perform the asana correctly", 190, 1.3, (60, 60, 255), 2, width=940)
                if timer_running:  # pause, keeping the time held so far
                    held_time += time.time() - start_time
                timer_running = False

        else:
            put_center(frm, "Make sure your full body is visible", 450, 0.8, (60, 60, 255), 2)
            if timer_running:  # pause, keeping the time held so far
                held_time += time.time() - start_time
            timer_running = False

    drawing.draw_landmarks(frm, res.pose_landmarks, holistic.POSE_CONNECTIONS,
                           connection_drawing_spec=drawing.DrawingSpec(color=(255, 255, 255), thickness=6),
                           landmark_drawing_spec=drawing.DrawingSpec(color=(0, 0, 255), circle_radius=3, thickness=3))

    if hand_res.multi_hand_landmarks:
        for hand_landmarks in hand_res.multi_hand_landmarks:
            drawing.draw_landmarks(frm, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
                                   connection_drawing_spec=drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                                   landmark_drawing_spec=drawing.DrawingSpec(color=(0, 0, 255), circle_radius=2, thickness=2))

    if not started_in_menu and not display_phase and not completed:  # use the step as it is now
        target = "asana menu" if time_setting_phase else "hold-time step"
        posing_now = not (time_setting_phase or countdown_phase)
        how = "step closer and hold up both hands" if posing_now else "hold up both hands"
        label = f"B = back to {target}" + ("  (or step closer + both hands)" if posing_now else "  (or both hands up)")
        w = cv2.getTextSize(label, TEXT_FONT, 0.55, 1)[0][0]
        cv2.rectangle(frm, (0, 0), (w + 20, 34 if back_progress <= 0 else 46), (0, 0, 0), -1)
        put(frm, label, (10, 23), None, 0.55, (230, 230, 230), 1)
        if back_progress > 0:
            cv2.rectangle(frm, (10, 32), (10 + int(w * back_progress), 40), (255, 170, 60), -1)

    # Calculate the coordinates to place the frame in the middle
    window_height, window_width = window.shape[:2]
    frame_height, frame_width = 480, 640
    start_y = (window_height - frame_height) // 2
    start_x = (window_width - frame_width) // 2

    window[start_y:start_y + frame_height, start_x:start_x + frame_width, :] = cv2.resize(frm, (frame_width, frame_height))
    if chosen_asana is None:  # step 1: menu of asanas beside the camera view
        side = draw_asana_menu(window_height, labels, confirm.count if display_phase else 0)
    else:  # afterwards: the chosen asana's name and a stick figure of the target pose
        side = draw_reference(window_height, chosen_asana)
    window = np.hstack((window, side))
    try:
        _, _, ww, wh = cv2.getWindowImageRect("window")
    except Exception:
        ww = wh = 0
    if ww > 0 and wh > 0 and (ww, wh) != (window.shape[1], window.shape[0]):
        h0, w0 = window.shape[:2]
        sc = min(ww / w0, wh / h0)
        fitted = cv2.resize(window, (int(w0 * sc), int(h0 * sc)), interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_CUBIC)
        canvas = np.zeros((wh, ww, 3), dtype="uint8")
        oy, ox = (wh - fitted.shape[0]) // 2, (ww - fitted.shape[1]) // 2
        canvas[oy:oy + fitted.shape[0], ox:ox + fitted.shape[1]] = fitted
        window = canvas
    cv2.imshow("window", window)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        cv2.destroyAllWindows()
        cap.release()
        break
