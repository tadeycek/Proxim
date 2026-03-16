#!/usr/bin/env python3
"""
hand_cursor.py — Webcam-based cursor control via hand tracking on Wayland (ydotool).
Uses the mediapipe Tasks API (mediapipe >= 0.10).

Gestures:
    Pinch (index + thumb)  →  left-click / hold left button while pinching
    Fist (all fingers curl) →  right-click / hold right button while fisting

Requirements:
    pip install mediapipe opencv-python numpy

System:
    ydotoold must be running (start with: sudo ydotoold)
    The hand landmarker model (~9 MB) is auto-downloaded on first run.
"""

import collections
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np

# ── Tunable constants ─────────────────────────────────────────────────────────

SMOOTH_MIN_SEC  = 0.08   # Smoothing window when moving fast (responsive)
SMOOTH_MAX_SEC  = 0.55   # Smoothing window when nearly still (most smoothing)
VELOCITY_SCALE  = 800    # px/s that maps to the minimum window
DEAD_ZONE_PX    = 6      # Cursor won't move unless hand moved more than this many px
PINCH_THRESHOLD = 0.06   # Normalised dist(thumb_tip, index_tip) for left-click
FIST_THRESHOLD  = 0.13   # Normalised avg dist(fingertips, palm) for right-click
GESTURE_FRAMES  = 2      # Consecutive frames a gesture must hold before activating
CAMERA_INDEX    = 0      # Webcam device index
FLIP_HORIZONTAL = True   # Mirror the feed for natural feel

# ── Screen resolution (adjust to your actual monitor) ────────────────────────
SCREEN_W = 1920
SCREEN_H = 1080

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# ── Hand landmark indices ──────────────────────────────────────────────────────
IDX_THUMB_TIP   = 4
IDX_INDEX_TIP   = 8
IDX_MIDDLE_TIP  = 12
IDX_RING_TIP    = 16
IDX_PINKY_TIP   = 20
IDX_PALM        = (0, 5, 9, 13, 17)   # wrist + MCP knuckles = palm centre
IDX_FINGERTIPS  = (8, 12, 16, 20)     # all four non-thumb tips

# MediaPipe hand skeleton connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]

# ydotool button codes: button | flags (0x40=down, 0x80=up, 0xC0=click)
_BTN_LEFT   = 0x00
_BTN_RIGHT  = 0x01
_DOWN       = 0x40
_UP         = 0x80

# ─────────────────────────────────────────────────────────────────────────────


def ensure_model() -> None:
    if MODEL_PATH.exists():
        return
    print(f"[INFO] Downloading hand landmarker model → {MODEL_PATH} (~9 MB)…")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[INFO] Download complete.")
    except Exception as exc:
        sys.exit(f"[ERROR] Failed to download model: {exc}\n"
                 f"        Download manually from:\n        {MODEL_URL}\n"
                 f"        and place it next to hand_cursor.py")


def check_ydotoold() -> bool:
    try:
        return subprocess.run(["pgrep", "-x", "ydotoold"],
                              capture_output=True).returncode == 0
    except FileNotFoundError:
        return False


def move_cursor(x: int, y: int) -> None:
    subprocess.Popen(
        ["ydotool", "mousemove", "--absolute", "-x", str(x), "-y", str(y)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def _mouse_event(button: int, flag: int) -> None:
    subprocess.Popen(
        ["ydotool", "click", hex(button | flag)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def btn_down(button: int) -> None:
    _mouse_event(button, _DOWN)


def btn_up(button: int) -> None:
    _mouse_event(button, _UP)


# ── Gesture detection ─────────────────────────────────────────────────────────

def pinch_dist(lm) -> float:
    """Normalised distance between thumb tip and index tip."""
    dx = lm[IDX_THUMB_TIP].x - lm[IDX_INDEX_TIP].x
    dy = lm[IDX_THUMB_TIP].y - lm[IDX_INDEX_TIP].y
    return float(np.sqrt(dx * dx + dy * dy))


def fist_dist(lm) -> float:
    """Average normalised distance of all four fingertips to the palm centre."""
    cx = float(np.mean([lm[i].x for i in IDX_PALM]))
    cy = float(np.mean([lm[i].y for i in IDX_PALM]))
    dists = [np.sqrt((lm[i].x - cx) ** 2 + (lm[i].y - cy) ** 2)
             for i in IDX_FINGERTIPS]
    return float(np.mean(dists))



# ── HUD drawing ───────────────────────────────────────────────────────────────

# Landmark indices that get highlighted per gesture
_PINCH_HIGHLIGHTS = {IDX_THUMB_TIP, IDX_INDEX_TIP}
_FIST_HIGHLIGHTS  = set(IDX_FINGERTIPS)


def draw_hand(frame, landmarks, mode_text: str, gesture: str) -> None:
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    skeleton_color = {
        "pinch": (0,  80, 255),
        "fist":  (255, 60,   0),
    }.get(gesture, (0, 200, 60))

    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], skeleton_color, 2, cv2.LINE_AA)

    highlights = {"pinch": _PINCH_HIGHLIGHTS, "fist": _FIST_HIGHLIGHTS}.get(gesture, set())
    for i, (x, y) in enumerate(pts):
        if i in highlights:
            cv2.circle(frame, (x, y), 9, skeleton_color, -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 9, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.circle(frame, (x, y), 5, (255, 255, 255), -1, cv2.LINE_AA)

    label_color = skeleton_color if gesture else (0, 220, 60)
    cv2.putText(frame, mode_text, (12, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, label_color, 2, cv2.LINE_AA)
    cv2.putText(frame, "Press Q to quit", (12, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    ensure_model()

    if not check_ydotoold():
        print(
            "[WARN] ydotoold daemon does not appear to be running.\n"
            "       Start it with:  sudo ydotoold\n"
            "       Cursor movement will fail until the daemon is up.",
            file=sys.stderr,
        )

    base_options = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=mp_vision.RunningMode.VIDEO,
    )
    detector = mp_vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open camera index {CAMERA_INDEX}")

    print(f"[INFO] Camera {CAMERA_INDEX} at "
          f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}×"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"[INFO] Screen {SCREEN_W}×{SCREEN_H}")
    print("[INFO] Pinch = left-hold | Fist = right-hold | Q to quit")

    # ── State ─────────────────────────────────────────────────────────────────
    # Rolling position buffer: deque of (timestamp, screen_x, screen_y)
    pos_buf: collections.deque = collections.deque()

    # Dead zone: last position actually sent to the OS
    last_sent_x: float = SCREEN_W / 2
    last_sent_y: float = SCREEN_H / 2

    # Velocity tracking for adaptive smoothing
    prev_raw_x: float = SCREEN_W / 2
    prev_raw_y: float = SCREEN_H / 2
    prev_raw_t: float = 0.0

    # Button hold state
    left_held:  bool = False
    right_held: bool = False

    # Debounce counters — gesture must persist N frames before activating
    pinch_frames: int = 0
    fist_frames:  int = 0

    start_time = time.monotonic()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            if FLIP_HORIZONTAL:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.monotonic() - start_time) * 1000)
            result = detector.detect_for_video(mp_image, timestamp_ms)

            mode_text = "Tracking"
            active_gesture = ""

            if result.hand_landmarks:
                lm = result.hand_landmarks[0]

                # ── Cursor movement (palm centre → screen) ────────────────
                raw_x = float(np.mean([lm[i].x for i in IDX_PALM])) * SCREEN_W
                raw_y = float(np.mean([lm[i].y for i in IDX_PALM])) * SCREEN_H
                now_t = time.monotonic()

                # Adaptive window: shrinks when moving fast, grows when still
                if prev_raw_t > 0:
                    dt = now_t - prev_raw_t
                    vel = (np.sqrt((raw_x - prev_raw_x) ** 2 +
                                   (raw_y - prev_raw_y) ** 2) / dt
                           if dt > 0 else 0.0)
                else:
                    vel = 0.0
                t = float(np.clip(vel / VELOCITY_SCALE, 0.0, 1.0))
                window = SMOOTH_MAX_SEC - t * (SMOOTH_MAX_SEC - SMOOTH_MIN_SEC)
                prev_raw_x, prev_raw_y, prev_raw_t = raw_x, raw_y, now_t

                pos_buf.append((now_t, raw_x, raw_y))
                cutoff = now_t - window
                while pos_buf and pos_buf[0][0] < cutoff:
                    pos_buf.popleft()

                smooth_x = float(np.mean([p[1] for p in pos_buf]))
                smooth_y = float(np.mean([p[2] for p in pos_buf]))

                # Dead zone: only send a new position if we've moved enough
                ddx = smooth_x - last_sent_x
                ddy = smooth_y - last_sent_y
                if np.sqrt(ddx * ddx + ddy * ddy) > DEAD_ZONE_PX:
                    move_cursor(int(np.clip(smooth_x, 0, SCREEN_W - 1)),
                                int(np.clip(smooth_y, 0, SCREEN_H - 1)))
                    last_sent_x, last_sent_y = smooth_x, smooth_y

                # ── Gesture classification ─────────────────────────────────
                pd = pinch_dist(lm)
                fd = fist_dist(lm)

                # Fist takes priority; pinch only counts when NOT a fist
                is_fist  = fd < FIST_THRESHOLD
                is_pinch = (pd < PINCH_THRESHOLD) and not is_fist

                # Debounce counters
                pinch_frames = (pinch_frames + 1) if is_pinch else 0
                fist_frames  = (fist_frames  + 1) if is_fist  else 0

                pinch_confirmed = pinch_frames >= GESTURE_FRAMES
                fist_confirmed  = fist_frames  >= GESTURE_FRAMES

                # ── Left button (pinch) ────────────────────────────────────
                if pinch_confirmed and not left_held:
                    btn_down(_BTN_LEFT)
                    left_held = True
                elif not pinch_confirmed and left_held:
                    btn_up(_BTN_LEFT)
                    left_held = False

                # ── Right button (fist) ────────────────────────────────────
                if fist_confirmed and not right_held:
                    btn_down(_BTN_RIGHT)
                    right_held = True
                elif not fist_confirmed and right_held:
                    btn_up(_BTN_RIGHT)
                    right_held = False

                # ── HUD text ───────────────────────────────────────────────
                if left_held:
                    active_gesture = "pinch"
                    mode_text = f"LEFT HELD  (pinch d={pd:.3f})"
                elif right_held:
                    active_gesture = "fist"
                    mode_text = f"RIGHT HELD (fist d={fd:.3f})"
                else:
                    mode_text = f"Tracking  p={pd:.3f} f={fd:.3f}"

                draw_hand(frame, lm, mode_text, active_gesture)

            else:
                # Release any held buttons if hand disappears
                if left_held:
                    btn_up(_BTN_LEFT)
                    left_held = False
                if right_held:
                    btn_up(_BTN_RIGHT)
                    right_held = False
                pinch_frames = fist_frames = 0

                h = frame.shape[0]
                cv2.putText(frame, "No hand detected", (12, 34),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60, 60, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, "Press Q to quit", (12, h - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

            cv2.imshow("Hand Cursor Tracker", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                print("[INFO] Quitting…")
                break

    finally:
        # Always release buttons on exit so nothing stays stuck
        if left_held:
            btn_up(_BTN_LEFT)
        if right_held:
            btn_up(_BTN_RIGHT)
        cap.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
