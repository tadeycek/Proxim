# Proxim — Hand-Tracking Cursor Control

Control your mouse with hand gestures via webcam. No special hardware needed — just a camera and a Wayland desktop.

## What it does

`hand_cursor.py` uses your webcam and [MediaPipe](https://developers.google.com/mediapipe) to detect your hand in real time and map it to cursor movement and clicks:

| Gesture | Action |
|---|---|
| Open hand (move) | Moves the cursor — palm centre maps to screen coordinates |
| Pinch (thumb + index) | Holds left mouse button |
| Fist (all fingers curled) | Holds right mouse button |

Releasing the gesture releases the button, so drag-and-drop works naturally.

## How it works

- **Hand detection** — MediaPipe's `HandLandmarker` model detects 21 landmarks on your hand each frame
- **Cursor movement** — the palm centre (wrist + MCP knuckles average) is mapped to screen coordinates
- **Adaptive smoothing** — a rolling time-window average smooths jitter; the window shrinks when you move fast (responsive) and grows when you're still (stable)
- **Dead zone** — the cursor only moves if the hand has moved more than 6 px, preventing drift while holding still
- **Gesture debounce** — a gesture must hold for 2 consecutive frames before activating, avoiding accidental clicks
- **Input injection** — cursor movement and clicks are sent via [`ydotool`](https://github.com/ReimuNotMoe/ydotool), which works on Wayland

## Requirements

### Python packages

```bash
pip install mediapipe opencv-python numpy
```

### System

- **Wayland** compositor
- **ydotool** + **ydotoold** daemon

```bash
# Install ydotool (Arch)
sudo pacman -S ydotool

# Start the daemon (required before running)
sudo ydotoold
```

The MediaPipe hand landmarker model (`hand_landmarker.task`, ~9 MB) is downloaded automatically on first run.

## Usage

```bash
python hand_cursor.py
```

A preview window shows the webcam feed with the hand skeleton overlaid. Press **Q** or **Esc** to quit.

## Configuration

All tunable parameters are at the top of `hand_cursor.py`:

| Constant | Default | Description |
|---|---|---|
| `SCREEN_W` / `SCREEN_H` | `1920` / `1080` | Your monitor resolution |
| `CAMERA_INDEX` | `0` | Webcam device index |
| `FLIP_HORIZONTAL` | `True` | Mirror feed for natural feel |
| `DEAD_ZONE_PX` | `6` | Min movement before cursor updates |
| `PINCH_THRESHOLD` | `0.06` | Normalised thumb-index distance for left-click |
| `FIST_THRESHOLD` | `0.13` | Normalised fingertip-palm distance for right-click |
| `GESTURE_FRAMES` | `2` | Frames a gesture must hold before activating |
| `SMOOTH_MIN_SEC` | `0.08` | Smoothing window at full speed |
| `SMOOTH_MAX_SEC` | `0.55` | Smoothing window when still |

## Current state

- [x] Real-time hand tracking via MediaPipe Tasks API
- [x] Palm-centre cursor positioning
- [x] Adaptive velocity-based smoothing
- [x] Dead zone to suppress jitter
- [x] Pinch → left-click/hold
- [x] Fist → right-click/hold
- [x] Gesture debouncing (N-frame confirmation)
- [x] HUD overlay with gesture feedback and landmark skeleton
- [x] Auto-download of the MediaPipe model on first run
- [x] Clean button release on exit / hand loss
