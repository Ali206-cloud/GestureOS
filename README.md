# GestureDeck

Control your computer with hand gestures — no mouse, no keyboard, just your webcam.

GestureDeck uses [MediaPipe](https://developers.google.com/mediapipe) hand tracking and OpenCV to convert hand poses into system actions: moving the cursor, clicking, scrolling, zooming, and more.

## Features

- Full mouse control — move, click, right-click, double-click, and click-and-drag
- Scroll and zoom with natural hand motions
- Quick actions — new tab, close tab, open file explorer, screenshot, minimize, volume control
- Hold-to-confirm protection on disruptive actions to prevent accidental triggers
- On-screen progress bar, FPS counter, and gesture legend

## Gesture Guide

| Gesture | Action |
|---|---|
| Index finger only | Move mouse |
| Thumb + index pinch | Left click (hold and move to drag) |
| Thumb + middle finger pinch | Right click |
| Thumb + pinky pinch | Double click |
| Index + middle finger up | Scroll |
| Index + middle + ring up (hold) | New tab |
| Thumb only (hold) | Open File Explorer |
| All five fingers up (hold) | Screenshot |
| Closed fist (hold) | Close tab |
| Pinky only (hold) | Minimize / show desktop |
| Ring + pinky up | Volume control |
| Thumb + index up, not pinched | Zoom in/out |

Press `p` to pause/resume detection, `h` to toggle the gesture legend, and `q` or `Esc` to quit.

## Requirements

- Python 3.8+
- A webcam
- Windows, macOS, or Linux (some shortcuts, like `Win+E`, are Windows-specific)

## Installation

```bash
git clone https://github.com/<your-username>/gesturedeck.git
cd gesturedeck
pip install -r requirements.txt
```

**`requirements.txt`:**
```
opencv-python
mediapipe
pyautogui
numpy
```

## Usage

```bash
python hand_gesture_control.py
```

A window opens with the webcam feed and hand landmarks overlaid. The detected action and FPS are shown in the top-left corner.

## How It Works

MediaPipe detects 21 hand landmarks per frame, which determine which fingers are extended. Finger patterns map to actions — quick gestures (clicks, scrolling) fire instantly, while disruptive ones require a brief hold to confirm. PyAutoGUI then simulates the corresponding mouse or keyboard action.

## Roadmap

- Cross-platform shortcut mapping
- Configurable gesture bindings
- Two-hand gesture support
  xD ;) 
