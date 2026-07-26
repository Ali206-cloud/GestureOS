"""
Hand Gesture Desktop Control
============================
Control your mouse and a handful of desktop actions using webcam + hand
tracking (mediapipe) instead of a physical mouse.

Run:
    pip install opencv-python mediapipe pyautogui numpy
    python gesture_control.py

Controls while running:
    p        pause / resume gesture control
    h        show / hide the on-screen gesture legend
    q / Esc  quit

Gesture reference
------------------
| Gesture                          | Action                         |
|-----------------------------------|--------------------------------|
| Index finger only                 | Move mouse                     |
| Thumb + index pinch                | Left click (hold+move = drag)  |
| Thumb + middle finger pinch        | Right click                    |
| Thumb + pinky pinch                | Double click                   |
| Index + middle finger up           | Scroll (move hand up/down)     |
| Index + middle + ring up (hold)    | New tab (Ctrl+T)                |
| Thumb only (hold)                  | Open File Explorer (Win+E)      |
| All five fingers up (hold)         | Screenshot                      |
| Closed fist (hold)                 | Close tab (Ctrl+W)              |
| Pinky only (hold)                  | Minimize / show desktop (Win+D) |
| Ring + pinky up                    | Volume control (move up/down)   |
| Thumb + index up, NOT pinched       | Zoom in/out (Ctrl+Scroll)       |
"""

import time
import math

import cv2
import numpy as np
import pyautogui
import mediapipe as mp


# ---------------------------------------------------------------------------
# Tunables - tweak these first if something feels too sensitive / too slow.
# ---------------------------------------------------------------------------
CAM_INDEX = 0
CAM_W, CAM_H = 640, 480
DISPLAY_W, DISPLAY_H = 1280, 720

# Smoothing for mouse movement - higher = smoother but laggier.
SMOOTHENING = 7
# Portion of the frame (in px, from each edge) that maps to the full screen.
# Keeping a margin makes it easier to reach screen edges/corners.
FRAME_MARGIN = 100

# Pinch distance (px) below which fingers count as "touching".
PINCH_CLOSE = 40
# Pinch distance above which fingers count as "apart" again (must be a bit
# larger than PINCH_CLOSE so tiny hand jitter doesn't flicker the state).
PINCH_OPEN = 55
# How long a thumb+index pinch must be held before it turns into a drag
# instead of a click. Lower = drags start easier, but taps must be quicker.
DRAG_HOLD_TIME = 0.5
# Minimum time between two separate clicks of the same kind.
CLICK_COOLDOWN = 0.6

# How long a "hold" gesture (new tab, explorer, screenshot, fist, minimize)
# must be held before it fires, and how long before it can fire again.
HOLD_TIME = 1.3
HOLD_COOLDOWN = 2.2
# Closing a fist is easy to do by accident, so it gets a longer hold.
CLOSE_TAB_HOLD_TIME = 1.8

VOLUME_STEP_PX = 20      # vertical px of hand movement per volume tick
ZOOM_STEP_PX = 22        # px change in pinch distance per zoom tick
SCROLL_SENSITIVITY = 0.9


def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


# ---------------------------------------------------------------------------
# Hand tracking wrapper
# ---------------------------------------------------------------------------
class HandDetector:
    def __init__(self, max_hands=1, detection_con=0.7, track_con=0.7, model_complexity=0):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            model_complexity=model_complexity,
            min_detection_confidence=detection_con,
            min_tracking_confidence=track_con,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.tip_ids = [4, 8, 12, 16, 20]
        self.lm_list = []
        self.results = None

    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False
        self.results = self.hands.process(img_rgb)
        if draw and self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, hand_no=0):
        self.lm_list = []
        if self.results and self.results.multi_hand_landmarks:
            if hand_no >= len(self.results.multi_hand_landmarks):
                return self.lm_list
            hand = self.results.multi_hand_landmarks[hand_no]
            h, w = img.shape[:2]
            for idx, lm in enumerate(hand.landmark):
                self.lm_list.append([idx, int(lm.x * w), int(lm.y * h)])
        return self.lm_list

    def fingers_up(self):
        """Returns [thumb, index, middle, ring, pinky] as 1 (up) or 0 (down)."""
        if not self.lm_list:
            return [0, 0, 0, 0, 0]
        fingers = []
        # Thumb: compare x, since it moves sideways rather than up/down.
        if self.lm_list[self.tip_ids[0]][1] < self.lm_list[self.tip_ids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # Other four fingers: tip above the joint two below it = "up".
        for i in range(1, 5):
            if self.lm_list[self.tip_ids[i]][2] < self.lm_list[self.tip_ids[i] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def distance(self, p1, p2):
        x1, y1 = self.lm_list[p1][1:]
        x2, y2 = self.lm_list[p2][1:]
        return math.hypot(x2 - x1, y2 - y1)


# ---------------------------------------------------------------------------
# Reusable "hold this gesture for N seconds to fire" helper, used for the
# discrete actions (new tab, explorer, screenshot, close tab, minimize).
# Firing once requires the gesture to be released and re-formed again.
# ---------------------------------------------------------------------------
class HoldTrigger:
    def __init__(self, hold_time=HOLD_TIME, cooldown=HOLD_COOLDOWN):
        self.hold_time = hold_time
        self.cooldown = cooldown
        self._start_time = None
        self._last_fired = 0.0
        self._fired_this_hold = False
        self._active = False

    def update(self, is_held):
        now = time.time()
        if not is_held:
            self._active = False
            self._start_time = None
            self._fired_this_hold = False
            return False

        if not self._active:
            self._active = True
            self._start_time = now
            self._fired_this_hold = False
            return False

        held_long_enough = (now - self._start_time) >= self.hold_time
        cooldown_ok = (now - self._last_fired) >= self.cooldown
        if held_long_enough and cooldown_ok and not self._fired_this_hold:
            self._fired_this_hold = True
            self._last_fired = now
            return True
        return False

    def progress(self):
        if self._start_time is None:
            return 0.0
        return clamp((time.time() - self._start_time) / self.hold_time, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Reusable pinch handler for thumb+index. Handles the click-vs-drag decision
# in one place: release a quick pinch -> click, hold it -> drag.
# ---------------------------------------------------------------------------
class PinchDrag:
    def __init__(self, close_dist=PINCH_CLOSE, open_dist=PINCH_OPEN,
                 drag_hold_time=DRAG_HOLD_TIME, cooldown=CLICK_COOLDOWN):
        self.close_dist = close_dist
        self.open_dist = open_dist
        self.drag_hold_time = drag_hold_time
        self.cooldown = cooldown
        self.pinching = False
        self.dragging = False
        self._pinch_start = None
        self._last_click = 0.0

    def update(self, dist):
        """Returns one of: None, 'click', 'drag_start', 'drag_move', 'drag_end'."""
        now = time.time()

        if not self.pinching:
            if dist < self.close_dist:
                self.pinching = True
                self._pinch_start = now
            return None

        # Currently pinching.
        if self.dragging:
            if dist > self.open_dist:
                self.pinching = False
                self.dragging = False
                return 'drag_end'
            return 'drag_move'

        # Pinching but not yet dragging.
        if dist > self.open_dist:
            # Released before the drag threshold -> a click.
            self.pinching = False
            held_time = now - self._pinch_start if self._pinch_start else 0
            if held_time < self.drag_hold_time and (now - self._last_click) > self.cooldown:
                self._last_click = now
                return 'click'
            return None

        if (now - self._pinch_start) >= self.drag_hold_time:
            self.dragging = True
            return 'drag_start'

        return None

    def release(self):
        """Force-release (e.g. hand left the frame)."""
        if self.dragging:
            self.dragging = False
            self.pinching = False
            return 'drag_end'
        self.pinching = False
        return None


class SimplePinch:
    """Fires once as soon as the pinch closes (for right-click / double-click,
    which don't need drag support)."""

    def __init__(self, close_dist=PINCH_CLOSE, cooldown=CLICK_COOLDOWN):
        self.close_dist = close_dist
        self.cooldown = cooldown
        self.pinching = False
        self._last_fired = 0.0

    def update(self, dist):
        now = time.time()
        if dist < self.close_dist:
            if not self.pinching and (now - self._last_fired) > self.cooldown:
                self.pinching = True
                self._last_fired = now
                return True
            self.pinching = True
            return False
        self.pinching = False
        return False


def main():
    print("Starting Hand Gesture Control...")

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"Could not open webcam at index {CAM_INDEX}. "
              f"Check your camera connection or try a different CAM_INDEX.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    try:
        cap.set(cv2.CAP_PROP_FPS, 30)
    except Exception:
        pass

    screen_w, screen_h = pyautogui.size()
    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0

    window_name = "Hand Gesture Control"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
    cv2.resizeWindow(window_name, DISPLAY_W, DISPLAY_H)

    detector = HandDetector(max_hands=1)

    # State
    plocX, plocY = 0, 0
    left_pinch = PinchDrag()
    right_click = SimplePinch()
    double_click = SimplePinch()

    scroll_active = False
    prev_scroll_y = 0

    volume_active = False
    prev_volume_y = 0

    zoom_active = False
    prev_zoom_dist = None

    new_tab_trigger = HoldTrigger()
    explorer_trigger = HoldTrigger()
    screenshot_trigger = HoldTrigger()
    close_tab_trigger = HoldTrigger(hold_time=CLOSE_TAB_HOLD_TIME)
    minimize_trigger = HoldTrigger()

    paused = False
    show_help = False
    action_label = "Ready"
    pTime = 0

    help_lines = [
        "p = pause/resume   h = toggle help   q/Esc = quit",
        "Index only ................. Move mouse",
        "Thumb+Index pinch .......... Left click (hold = drag)",
        "Thumb+Middle pinch ......... Right click",
        "Thumb+Pinky pinch .......... Double click",
        "Index+Middle up ............ Scroll (move hand up/down)",
        "Index+Middle+Ring (hold) ... New Tab",
        "Thumb only (hold) .......... Open File Explorer",
        "All 5 fingers (hold) ....... Screenshot",
        "Fist (hold) ................ Close Tab",
        "Pinky only (hold) .......... Minimize / Show Desktop",
        "Ring+Pinky up ............... Volume (move hand up/down)",
        "Thumb+Index up, not pinched . Zoom (spread/pinch apart)",
    ]

    def reset_transient_state():
        nonlocal scroll_active, volume_active, zoom_active
        scroll_active = False
        volume_active = False
        zoom_active = False
        if left_pinch.release() == 'drag_end':
            pyautogui.mouseUp()
        new_tab_trigger.update(False)
        explorer_trigger.update(False)
        screenshot_trigger.update(False)
        close_tab_trigger.update(False)
        minimize_trigger.update(False)

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to read from webcam.")
            break

        img = cv2.flip(img, 1)
        img = detector.find_hands(img, draw=True)
        h, w = img.shape[:2]
        lm_list = detector.find_position(img)

        hold_progress = 0.0

        if paused:
            action_label = "PAUSED (press 'p' to resume)"
            reset_transient_state()

        elif not lm_list:
            action_label = "No hand detected"
            reset_transient_state()

        else:
            fingers = detector.fingers_up()
            action_label = f"Fingers: {fingers}"

            # --- Move mouse: index finger only ---
            if fingers == [0, 1, 0, 0, 0]:
                x1, y1 = lm_list[8][1:]
                x = np.interp(x1, (FRAME_MARGIN, w - FRAME_MARGIN), (0, screen_w))
                y = np.interp(y1, (FRAME_MARGIN, h - FRAME_MARGIN), (0, screen_h))
                clocX = plocX + (x - plocX) / SMOOTHENING
                clocY = plocY + (y - plocY) / SMOOTHENING
                clocX = clamp(clocX, 0, screen_w - 1)
                clocY = clamp(clocY, 0, screen_h - 1)
                pyautogui.moveTo(clocX, clocY)
                plocX, plocY = clocX, clocY
                action_label = "Move Mouse"

            # --- Left click / drag: thumb + index pinch ---
            if len(lm_list) >= 9:
                dist = detector.distance(4, 8)
                event = left_pinch.update(dist)
                if event == 'click':
                    pyautogui.click()
                    action_label = "Left Click"
                elif event == 'drag_start':
                    pyautogui.mouseDown()
                    action_label = "Drag Started"
                elif event == 'drag_move':
                    x1, y1 = lm_list[8][1:]
                    x = np.interp(x1, (FRAME_MARGIN, w - FRAME_MARGIN), (0, screen_w))
                    y = np.interp(y1, (FRAME_MARGIN, h - FRAME_MARGIN), (0, screen_h))
                    clocX = plocX + (x - plocX) / SMOOTHENING
                    clocY = plocY + (y - plocY) / SMOOTHENING
                    clocX = clamp(clocX, 0, screen_w - 1)
                    clocY = clamp(clocY, 0, screen_h - 1)
                    pyautogui.moveTo(clocX, clocY)
                    plocX, plocY = clocX, clocY
                    action_label = "Dragging..."
                elif event == 'drag_end':
                    pyautogui.mouseUp()
                    action_label = "Drag Released"

            # --- Right click: thumb + middle finger pinch ---
            if len(lm_list) >= 13:
                dist_r = detector.distance(4, 12)
                if right_click.update(dist_r):
                    pyautogui.click(button='right')
                    action_label = "Right Click"

            # --- Double click: thumb + pinky pinch ---
            if len(lm_list) >= 21:
                dist_d = detector.distance(4, 20)
                if double_click.update(dist_d):
                    pyautogui.doubleClick()
                    action_label = "Double Click"

            # --- Scroll: index + middle up, move hand up/down ---
            if fingers == [0, 1, 1, 0, 0]:
                y_index = lm_list[8][2]
                if scroll_active:
                    delta_y = prev_scroll_y - y_index
                    if abs(delta_y) > 2:
                        pyautogui.scroll(int(delta_y * SCROLL_SENSITIVITY))
                else:
                    scroll_active = True
                prev_scroll_y = y_index
                action_label = "Scroll"
            else:
                scroll_active = False

            # --- New tab: index+middle+ring up, held ---
            is_new_tab = fingers == [0, 1, 1, 1, 0]
            if new_tab_trigger.update(is_new_tab):
                pyautogui.hotkey('ctrl', 't')
                action_label = "New Tab"
            elif is_new_tab:
                hold_progress = new_tab_trigger.progress()
                action_label = f"Hold for New Tab... {int(hold_progress * 100)}%"

            # --- Open File Explorer: thumb only, held ---
            is_explorer = fingers == [1, 0, 0, 0, 0]
            if explorer_trigger.update(is_explorer):
                pyautogui.hotkey('win', 'e')
                action_label = "Open Explorer"
            elif is_explorer:
                hold_progress = explorer_trigger.progress()
                action_label = f"Hold for Explorer... {int(hold_progress * 100)}%"

            # --- Screenshot: all five fingers up, held ---
            is_screenshot = fingers == [1, 1, 1, 1, 1]
            if screenshot_trigger.update(is_screenshot):
                filename = f"gesture_screenshot_{int(time.time())}.png"
                pyautogui.screenshot(filename)
                action_label = f"Screenshot: {filename}"
            elif is_screenshot:
                hold_progress = screenshot_trigger.progress()
                action_label = f"Hold for Screenshot... {int(hold_progress * 100)}%"

            # --- Close tab: closed fist, held longer ---
            is_fist = fingers == [0, 0, 0, 0, 0]
            if close_tab_trigger.update(is_fist):
                pyautogui.hotkey('ctrl', 'w')
                action_label = "Close Tab (Fist Held)"
            elif is_fist:
                hold_progress = close_tab_trigger.progress()
                action_label = f"Hold to Close Tab... {int(hold_progress * 100)}%"

            # --- Minimize / Show Desktop: pinky only, held ---
            is_minimize = fingers == [0, 0, 0, 0, 1]
            if minimize_trigger.update(is_minimize):
                pyautogui.hotkey('win', 'd')
                action_label = "Minimize / Show Desktop"
            elif is_minimize:
                hold_progress = minimize_trigger.progress()
                action_label = f"Hold to Minimize... {int(hold_progress * 100)}%"

            # --- Volume: ring + pinky up, move hand vertically ---
            is_volume = fingers == [0, 0, 0, 1, 1]
            if is_volume:
                y_ref = lm_list[8][2]
                if volume_active:
                    delta_y = prev_volume_y - y_ref
                    if abs(delta_y) > VOLUME_STEP_PX:
                        ticks = min(int(abs(delta_y) / VOLUME_STEP_PX), 5)
                        key = 'volumeup' if delta_y > 0 else 'volumedown'
                        for _ in range(ticks):
                            pyautogui.press(key)
                        prev_volume_y = y_ref
                        action_label = f"Volume {'Up' if delta_y > 0 else 'Down'}"
                else:
                    volume_active = True
                    prev_volume_y = y_ref
                    action_label = "Volume Mode (move hand up/down)"
            else:
                volume_active = False

            # --- Zoom: thumb + index up, NOT pinched, change distance ---
            is_zoom_pose = fingers[0] == 1 and fingers[1] == 1 and fingers[2:] == [0, 0, 0]
            if is_zoom_pose and len(lm_list) >= 9:
                zoom_dist = detector.distance(4, 8)
                if zoom_dist > PINCH_OPEN:  # not actually pinching/clicking
                    if zoom_active and prev_zoom_dist is not None:
                        delta_d = zoom_dist - prev_zoom_dist
                        if abs(delta_d) > ZOOM_STEP_PX:
                            ticks = int(abs(delta_d) / ZOOM_STEP_PX)
                            pyautogui.keyDown('ctrl')
                            pyautogui.scroll(ticks if delta_d > 0 else -ticks)
                            pyautogui.keyUp('ctrl')
                            prev_zoom_dist = zoom_dist
                            action_label = f"Zoom {'In' if delta_d > 0 else 'Out'}"
                    else:
                        zoom_active = True
                        prev_zoom_dist = zoom_dist
                        action_label = "Zoom Mode (spread/pinch apart)"
                else:
                    zoom_active = False
            else:
                zoom_active = False

        # --- HUD ---
        cTime = time.time()
        fps = 1 / (cTime - pTime) if cTime != pTime else 0
        pTime = cTime

        overlay_color = (0, 0, 200) if paused else (0, 255, 0)
        cv2.rectangle(img, (0, 0), (320, 60), (0, 0, 0), cv2.FILLED)
        cv2.putText(img, f'Action: {action_label}', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, overlay_color, 2)
        cv2.putText(img, f'FPS: {int(fps)}   (p=pause h=help)', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, overlay_color, 2)

        if hold_progress > 0.0:
            bar_x, bar_y, bar_w, bar_h = 10, 65, 200, 18
            cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), cv2.FILLED)
            filled_w = int(bar_w * hold_progress)
            cv2.rectangle(img, (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_h), (0, 200, 255), cv2.FILLED)
            cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 1)

        if show_help:
            panel_h = 24 + 20 * len(help_lines)
            cv2.rectangle(img, (0, h - panel_h), (560, h), (0, 0, 0), cv2.FILLED)
            for i, line in enumerate(help_lines):
                cv2.putText(img, line, (10, h - panel_h + 20 + i * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        try:
            _, _, win_w, win_h = cv2.getWindowImageRect(window_name)
            display_img = cv2.resize(img, (win_w, win_h), interpolation=cv2.INTER_LINEAR) \
                if win_w > 0 and win_h > 0 else img
        except Exception:
            display_img = img

        cv2.imshow(window_name, display_img)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('p'):
            paused = not paused
            if paused:
                reset_transient_state()
        elif key == ord('h'):
            show_help = not show_help

    if left_pinch.dragging:
        pyautogui.mouseUp()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
