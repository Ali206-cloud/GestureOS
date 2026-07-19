import cv2
import mediapipe as mp
import pyautogui
import time
import math
import numpy as np


def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


class HandDetector:
    def __init__(self, mode=False, maxHands=1, detectionCon=0.7, trackCon=0.6, modelComplexity=0):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            model_complexity=modelComplexity,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon,
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.lmList = []

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgRGB.flags.writeable = False
        self.results = self.hands.process(imgRGB)
        if draw and self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        self.lmList = []
        self.bbox = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            xList = []
            yList = []
            h, w = img.shape[:2]
            for id, lm in enumerate(myHand.landmark):
                px, py = int(lm.x * w), int(lm.y * h)
                xList.append(px)
                yList.append(py)
                self.lmList.append([id, px, py])
                if draw:
                    cv2.circle(img, (px, py), 4, (255, 0, 255), cv2.FILLED)
            self.bbox = [min(xList), min(yList), max(xList), max(yList)]
        return self.lmList, self.bbox

    def fingersUp(self):
        if not self.lmList:
            return [0, 0, 0, 0, 0]
        fingers = []
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def findDistance(self, p1, p2, img=None):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        if img is not None:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.circle(img, (x1, y1), 8, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 8, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), 8, (0, 0, 255), cv2.FILLED)
        return length, img, [x1, y1, x2, y2, cx, cy]


class HoldTrigger:
    """
    Tracks how long a particular gesture has been held continuously.
    An action only fires once the gesture has been held for `hold_time`
    seconds without interruption, and then won't fire again until
    `cooldown` seconds have passed AND the gesture has been released
    and re-formed.
    """

    def __init__(self, hold_time=0.8, cooldown=1.5):
        self.hold_time = hold_time
        self.cooldown = cooldown
        self._active_name = None
        self._start_time = None
        self._last_fired_time = 0
        self._fired_for_current_hold = False

    def update(self, current_name):
        """
        Call once per frame with the name of the gesture currently detected
        (or None if no matching gesture is active this frame).
        Returns True exactly once per successful "hold", when the hold_time
        threshold is crossed (and cooldown has elapsed since the last fire).
        """
        now = time.time()

        if current_name is None:
            # Gesture released - reset everything so next hold starts fresh.
            self._active_name = None
            self._start_time = None
            self._fired_for_current_hold = False
            return False

        if current_name != self._active_name:
            # A new/different gesture just started being held.
            self._active_name = current_name
            self._start_time = now
            self._fired_for_current_hold = False
            return False

        # Same gesture is still being held.
        held_long_enough = (now - self._start_time) >= self.hold_time
        cooldown_ok = (now - self._last_fired_time) >= self.cooldown

        if held_long_enough and cooldown_ok and not self._fired_for_current_hold:
            self._fired_for_current_hold = True
            self._last_fired_time = now
            return True

        return False

    def progress(self):
        """Returns 0.0-1.0 progress toward firing, for on-screen feedback."""
        if self._start_time is None:
            return 0.0
        return clamp((time.time() - self._start_time) / self.hold_time, 0.0, 1.0)


def main():
    print("Starting Hand Gesture Control...")

    wCam, hCam = 640, 480
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam. Check your camera connection.")
        return

    print("Webcam opened successfully.")
    screenW, screenH = pyautogui.size()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, wCam)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hCam)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    try:
        cap.set(cv2.CAP_PROP_FPS, 30)
    except Exception:
        pass

    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0

    window_name = "Hand Gesture Control"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
    cv2.resizeWindow(window_name, 1280, 720)

    detector = HandDetector(maxHands=1, detectionCon=0.7, trackCon=0.7, modelComplexity=0)

    pTime = 0
    plocX, plocY = 0, 0
    clocX, clocY = 0, 0
    smoothening = 5

    action_label = "Ready"

    # --- Click gestures keep a simple cooldown (they're intentional pinches,
    # not accidental poses you'd hold by mistake), but the cooldown is now
    # longer to avoid accidental rapid double/triple firing. ---
    last_click_time = 0
    last_rclick_time = 0
    last_dclick_time = 0
    click_cooldown = 0.6

    scroll_active = False
    prev_scroll_y = 0
    scroll_sensitivity = 1.5

    # --- Discrete/disruptive actions now require the gesture to be HELD
    # for a bit before firing, instead of firing the instant fingers match.
    # This is what stops "New Tab" / "Open Explorer" from triggering the
    # split-second your hand passes through that shape. ---
    new_tab_trigger = HoldTrigger(hold_time=0.9, cooldown=2.0)
    explorer_trigger = HoldTrigger(hold_time=0.9, cooldown=2.0)
    screenshot_trigger = HoldTrigger(hold_time=0.9, cooldown=2.0)
    close_tab_trigger = HoldTrigger(hold_time=1.2, cooldown=2.0)

    # --- more hold-triggers for the extended features ---
    minimize_trigger = HoldTrigger(hold_time=0.9, cooldown=2.0)     # pinky only
    play_pause_trigger = HoldTrigger(hold_time=0.9, cooldown=1.0)   # ring only
    switch_app_trigger = HoldTrigger(hold_time=0.9, cooldown=1.2)   # index + pinky
    lock_screen_trigger = HoldTrigger(hold_time=1.6, cooldown=3.0)  # thumb + pinky, extra-safe

    # --- click-and-drag state (thumb+index pinch, held + moved) ---
    dragging = False
    drag_pinch_threshold = 45
    drag_release_threshold = 65  # slightly larger than press threshold, avoids flicker

    # --- volume control (ring + pinky up, move hand vertically) ---
    volume_active = False
    prev_volume_y = 0
    volume_step_px = 12  # pixels of vertical movement per volume tick

    # --- zoom control (thumb + index up but NOT pinched, change distance) ---
    zoom_active = False
    prev_zoom_dist = None
    zoom_dist_per_tick = 15  # px change in pinch distance per zoom tick

    # --- pause toggle and on-screen help ---
    paused = False
    show_help = False

    help_lines = [
        "p = pause/resume control   h = toggle this help   q/Esc = quit",
        "Index only ......... Move mouse",
        "Thumb+Index pinch ... Left click  (hold+move = drag)",
        "Thumb+Middle pinch .. Right click",
        "Thumb+Pinky pinch ... Double click",
        "Index+Middle up ..... Scroll (move hand up/down)",
        "Index+Middle+Ring ... Hold: New Tab",
        "Thumb only .......... Hold: Open File Explorer",
        "All 5 fingers ....... Hold: Screenshot",
        "Fist ................ Hold: Close Tab",
        "Pinky only .......... Hold: Minimize (Show Desktop)",
        "Ring only ........... Hold: Play/Pause media",
        "Index+Pinky (rock) .. Hold: Switch App (Alt+Tab)",
        "Thumb+Pinky (shaka) . Hold longer: Lock Screen",
        "Ring+Pinky up ....... Move hand up/down: Volume",
        "Thumb+Index up (no pinch) . Move apart/together: Zoom (Ctrl+Scroll)",
    ]

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to read from webcam.")
            break

        img = cv2.flip(img, 1)
        img = detector.findHands(img, draw=True)
        h, w = img.shape[:2]
        lmList, bbox = detector.findPosition(img, draw=False)

        # Progress bar value to show while a hold-gesture is charging up.
        hold_progress = 0.0
        hold_label = ""

        if paused:
            action_label = "PAUSED (press 'p' to resume)"
            scroll_active = False
            dragging = False
            volume_active = False
            zoom_active = False
            new_tab_trigger.update(None)
            explorer_trigger.update(None)
            screenshot_trigger.update(None)
            close_tab_trigger.update(None)
            minimize_trigger.update(None)
            play_pause_trigger.update(None)
            switch_app_trigger.update(None)
            lock_screen_trigger.update(None)
        elif lmList:
            fingers = detector.fingersUp()
            action_label = f"Fingers: {fingers}"

            # --- Move mouse: index finger only ---
            if fingers == [0, 1, 0, 0, 0]:
                x1, y1 = lmList[8][1:]
                x = np.interp(x1, (100, w - 100), (0, screenW))
                y = np.interp(y1, (100, h - 100), (0, screenH))
                clocX = plocX + (x - plocX) / smoothening
                clocY = plocY + (y - plocY) / smoothening
                clocX = clamp(clocX, 0, screenW - 1)
                clocY = clamp(clocY, 0, screenH - 1)
                pyautogui.moveTo(clocX, clocY)
                plocX, plocY = clocX, clocY
                action_label = "Move Mouse"

            # --- Left click / Click-and-drag: thumb + index pinch ---
            # A quick pinch-and-release is a click (old behavior). Holding
            # the pinch and moving your hand now drags instead.
            if len(lmList) >= 9:
                length, img, _ = detector.findDistance(4, 8, img)

                if dragging:
                    # Already dragging: keep moving the mouse with the drag held.
                    x1, y1 = lmList[8][1:]
                    x = np.interp(x1, (100, w - 100), (0, screenW))
                    y = np.interp(y1, (100, h - 100), (0, screenH))
                    clocX = plocX + (x - plocX) / smoothening
                    clocY = plocY + (y - plocY) / smoothening
                    clocX = clamp(clocX, 0, screenW - 1)
                    clocY = clamp(clocY, 0, screenH - 1)
                    pyautogui.moveTo(clocX, clocY)
                    plocX, plocY = clocX, clocY
                    action_label = "Dragging..."

                    if length > drag_release_threshold:
                        pyautogui.mouseUp()
                        dragging = False
                        action_label = "Drag Released"

                elif length < drag_pinch_threshold:
                    if time.time() - last_click_time > click_cooldown:
                        # Not dragging yet: register as a click first...
                        pyautogui.click()
                        action_label = "Left Click"
                        last_click_time = time.time()
                    # ...and if the pinch is still held next frame without
                    # releasing, upgrade it into a drag.
                    elif time.time() - last_click_time > 0.15:
                        pyautogui.mouseDown()
                        dragging = True
                        action_label = "Drag Started"

            # --- Right click: thumb + middle finger pinch ---
            if len(lmList) >= 13:
                lengthR, img, _ = detector.findDistance(4, 12, img)
                if lengthR < 40 and time.time() - last_rclick_time > click_cooldown:
                    pyautogui.click(button='right')
                    action_label = "Right Click"
                    last_rclick_time = time.time()

            # --- Double click: thumb + pinky pinch ---
            if len(lmList) >= 21:
                lengthD, img, _ = detector.findDistance(4, 20, img)
                if lengthD < 40 and time.time() - last_dclick_time > click_cooldown:
                    pyautogui.doubleClick()
                    action_label = "Double Click"
                    last_dclick_time = time.time()

            # --- Scroll: index + middle up, move hand up/down ---
            if fingers == [0, 1, 1, 0, 0]:
                y_index = lmList[8][2]
                if scroll_active:
                    deltaY = prev_scroll_y - y_index
                    if abs(deltaY) > 2:
                        pyautogui.scroll(int(deltaY * scroll_sensitivity))
                else:
                    scroll_active = True
                prev_scroll_y = y_index
                action_label = "Scroll"
            else:
                scroll_active = False

            # --- Open new browser tab: index+middle+ring up, HELD ~0.9s ---
            is_new_tab_pose = (fingers == [0, 1, 1, 1, 0])
            if new_tab_trigger.update("new_tab" if is_new_tab_pose else None):
                pyautogui.hotkey('ctrl', 't')
                action_label = "New Tab"
            elif is_new_tab_pose:
                hold_progress = new_tab_trigger.progress()
                hold_label = "New Tab"
                action_label = f"Hold for New Tab... {int(hold_progress * 100)}%"

            # --- Open File Explorer: thumb only, HELD ~0.9s ---
            is_explorer_pose = (fingers == [1, 0, 0, 0, 0])
            if explorer_trigger.update("explorer" if is_explorer_pose else None):
                pyautogui.hotkey('win', 'e')
                action_label = "Open Explorer"
            elif is_explorer_pose:
                hold_progress = explorer_trigger.progress()
                hold_label = "Open Explorer"
                action_label = f"Hold for Explorer... {int(hold_progress * 100)}%"

            # --- Screenshot: all five fingers up, HELD ~0.9s ---
            is_screenshot_pose = (fingers == [1, 1, 1, 1, 1])
            if screenshot_trigger.update("screenshot" if is_screenshot_pose else None):
                filename = f"gesture_screenshot_{int(time.time())}.png"
                pyautogui.screenshot(filename)
                action_label = f"Screenshot: {filename}"
            elif is_screenshot_pose:
                hold_progress = screenshot_trigger.progress()
                hold_label = "Screenshot"
                action_label = f"Hold for Screenshot... {int(hold_progress * 100)}%"

            # --- Close tab: closed fist, HELD ~1.2s ---
            is_fist_pose = (fingers == [0, 0, 0, 0, 0])
            if close_tab_trigger.update("close_tab" if is_fist_pose else None):
                pyautogui.hotkey('ctrl', 'w')
                action_label = "Close Tab (Fist Held)"
            elif is_fist_pose:
                hold_progress = close_tab_trigger.progress()
                hold_label = "Close Tab"
                action_label = f"Hold to Close Tab... {int(hold_progress * 100)}%"

            # --- Minimize / Show Desktop: pinky only, HELD ~0.9s ---
            is_minimize_pose = (fingers == [0, 0, 0, 0, 1])
            if minimize_trigger.update("minimize" if is_minimize_pose else None):
                pyautogui.hotkey('win', 'd')
                action_label = "Minimize / Show Desktop"
            elif is_minimize_pose:
                hold_progress = minimize_trigger.progress()
                hold_label = "Minimize"
                action_label = f"Hold to Minimize... {int(hold_progress * 100)}%"

            # --- Play/Pause media: ring finger only, HELD ~0.9s ---
            is_play_pause_pose = (fingers == [0, 0, 0, 1, 0])
            if play_pause_trigger.update("play_pause" if is_play_pause_pose else None):
                pyautogui.press('playpause')
                action_label = "Play/Pause Media"
            elif is_play_pause_pose:
                hold_progress = play_pause_trigger.progress()
                hold_label = "Play/Pause"
                action_label = f"Hold for Play/Pause... {int(hold_progress * 100)}%"

            # --- Switch App (Alt+Tab): index+pinky "rock sign", HELD ~0.9s ---
            is_switch_app_pose = (fingers == [0, 1, 0, 0, 1])
            if switch_app_trigger.update("switch_app" if is_switch_app_pose else None):
                pyautogui.hotkey('alt', 'tab')
                action_label = "Switch App (Alt+Tab)"
            elif is_switch_app_pose:
                hold_progress = switch_app_trigger.progress()
                hold_label = "Switch App"
                action_label = f"Hold to Switch App... {int(hold_progress * 100)}%"

            # --- Lock Screen: thumb+pinky "shaka", HELD LONGER (~1.6s) for safety ---
            is_lock_pose = (fingers == [1, 0, 0, 0, 1])
            if lock_screen_trigger.update("lock" if is_lock_pose else None):
                pyautogui.hotkey('win', 'l')
                action_label = "Locking Screen"
            elif is_lock_pose:
                hold_progress = lock_screen_trigger.progress()
                hold_label = "Lock Screen"
                action_label = f"Hold to Lock Screen... {int(hold_progress * 100)}%"

            # --- Volume control: ring+pinky up, move hand up/down ---
            is_volume_pose = (fingers == [0, 0, 0, 1, 1])
            if is_volume_pose:
                y_ref = lmList[8][2]
                if volume_active:
                    deltaY = prev_volume_y - y_ref
                    if abs(deltaY) > volume_step_px:
                        ticks = int(abs(deltaY) / volume_step_px)
                        key = 'volumeup' if deltaY > 0 else 'volumedown'
                        for _ in range(min(ticks, 5)):
                            pyautogui.press(key)
                        prev_volume_y = y_ref
                        action_label = f"Volume {'Up' if deltaY > 0 else 'Down'}"
                else:
                    volume_active = True
                    prev_volume_y = y_ref
                    action_label = "Volume Mode (move hand up/down)"
            else:
                volume_active = False

            # --- Zoom control: thumb+index up but NOT pinched,
            # change the distance between them to zoom in/out (Ctrl+Scroll) ---
            is_zoom_pose = (fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 0
                            and fingers[3] == 0 and fingers[4] == 0)
            if is_zoom_pose and len(lmList) >= 9:
                zoom_dist, img, _ = detector.findDistance(4, 8, img)
                if zoom_dist > drag_release_threshold:  # ignore if it's really a pinch/click
                    if zoom_active and prev_zoom_dist is not None:
                        deltaD = zoom_dist - prev_zoom_dist
                        if abs(deltaD) > zoom_dist_per_tick:
                            ticks = int(abs(deltaD) / zoom_dist_per_tick)
                            pyautogui.keyDown('ctrl')
                            pyautogui.scroll(ticks if deltaD > 0 else -ticks)
                            pyautogui.keyUp('ctrl')
                            prev_zoom_dist = zoom_dist
                            action_label = f"Zoom {'In' if deltaD > 0 else 'Out'}"
                    else:
                        zoom_active = True
                        prev_zoom_dist = zoom_dist
                        action_label = "Zoom Mode (spread/pinch fingers)"
                else:
                    zoom_active = False
            else:
                zoom_active = False

        else:
            action_label = "No hand detected"
            scroll_active = False
            volume_active = False
            zoom_active = False
            if dragging:
                pyautogui.mouseUp()
                dragging = False
            # Release all hold-triggers when the hand disappears.
            new_tab_trigger.update(None)
            explorer_trigger.update(None)
            screenshot_trigger.update(None)
            close_tab_trigger.update(None)
            minimize_trigger.update(None)
            play_pause_trigger.update(None)
            switch_app_trigger.update(None)
            lock_screen_trigger.update(None)

        cTime = time.time()
        fps = 1 / (cTime - pTime) if cTime != pTime else 0
        pTime = cTime

        overlay_color = (0, 0, 200) if paused else (0, 255, 0)
        cv2.rectangle(img, (0, 0), (300, 60), (0, 0, 0), cv2.FILLED)
        cv2.putText(img, f'Action: {action_label}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, overlay_color, 2)
        cv2.putText(img, f'FPS: {int(fps)}   (p=pause h=help)', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, overlay_color, 2)

        # Draw a progress bar while a hold-gesture is charging up, so the
        # user gets visual feedback of "almost triggered".
        if hold_progress > 0.0:
            bar_x, bar_y, bar_w, bar_h = 10, 65, 200, 18
            cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), cv2.FILLED)
            filled_w = int(bar_w * hold_progress)
            cv2.rectangle(img, (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_h), (0, 200, 255), cv2.FILLED)
            cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 1)

        # On-screen gesture legend, toggled with 'h'.
        if show_help:
            panel_h = 24 + 20 * len(help_lines)
            cv2.rectangle(img, (0, h - panel_h), (620, h), (0, 0, 0), cv2.FILLED)
            for i, line in enumerate(help_lines):
                cv2.putText(img, line, (10, h - panel_h + 20 + i * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        try:
            _, _, win_w, win_h = cv2.getWindowImageRect(window_name)
            if win_w > 0 and win_h > 0:
                display_img = cv2.resize(img, (win_w, win_h), interpolation=cv2.INTER_LINEAR)
            else:
                display_img = img
        except Exception:
            display_img = img

        cv2.imshow(window_name, display_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('p'):
            paused = not paused
            if paused and dragging:
                pyautogui.mouseUp()
                dragging = False
        elif key == ord('h'):
            show_help = not show_help

    if dragging:
        pyautogui.mouseUp()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()