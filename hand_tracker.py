import warnings
warnings.simplefilter("ignore", category=UserWarning)

import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Screen size setup
screen_w, screen_h = pyautogui.size()
pyautogui.FAILSAFE = True

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Open camera (using robust multi-index macOS fallback)
cap = None
for index in [0, 1, 2, 4]:
    print(f"Trying to open camera index {index}...")
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"Successfully opened camera index {index}!")
            break
        else:
            cap.release()
            cap = None
    else:
        cap = None

if cap is None:
    print("[ERROR] Could not open webcam.")
    exit(1)

# Active interaction area in the camera frame (normalized coords 0.0 - 1.0)
# We define a box in the center of the frame.
BOX_X_MIN, BOX_X_MAX = 0.25, 0.75
BOX_Y_MIN, BOX_Y_MAX = 0.25, 0.75

# Cursor smoothing
smooth_x, smooth_y = screen_w / 2, screen_h / 2
alpha = 0.25  # Smoothing factor

# Gestures & Actions States
is_dragging = False
hand_x_history = []  # Tracks middle-finger MCP X coordinate and timestamps
swipe_cooldown = 0   # Prevents multiple triggers from a single swipe
flash_text = ""      # Temporary feedback notification text
flash_timer = 0      # Frames to show feedback notification

print("--------------------------------------------------")
print("Hand-Tracking Mouse Control & Swipe Running!")
print("\nControls:")
print("  - Cursor Move: Raise only Index Finger and move it around.")
print("  - Click & Drag: Pinch Index Finger and Thumb together.")
print("  - Swipe Desktops: Open flat palm (all fingers raised) and wave Left or Right.")
print("  - Move mouse to any corner to abort (Fail-safe).")
print("  - Press 'q' in the window to quit.")
print("--------------------------------------------------")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640, 480))
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process hand landmarks
    results = hands.process(rgb_frame)
    
    # Draw active interaction box boundaries on camera feed
    bx_min, bx_max = int(BOX_X_MIN * w), int(BOX_X_MAX * w)
    by_min, by_max = int(BOX_Y_MIN * h), int(BOX_Y_MAX * h)
    cv2.rectangle(frame, (bx_min, by_min), (bx_max, by_max), (255, 0, 0), 2)
    cv2.putText(frame, "Cursor Active Area", (bx_min, by_min - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract thumb tip (4) and index tip (8)
            thumb = hand_landmarks.landmark[4]
            index_finger = hand_landmarks.landmark[8]
            
            # 1. Count Raised Fingers to determine state (Cursor vs Swipe)
            raised_count = 0
            # Check four fingers (Index, Middle, Ring, Pinky)
            # Y-axis points downwards, so if tip Y is less than joint Y, it's raised
            for tip, joint in [(8, 6), (12, 10), (16, 14), (20, 18)]:
                if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[joint].y:
                    raised_count += 1
            # Check Thumb (if thumb tip is horizontally far from index knuckle)
            if abs(hand_landmarks.landmark[4].x - hand_landmarks.landmark[5].x) > 0.04:
                raised_count += 1
                
            # 2. Track hand velocity (using Middle Knuckle - landmark 9)
            current_time = time.time()
            hand_center_x = hand_landmarks.landmark[9].x
            hand_x_history.append((current_time, hand_center_x))
            
            # Retain history from the last 0.2 seconds
            hand_x_history = [(t, x) for t, x in hand_x_history if current_time - t < 0.20]
            
            # 3. Detect Swipes (Only if palm is open: >= 4 fingers raised)
            if raised_count >= 4 and (current_time - swipe_cooldown > 1.2):
                if len(hand_x_history) >= 3:
                    # Difference between earliest and latest X position in window
                    dx = hand_x_history[-1][1] - hand_x_history[0][1]
                    
                    # Swipe Left (hand moved quickly to the left)
                    if dx < -0.15:
                        print("Swipe Left: Switching Desktop Right")
                        pyautogui.hotkey('ctrl', 'right')
                        flash_text = "SWIPE LEFT (DESKTOP RIGHT)"
                        flash_timer = 25
                        swipe_cooldown = current_time
                        hand_x_history.clear()
                    # Swipe Right (hand moved quickly to the right)
                    elif dx > 0.15:
                        print("Swipe Right: Switching Desktop Left")
                        pyautogui.hotkey('ctrl', 'left')
                        flash_text = "SWIPE RIGHT (DESKTOP LEFT)"
                        flash_timer = 25
                        swipe_cooldown = current_time
                        hand_x_history.clear()

            # 4. Move Cursor (Only if palm is NOT open, to freeze cursor during swipe)
            if raised_count < 4:
                # Convert index finger tip location to normalized active box space
                norm_x = (index_finger.x - BOX_X_MIN) / (BOX_X_MAX - BOX_X_MIN)
                norm_y = (index_finger.y - BOX_Y_MIN) / (BOX_Y_MAX - BOX_Y_MIN)
                
                # Clamp to screen space limits
                norm_x = max(0.0, min(1.0, norm_x))
                norm_y = max(0.0, min(1.0, norm_y))
                
                # Map to screen pixels
                target_x = norm_x * screen_w
                target_y = norm_y * screen_h
                
                # Apply smoothing
                smooth_x = alpha * target_x + (1 - alpha) * smooth_x
                smooth_y = alpha * target_y + (1 - alpha) * smooth_y
                
                # Check pinch distance for Click/Drag
                pinch_dist = np.sqrt((index_finger.x - thumb.x)**2 + (index_finger.y - thumb.y)**2)
                
                # Visual link between thumb and index
                thumb_px = int(thumb.x * w), int(thumb.y * h)
                index_px = int(index_finger.x * w), int(index_finger.y * h)
                cv2.line(frame, thumb_px, index_px, (0, 255, 0) if pinch_dist < 0.05 else (0, 0, 255), 2)
                
                # Click states
                if pinch_dist < 0.05:
                    if not is_dragging:
                        pyautogui.mouseDown()
                        is_dragging = True
                        print("Pinch: Mouse Down")
                else:
                    if is_dragging:
                        pyautogui.mouseUp()
                        is_dragging = False
                        print("Pinch Released: Mouse Up")
                
                # Move mouse
                try:
                    pyautogui.moveTo(int(smooth_x), int(smooth_y))
                except pyautogui.FailSafeException:
                    print("Fail-safe activated. Exiting.")
                    break

            # 5. Display visual status metrics on preview window
            cv2.putText(frame, "TRACKING ACTIVE", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Fingers Raised: {raised_count}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            cv2.putText(frame, f"State: {'DRAGGING' if is_dragging else ('GESTURE LOCK' if raised_count >= 4 else 'MOVING')}", 
                        (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if is_dragging else (255, 128, 0), 2)

    # Render swipe flash notifications
    if flash_timer > 0:
        cv2.putText(frame, flash_text, (50, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
        flash_timer -= 1

    cv2.imshow("Hand Gesture Mouse Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
