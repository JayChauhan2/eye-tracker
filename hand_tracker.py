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
is_left_dragging = False
prev_scroll_y = None  # Reference height for scrolling displacement

print("--------------------------------------------------")
print("Hand-Tracking Mouse Control Running!")
print("\nControls:")
print("  - Cursor Move: Raise only Index Finger and move it around.")
print("  - Left Click & Drag: Pinch Index Finger and Thumb.")
print("  - Right Click: Pinch Middle Finger and Thumb.")
print("  - Scroll Up/Down: Show Index + Middle (peace sign) and move hand Up/Down.")
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
            
            # Extract thumb tip (4), index tip (8), and middle tip (12)
            thumb = hand_landmarks.landmark[4]
            index_finger = hand_landmarks.landmark[8]
            middle_finger = hand_landmarks.landmark[12]
            
            # 1. Check Finger raised states
            # Y-axis points downwards, so if tip Y is less than joint Y, it's raised
            index_raised = hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y
            middle_raised = hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y
            ring_raised = hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y
            pinky_raised = hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y
            
            # 2. Distance calculation for left click pinch
            left_pinch_dist = np.sqrt((index_finger.x - thumb.x)**2 + (index_finger.y - thumb.y)**2)
            
            # 3. Detect Scroll Gesture (Index & Middle raised, Ring & Pinky folded, and not clicking)
            is_scrolling = index_raised and middle_raised and not ring_raised and not pinky_raised
            
            if is_scrolling and left_pinch_dist > 0.05:
                # Release drag if we transition directly to scroll
                if is_left_dragging:
                    pyautogui.mouseUp()
                    is_left_dragging = False
                
                # Draw scroll lines between index and middle for visual feedback
                idx_px = int(index_finger.x * w), int(index_finger.y * h)
                mid_px = int(middle_finger.x * w), int(middle_finger.y * h)
                cv2.line(frame, idx_px, mid_px, (255, 128, 0), 3)
                
                # Perform Scroll Calculation
                if prev_scroll_y is not None:
                    # dy is change in vertical height of the hand
                    dy = index_finger.y - prev_scroll_y
                    
                    # Scale scroll displacement (Y-axis is inverted: moving hand down -> index.y increases -> scroll down)
                    scroll_amount = int(-dy * 150)
                    if abs(scroll_amount) >= 1:
                        pyautogui.scroll(scroll_amount)
                        # Set current Y as reference for next frame
                        prev_scroll_y = index_finger.y
                else:
                    # Initialize scroll baseline height
                    prev_scroll_y = index_finger.y
                    
                # Display HUD status
                cv2.putText(frame, "TRACKING ACTIVE", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, "State: SCROLLING", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2)
                
            else:
                prev_scroll_y = None  # Reset scroll reference when gesture is broken
                # A. Handle Left Click & Drag (Index + Thumb pinch)
                if left_pinch_dist < 0.05:
                    if not is_left_dragging:
                        pyautogui.mouseDown()
                        is_left_dragging = True
                        print("Index Finger Pinch: Mouse Down (Drag Start)")
                else:
                    if is_left_dragging:
                        pyautogui.mouseUp()
                        is_left_dragging = False
                        print("Pinch Released: Mouse Up (Drag End)")
                
                # B. Move Mouse Cursor (Only if Index finger is extended)
                if index_raised:
                    # Normalize index finger position relative to active box
                    norm_x = (index_finger.x - BOX_X_MIN) / (BOX_X_MAX - BOX_X_MIN)
                    norm_y = (index_finger.y - BOX_Y_MIN) / (BOX_Y_MAX - BOX_Y_MIN)
                    
                    norm_x = max(0.0, min(1.0, norm_x))
                    norm_y = max(0.0, min(1.0, norm_y))
                    
                    # Map to screen pixels
                    target_x = norm_x * screen_w
                    target_y = norm_y * screen_h
                    
                    # Apply exponential smoothing
                    smooth_x = alpha * target_x + (1 - alpha) * smooth_x
                    smooth_y = alpha * target_y + (1 - alpha) * smooth_y
                    
                    try:
                        pyautogui.moveTo(int(smooth_x), int(smooth_y))
                    except pyautogui.FailSafeException:
                        print("Fail-safe activated. Exiting.")
                        break
                
                # Display HUD status
                state_str = "DRAGGING" if is_left_dragging else "MOVING"
                cv2.putText(frame, "TRACKING ACTIVE", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"State: {state_str}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if is_left_dragging else (0, 0, 255), 2)

    cv2.imshow("Hand Gesture Mouse Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
