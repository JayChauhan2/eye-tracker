import warnings
warnings.simplefilter("ignore", category=UserWarning)

import cv2
import mediapipe as mp
import pyautogui
import numpy as np

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
# This prevents you from having to move your hand to the extreme edges of the camera view.
# We define a box in the center of the frame.
BOX_X_MIN, BOX_X_MAX = 0.25, 0.75
BOX_Y_MIN, BOX_Y_MAX = 0.25, 0.75

# Cursor smoothing
smooth_x, smooth_y = screen_w / 2, screen_h / 2
alpha = 0.25  # Smoothing factor (higher = faster, lower = smoother)

# Click / Drag state
is_dragging = False

print("--------------------------------------------------")
print("Hand-Tracking Mouse Control Running!")
print("\nControls:")
print("  - Move your Index Finger to control the cursor.")
print("  - Pinch your Index Finger and Thumb together to Click & Drag.")
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
    cv2.putText(frame, "Active Area", (bx_min, by_min - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract thumb tip (4) and index tip (8)
            thumb = hand_landmarks.landmark[4]
            index_finger = hand_landmarks.landmark[8]
            
            # Convert index finger tip location to normalized active box space
            # Clamp the position to the boundaries of our active box
            norm_x = (index_finger.x - BOX_X_MIN) / (BOX_X_MAX - BOX_X_MIN)
            norm_y = (index_finger.y - BOX_Y_MIN) / (BOX_Y_MAX - BOX_Y_MIN)
            
            norm_x = max(0.0, min(1.0, norm_x))
            norm_y = max(0.0, min(1.0, norm_y))
            
            # Map to screen pixels
            target_x = norm_x * screen_w
            target_y = norm_y * screen_h
            
            # Apply smoothing
            smooth_x = alpha * target_x + (1 - alpha) * smooth_x
            smooth_y = alpha * target_y + (1 - alpha) * smooth_y
            
            # Check pinch distance (normalized Euclidean distance)
            pinch_dist = np.sqrt((index_finger.x - thumb.x)**2 + (index_finger.y - thumb.y)**2)
            
            # Visual feedback on pinch
            thumb_px = int(thumb.x * w), int(thumb.y * h)
            index_px = int(index_finger.x * w), int(index_finger.y * h)
            cv2.line(frame, thumb_px, index_px, (0, 255, 0) if pinch_dist < 0.05 else (0, 0, 255), 2)
            
            # Handle Click & Drag states
            if pinch_dist < 0.05:
                if not is_dragging:
                    pyautogui.mouseDown()
                    is_dragging = True
                    print("Pinch Detected: Mouse Down (Drag Start)")
            else:
                if is_dragging:
                    pyautogui.mouseUp()
                    is_dragging = False
                    print("Pinch Released: Mouse Up (Drag End)")
            
            # Move mouse
            try:
                pyautogui.moveTo(int(smooth_x), int(smooth_y))
            except pyautogui.FailSafeException:
                print("Fail-safe activated. Exiting.")
                break
                
            # Draw cursor position trace
            cv2.putText(frame, "TRACKING HAND", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"State: {'DRAGGING' if is_dragging else 'MOVING'}", (20, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if is_dragging else (0, 0, 255), 2)
            
    cv2.imshow("Hand Gesture Mouse Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
