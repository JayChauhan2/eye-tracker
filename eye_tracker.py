import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np

# Screen setup
screen_w, screen_h = pyautogui.size()
# Fail-safe: moving mouse to any corner of the screen aborts the script
pyautogui.FAILSAFE = True

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,  # Crucial: enables precise iris landmark detection
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Open webcam
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
    print("\n[ERROR] Could not open any webcam.")
    print("Please check that your camera is not in use and that permissions are granted.")
    exit(1)

# Calibration state
# 0: Calibration not done - prompt Top-Left
# 1: Calibration not done - prompt Bottom-Right
# 2: Calibration finished and active
calibration_step = 0
X_MIN, X_MAX = 0.38, 0.62  # Default fallbacks
Y_MIN, Y_MAX = 0.38, 0.62
calib_tl_x, calib_tl_y = 0.0, 0.0
calib_br_x, calib_br_y = 0.0, 0.0

# Smoothing variables (Exponential moving average)
smooth_x, smooth_y = screen_w / 2, screen_h / 2
alpha = 0.30  # Default smoothing factor (higher = faster response/more jitter, lower = slower/smoother)

# Blink detection variables
EAR_THRESHOLD = 0.20  # Eye Aspect Ratio threshold
blink_state = False   # Tracks if eye was closed
last_blink_time = 0
double_blink_window = 0.6  # Time window for double blink click

def get_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

print("--------------------------------------------------")
print("Calibration Steps:")
print("  1. Look at the TOP-LEFT corner of your screen and press SPACE.")
print("  2. Look at the BOTTOM-RIGHT corner of your screen and press SPACE.")
print("\nControls:")
print("  - Press 'c' to restart calibration at any time.")
print("  - Press ']' to increase speed (reduce latency).")
print("  - Press '[' to decrease speed (increase smoothing).")
print("  - Move mouse to any corner to abort (Fail-safe).")
print("  - Press 'q' in the window to quit.")
print("--------------------------------------------------")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break
    
    # Mirror the frame horizontally for intuitive movement
    frame = cv2.flip(frame, 1)
    
    # DOWNSCALE frame to 640x480 for massive speedup/lag reduction!
    frame = cv2.resize(frame, (640, 480))
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process facial landmarks
    results = face_mesh.process(rgb_frame)
    
    # Check keyboard inputs
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord('c'):
        calibration_step = 0
        print("Recalibration started. Please look at the TOP-LEFT corner and press SPACE.")
    elif key == ord(']'):
        alpha = min(1.0, alpha + 0.05)
        print(f"Tracking speed increased. Alpha: {alpha:.2f}")
    elif key == ord('['):
        alpha = max(0.05, alpha - 0.05)
        print(f"Tracking speed decreased (more smooth). Alpha: {alpha:.2f}")
        
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Convert landmark coordinates to pixels
        def get_pt(idx):
            pt = landmarks[idx]
            return int(pt.x * w), int(pt.y * h)
            
        # Left Eye: 33: Outer, 133: Inner, 159: Top, 145: Bottom, 468: Iris center
        p_left_out = get_pt(33)
        p_left_in = get_pt(133)
        p_left_top = get_pt(159)
        p_left_bottom = get_pt(145)
        p_left_iris = get_pt(468)
        
        # Right Eye: 362: Inner, 263: Outer, 386: Top, 374: Bottom, 473: Iris center
        p_right_in = get_pt(362)
        p_right_out = get_pt(263)
        p_right_top = get_pt(386)
        p_right_bottom = get_pt(374)
        p_right_iris = get_pt(473)
        
        # --- Blink Detection (Eye Aspect Ratio - EAR) ---
        ear_left = get_distance(p_left_top, p_left_bottom) / max(1, get_distance(p_left_out, p_left_in))
        ear_right = get_distance(p_right_top, p_right_bottom) / max(1, get_distance(p_right_out, p_right_in))
        avg_ear = (ear_left + ear_right) / 2.0
        
        current_time = time.time()
        
        # Handle blink state
        if avg_ear < EAR_THRESHOLD:
            if not blink_state:
                blink_state = True
        else:
            if blink_state:
                blink_state = False
                time_since_last_blink = current_time - last_blink_time
                if time_since_last_blink < double_blink_window:
                    if calibration_step == 2:  # Only click if calibrated
                        print("Double Blink Detected! Triggering Mouse Click.")
                        pyautogui.click()
                    last_blink_time = 0
                else:
                    last_blink_time = current_time
                    
        # --- Gaze Position Calculations ---
        x_ratio = (p_left_iris[0] - p_left_out[0]) / max(1, (p_left_in[0] - p_left_out[0]))
        y_ratio = (p_left_iris[1] - p_left_top[1]) / max(1, (p_left_bottom[1] - p_left_top[1]))
        
        # Calibration state machine
        if calibration_step == 0:
            # Calibrating Top-Left
            cv2.putText(frame, "CALIBRATION: Look at TOP-LEFT corner", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "and press SPACE key.", (20, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # Draw visual helper circle at top-left of the camera view
            cv2.circle(frame, (30, 30), 15, (0, 0, 255), -1)
            
            if key == 32:  # Space bar
                calib_tl_x, calib_tl_y = x_ratio, y_ratio
                calibration_step = 1
                print("Top-Left saved. Now look at BOTTOM-RIGHT and press SPACE.")
                time.sleep(0.3)  # Debounce keypress
                
        elif calibration_step == 1:
            # Calibrating Bottom-Right
            cv2.putText(frame, "CALIBRATION: Look at BOTTOM-RIGHT corner", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "and press SPACE key.", (20, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # Draw visual helper circle at bottom-right of the camera view
            cv2.circle(frame, (w - 30, h - 30), 15, (0, 0, 255), -1)
            
            if key == 32:  # Space bar
                calib_br_x, calib_br_y = x_ratio, y_ratio
                X_MIN = min(calib_tl_x, calib_br_x)
                X_MAX = max(calib_tl_x, calib_br_x)
                Y_MIN = min(calib_tl_y, calib_br_y)
                Y_MAX = max(calib_tl_y, calib_br_y)
                
                # Prevent division by zero
                if X_MIN == X_MAX: X_MAX += 0.01
                if Y_MIN == Y_MAX: Y_MAX += 0.01
                
                calibration_step = 2
                print(f"Calibration Complete! Bounds X:[{X_MIN:.3f}, {X_MAX:.3f}] Y:[{Y_MIN:.3f}, {Y_MAX:.3f}]")
                time.sleep(0.3)
                
        elif calibration_step == 2:
            # Active tracking mode
            # Normalize coordinates using calibrated screen boundary ratios
            x_ratio_norm = (x_ratio - X_MIN) / (X_MAX - X_MIN)
            y_ratio_norm = (y_ratio - Y_MIN) / (Y_MAX - Y_MIN)
            
            # Clamp ratios to [0.0, 1.0]
            x_ratio_norm = max(0.0, min(1.0, x_ratio_norm))
            y_ratio_norm = max(0.0, min(1.0, y_ratio_norm))
            
            # Target screen position
            target_x = x_ratio_norm * screen_w
            target_y = y_ratio_norm * screen_h
            
            # Apply exponential smoothing
            smooth_x = alpha * target_x + (1 - alpha) * smooth_x
            smooth_y = alpha * target_y + (1 - alpha) * smooth_y
            
            # Move mouse if eyes are open
            if avg_ear >= EAR_THRESHOLD:
                try:
                    pyautogui.moveTo(int(smooth_x), int(smooth_y))
                except pyautogui.FailSafeException:
                    print("Fail-safe activated (cursor moved to corner). Program closed.")
                    break
                    
            # Visual feedback on camera preview
            cv2.circle(frame, p_left_iris, 3, (0, 255, 0), -1)
            cv2.circle(frame, p_right_iris, 3, (0, 255, 0), -1)
            cv2.rectangle(frame, (p_left_out[0], p_left_top[1]), (p_left_in[0], p_left_bottom[1]), (0, 255, 0), 1)
            
            cv2.putText(frame, "ACTIVE TRACKING MODE", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            cv2.putText(frame, f"Cursor: {int(smooth_x)}, {int(smooth_y)}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            cv2.putText(frame, f"Speed [Alpha]: {alpha:.2f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    cv2.imshow("Gaze Mouse Control Tracker", frame)

cap.release()
cv2.destroyAllWindows()
