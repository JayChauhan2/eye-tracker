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
    refine_landmarks=True,  # Crucial: enables precise iris landmark detection (indices 468-477)
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Open webcam
cap = cv2.VideoCapture(0)

# Calibration bounds (X and Y ratios of iris within the eye bounding box)
# These represent the min/max limits of eye movement. You might need to adjust these values.
X_MIN, X_MAX = 0.38, 0.62
Y_MIN, Y_MAX = 0.38, 0.62

# Smoothing variables (Exponential moving average)
smooth_x, smooth_y = screen_w / 2, screen_h / 2
alpha = 0.15  # Smoothing factor (lower = smoother but slower, higher = faster/noisier)

# Blink detection variables
EAR_THRESHOLD = 0.20  # Eye Aspect Ratio threshold (below this means closed)
blink_state = False   # Tracks if eye was closed in the previous frame
last_blink_time = 0
double_blink_window = 0.6  # Time window (in seconds) to trigger double blink click

def get_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

print("--------------------------------------------------")
print("Eye-Tracking Mouse Control Script Starting...")
print("To STOP the program at any time:")
print("  1. Move the mouse to any corner of your screen (Fail-Safe)")
print("  2. Press 'q' key in the camera preview window")
print("--------------------------------------------------")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break
    
    # Mirror the frame horizontally for intuitive movement
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process facial landmarks
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Helper function to convert landmark coordinates to pixels
        def get_pt(idx):
            pt = landmarks[idx]
            return int(pt.x * w), int(pt.y * h)
            
        # Left Eye Landmark Indices:
        # 33: Outer corner, 133: Inner corner, 159: Top lid, 145: Bottom lid, 468: Iris center
        p_left_out = get_pt(33)
        p_left_in = get_pt(133)
        p_left_top = get_pt(159)
        p_left_bottom = get_pt(145)
        p_left_iris = get_pt(468)
        
        # Right Eye Landmark Indices:
        # 362: Inner corner, 263: Outer corner, 386: Top lid, 374: Bottom lid, 473: Iris center
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
        
        # Check if eyes are closed
        if avg_ear < EAR_THRESHOLD:
            if not blink_state:
                blink_state = True
        else:
            if blink_state:
                # Transited from closed -> open (one complete blink)
                blink_state = False
                time_since_last_blink = current_time - last_blink_time
                
                if time_since_last_blink < double_blink_window:
                    print("Double Blink Detected! Triggering Mouse Click.")
                    pyautogui.click()
                    last_blink_time = 0  # Reset to prevent consecutive multi-clicks
                else:
                    last_blink_time = current_time
                    
        # --- Gaze Mapping ---
        # Calculate horizontal and vertical position of left iris relative to eye socket boundaries
        x_ratio = (p_left_iris[0] - p_left_out[0]) / max(1, (p_left_in[0] - p_left_out[0]))
        y_ratio = (p_left_iris[1] - p_left_top[1]) / max(1, (p_left_bottom[1] - p_left_top[1]))
        
        # Normalize target position to screen space
        x_ratio_norm = (x_ratio - X_MIN) / (X_MAX - X_MIN)
        y_ratio_norm = (y_ratio - Y_MIN) / (Y_MAX - Y_MIN)
        
        # Clamp ratios to [0, 1] range
        x_ratio_norm = max(0.0, min(1.0, x_ratio_norm))
        y_ratio_norm = max(0.0, min(1.0, y_ratio_norm))
        
        # Target screen coordinate
        target_x = x_ratio_norm * screen_w
        target_y = y_ratio_norm * screen_h
        
        # Exponential smoothing (Linear Interpolation) to reduce jitter
        smooth_x = alpha * target_x + (1 - alpha) * smooth_x
        smooth_y = alpha * target_y + (1 - alpha) * smooth_y
        
        # Move cursor if the eye is not closed (otherwise cursor jumps during blink)
        if avg_ear >= EAR_THRESHOLD:
            try:
                pyautogui.moveTo(int(smooth_x), int(smooth_y))
            except pyautogui.FailSafeException:
                print("Fail-safe activated (cursor moved to corner). Program closed.")
                break
                
        # --- Visualization overlay ---
        # Draw circles over irises
        cv2.circle(frame, p_left_iris, 3, (0, 255, 0), -1)
        cv2.circle(frame, p_right_iris, 3, (0, 255, 0), -1)
        
        # Draw eye bounding boxes
        cv2.rectangle(frame, (p_left_out[0], p_left_top[1]), (p_left_in[0], p_left_bottom[1]), (255, 0, 0), 1)
        
        # On-screen Text info
        cv2.putText(frame, f"EAR (Blink): {avg_ear:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Gaze Position: {int(smooth_x)}, {int(smooth_y)}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
    cv2.imshow("Gaze Mouse Control Tracker", frame)
    
    # Check for 'q' key to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
