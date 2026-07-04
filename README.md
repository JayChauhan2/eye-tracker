# Webcam-Based Mouse Control Systems

This repository contains two computer-vision prototypes for controlling your macOS mouse cursor using a standard built-in camera:

1. **[hand_tracker.py](file:///Users/jaychauhan/.gemini/antigravity-cli/brain/49bb2ba2-03ce-4688-97ba-c5648401ec23/hand_tracker.py)** (Recommended) - Uses hand-pose estimation to map index finger movements to the cursor, and pinch-to-click. Extremely accurate and stable.
2. **[eye_tracker.py](file:///Users/jaychauhan/.gemini/antigravity-cli/brain/49bb2ba2-03ce-4688-97ba-c5648401ec23/eye_tracker.py)** - Uses eye-gaze estimation (iris center relative to fixed eye corners) and double-blink detection to click.

---

## 1. Hand Gesture Controller (`hand_tracker.py`)

This approach maps hand positions to screen space. Because your hand covers hundreds of camera pixels (compared to just ~20 pixels for eyeballs), tracking is **incredibly accurate and stable**.

```mermaid
graph TD
    A[Webcam Frame Input] --> B[MediaPipe Hands]
    B --> C{Hand Detected?}
    C -- Yes --> D[Extract Thumb & Index Tip Landmarks]
    C -- No --> A
    D --> E[Check Pinch Distance between Index & Thumb]
    D --> F[Map Index position inside 'Active Box' to Screen Pixels]
    E --> G{Pinch Distance < 0.05?}
    F --> H[Apply Exponential Smoothing to Move Cursor]
    G -- Yes --> I[Trigger MouseDown / Drag Start]
    G -- No --> J[Trigger MouseUp / Drag End]
```

### Gestures:
* **Cursor Movement**: Move your **Index Finger** inside the blue bounding box displayed on the camera preview. 
* **Pinch-to-Click**: Bring your **Index Finger and Thumb** together (pinch).
* **Drag-and-Drop**: Keep your finger and thumb pinched. Move your hand to drag items, paint, or highlight text. Release the pinch to drop.

---

## 2. Eye Gaze Controller (`eye_tracker.py`)

This approach calculates your gaze by measuring the relative offset of your irises relative to the fixed horizontal axis of your eye corners.

### Key Algorithms:
* **Stable Skeletal Baselines**: Uses fixed eye corners (indices 33, 133, 362, 263) rather than moving eyelids as coordinate reference limits.
* **Dual-Eye Averaging**: Calculates iris position for **both eyes** and averages them, cancelling out high-frequency tracking noise.
* **Adaptive Stabilization Filter**: Detects when your eyes are stationary and freezes the cursor in a small deadzone, while dynamically increasing tracking speed when your eyes scan the screen.
* **Double Blink Detection**: Detects two rapid Eye Aspect Ratio (EAR) drops (eyelid closures) within `0.6 seconds` to trigger a click.

---

## 3. macOS Setup & Running

### Step 1: Install Dependencies
Open your macOS Terminal and run:
```bash
pip install opencv-python mediapipe pyautogui pyobjc-core pyobjc
```

### Step 2: Grant Permissions
1. Open **System Settings > Privacy & Security > Accessibility**.
2. Ensure your **Terminal** application (or IDE running the script) is checked/authorized.
3. Open **System Settings > Privacy & Security > Camera** and ensure your Terminal is allowed to access the webcam.

### Step 3: Run the Scripts
Navigate to the directory:
```bash
cd /Users/jaychauhan/.gemini/antigravity-cli/brain/49bb2ba2-03ce-4688-97ba-c5648401ec23
```

* **To run Hand Tracking (Recommended)**:
  ```bash
  python3 hand_tracker.py
  ```

* **To run Eye Tracking**:
  ```bash
  python3 eye_tracker.py
  ```
