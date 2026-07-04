# Eye-Tracking & Cursor-Control via Webcam

Yes! It is absolutely feasible to build software that tracks where you look on the screen using your Mac's camera and executes a click on a double blink. In fact, you can implement a prototype in python with **less than 150 lines of code** by combining machine learning frameworks and macOS interface tools.

A starter prototype script has been created for you at [eye_tracker.py](file:///Users/jaychauhan/.gemini/antigravity-cli/brain/49bb2ba2-03ce-4688-97ba-c5648401ec23/scratch/eye_tracker.py).

---

## 1. System Architecture

The software operates as a high-frequency loop matching camera frames to screen actions:

```mermaid
graph TD
    A[Webcam Frame Input] --> B[MediaPipe FaceMesh]
    B --> C{Landmarks Detected?}
    C -- Yes --> D[Extract Iris & Eyelid Coordinates]
    C -- No --> A
    D --> E[Compute Eye Aspect Ratio - EAR]
    D --> F[Calculate Iris Position Relative to Eye Sockets]
    E --> G{Double Blink Detected?}
    F --> H[Apply Exponential Smoothing & Map to Screen Pixels]
    G -- Yes --> I[Trigger PyAutoGUI Mouse Click]
    H --> J[Update PyAutoGUI Cursor Position]
```

### Core Technologies
*   **MediaPipe Face Mesh:** A lightweight machine learning model by Google that estimates 468+ 3D facial landmarks in real-time. By utilizing `refine_landmarks=True`, we unlock 10 additional landmarks specifically tracking the boundaries and centers of the **irises**.
*   **OpenCV:** Responsible for opening the camera, fetching frames, mirroring/flipping the image, and rendering a debugging display overlay.
*   **PyAutoGUI:** Handles programmatic control of the mouse pointer and click execution on the macOS operating system.

---

## 2. Key Algorithms

### Gaze Estimation (Relative Mapping)
Because standard webcams are off-axis (mounted at the top of screens) and have varying user distances, we estimate gaze by measuring the relative position of the iris inside the eye socket:

$$\text{Ratio}_X = \frac{\text{Iris}_x - \text{OuterCorner}_x}{\text{InnerCorner}_x - \text{OuterCorner}_x}$$

*   When you look **left**, the iris moves towards the outer corner.
*   When you look **right**, it moves towards the inner corner.
*   We calibrate these values by bounding them between a minimum threshold (e.g., `0.38`) and maximum threshold (e.g., `0.62`), and scaling the result to screen width and height.

### Noise Reduction & Jitter Smoothing
Raw eye tracking coordinates bounce constantly due to microsaccades (tiny involuntary eye movements) and camera noise. To solve this, the script uses **Exponential Smoothing**:

$$\text{Smooth}_{t} = \alpha \cdot \text{Target}_{t} + (1 - \alpha) \cdot \text{Smooth}_{t-1}$$

*   Setting $\alpha \approx 0.15$ ensures mouse movements look fluid instead of violently shaking.

### Eyelid Aspect Ratio (EAR) & Double Blink
To distinguish regular blinks from double-blink clicks, the software calculates the Eye Aspect Ratio (EAR):

$$\text{EAR} = \frac{||\text{TopLid} - \text{BottomLid}||}{||\text{LeftCorner} - \text{RightCorner}||}$$

```
     TopLid (159)
       \ | /
  Left ------- Right (133)
  (33) / | \
    BottomLid (145)
```

1.  **Closed Eye:** The vertical distance shrinks to near-zero, dropping the EAR below `0.20`.
2.  **Open Eye:** EAR rebounds to `0.30 - 0.40`.
3.  **Double Blink:** If two transitions from closed to open occur within `0.6 seconds` (the double blink window), a mouse click is triggered.

---

## 3. macOS Setup & Run Instructions

To test this on your Mac, perform the following steps:

### Step 1: Install Dependencies
Open your terminal and install the required Python packages:
```bash
pip install opencv-python mediapipe pyautogui pyobjc-core pyobjc
```
> [!NOTE]
> `pyobjc` is highly recommended on macOS as it allows `pyautogui` to interact natively with Apple's Quartz and Cocoa APIs.

### Step 2: Grant Accessibility Permissions
macOS security prevents scripts from controlling your mouse unless explicitly authorized.
1.  Open **System Settings** > **Privacy & Security** > **Accessibility**.
2.  Add and check your **Terminal** application (or IDE like VS Code / Cursor if you run scripts from inside it).
3.  *If you run this directly via python in the shell, authorize your terminal emulator (e.g., Terminal.app or iTerm.app).*

### Step 3: Run the Script
Execute the script from your terminal:
```bash
python3 /Users/jaychauhan/.gemini/antigravity-cli/brain/49bb2ba2-03ce-4688-97ba-c5648401ec23/scratch/eye_tracker.py
```

---

## 4. Production Challenges & Enhancements
If you want to build a fully functional application, consider these improvements:
*   **Calibration UI:** Build a setup step showing points at the 4 corners of the screen. Asking the user to look at each point records their exact minimum and maximum iris ratio limits, dramatically increasing accuracy.
*   **Head Pose Correction:** Iris-tracking alone fails if you move or tilt your head. Advanced tools estimate the head's rotation vector and offset the gaze estimation to compensate.
*   **Operating System Native Service:** Run the tracking inside a lightweight C++/Swift daemon utilizing Apple's **Vision Framework** for optimal battery life and background performance.
