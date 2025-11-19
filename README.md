# Computer-Vision-Real-time-Posture-Control
A lightweight **computer vision** application that uses **YOLOv8 Pose (PyTorch)** + **OpenCV** to monitor your sitting posture in real time and send desktop notifications when you have been in a bad posture for too long.

> Built with Python, PyTorch, Ultralytics YOLOv8, OpenCV and a bit of ergonomics

---

##  Features

- **Pose estimation** with YOLOv8 Pose (17 keypoints – COCO format)
- **Posture analysis** based on:
  - Torso inclination angle (hips → shoulders vs. vertical)
  - Forward head posture (head vs. shoulder center)
- **Time-based alerts**: a desktop notification is triggered if bad posture is detected continuously for more than a configurable threshold (default: **3 minutes**)
- **Native desktop notifications** (Windows, via `plyer`)
- **Real-time visualization**:
  - Camera feed
  - Keypoints overlay (head, shoulders, hips)
  - Live metrics: torso angle, head offset, posture state

---

## How it works (high level)

1. The webcam feed is captured with **OpenCV**.
2. Each frame is passed to a **YOLOv8 Pose** model from `ultralytics`, which returns 17 body keypoints.
3. The script computes:
   - The **torso angle** as the angle between the vector (hips → shoulders) and the vertical axis.
   - The **head offset** as the horizontal displacement of the head (nose/ears) with respect to the shoulders.
4. If the torso angle exceeds a threshold or the head is too far forward, the posture is classified as **bad**.
5. If bad posture is detected **continuously** for more than `BAD_POSTURE_THRESHOLD_SECONDS`, a desktop notification is shown to remind you to correct your posture.

---

## Tech stack

- **Language**: Python 3.11+
- **Deep Learning / CV**:
  - PyTorch
  - Ultralytics YOLOv8 Pose (`yolov8n-pose.pt`)
  - OpenCV
- **Notifications**:
  - `plyer` (desktop notifications on Windows)
- **Containerization**:
  - Docker (for reproducible environments / experiments)

---

## Installation (local)
Clone the repository:
git clone https://github.com/<tu-usuario>/posture-monitor.git
cd posture-monitor

Create and activate a virtual environment (recommended):
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
.venv\Scripts\activate      # Windows

Install dependencies:
pip install --upgrade pip
pip install -r requirements.txt
If you have issues installing torch, follow the official PyTorch installation guide for your OS / GPU and then reinstall ultralytics / opencv-python if needed.

---

## Usage (local)

From the project root:
python src/posture_monitor.py

A window will open showing:
  - Webcam feed
  - Keypoints (head, shoulders, hips)
  - Torso angle and head offset
  - Bad posture: True/False
- Press q to quit.

By default, a notification is triggered if bad posture is detected for more than 3 minutes.

You can adjust thresholds at the top of src/posture_monitor.py:
BAD_POSTURE_THRESHOLD_SECONDS = 3 * 60
TORSO_ANGLE_THRESHOLD_DEG = 25
HEAD_FORWARD_THRESHOLD_REL = 0.07

---

## Auto-start on Windows (optional)

If you want the posture monitor to run automatically when you log into Windows:

1. Edit scripts/start_posture_monitor_windows.bat if needed:
@echo off
cd /d "%~dp0.."
cd src
python posture_monitor.py

2. Open the Startup folder:
- Press Win + R
- Type: shell:startup
- Press Enter
3. Copy start_posture_monitor_windows.bat into that folder.

  
Next time you log in, the script will start automatically and monitor your posture in the background.

---
