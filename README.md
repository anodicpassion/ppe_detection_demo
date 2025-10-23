# 👷‍♂️ PPE and Safety Monitoring System
This project implements a real-time Personal Protective Equipment (PPE) and general safety monitoring system using YOLOv8 for object detection, OpenCV for video processing, and Flask-SocketIO for a live, web-based user interface. The system processes video streams (from a file or webcam) and tracks individuals, checking their compliance with safety regulations (hardhats, face masks, and safety vests) and reporting violations.

## 🚀 Features
*Real-Time Detection:* Processes video frames in real-time for live monitoring.
*Robust Object Tracking:* Uses a CentroidTracker with the Hungarian algorithm (scipy.optimize.linear_sum_assignment) for tracking individuals across frames.
*PPE Compliance Check:* Monitors individuals for:
- Hardhats
- Face Masks
- Safety Vests

 *Violation Reporting:* Logs and reports safety violations in real-time, focused on persistent non-compliance.
Web Interface: Provides a live video feed, real-time statistics, and control buttons (Start/Pause/Stop) via a web browser using Flask and SocketIO.
*Video/Webcam Support:* Supports processing uploaded video files or live streams from a webcam.
*Configurable Settings:* Allows adjustment of detection confidence and IoU thresholds.
