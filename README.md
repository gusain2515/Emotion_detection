# Emotion_detection
This code implements a multi-camera real-time emotion detection and tracking system using YOLO for person and face detection, DeepFace for emotion analysis, and DeepSORT for tracking individuals across frames.

# Key Features:
# Multi-Camera Support:
Processes multiple RTSP streams simultaneously.
# Real-Time Display:
Displays frames from different cameras in separate windows.
# Person & Face Detection:
Uses YOLOv8 for detecting people and YOLOv8-face for face detection.
# Emotion Analysis:
DeepFace predicts emotions (Happy, Sad, Neutral, Angry) for detected faces.
# Tracking & Filtering:
Uses an emotion confidence threshold and tracks emotions over multiple frames for improved accuracy.
# CSV Logging:
Periodically records the number of people and their emotions into a CSV file.
# Threaded Processing:
Uses multithreading for efficient performance, ensuring smooth real-time processing.
