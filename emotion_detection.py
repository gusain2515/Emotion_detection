import os
import cv2
import torch
import numpy as np
import csv
from ultralytics import YOLO
from deepface import DeepFace
from collections import defaultdict, deque
import logging
import time
import threading
import queue
from datetime import datetime
import argparse

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console logging
        # Uncomment the following line to enable file logging
        # logging.FileHandler("attendance_system.log"),
    ]
)

def display_frames(display_queue):
    """
    Display frames from multiple camera streams.
    """
    windows = {}
    while True:
        try:
            camera_id, frame = display_queue.get(timeout=1)
            window_name = f"Camera {camera_id}"

            if frame is None or frame.size == 0:
                logging.warning(f"Received empty frame for camera {camera_id}. Skipping display.")
                continue

            if window_name not in windows:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, 640, 480)
                windows[window_name] = True

            cv2.imshow(window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info(f"Exit key pressed for {window_name}. Shutting down.")
                break
        except queue.Empty:
            # No frame received, continue
            pass

    for window in windows:
        cv2.destroyWindow(window)

class SharedCSVHandler:
    """
    Handles writing emotion data from multiple cameras to a single CSV file.
    """
    def __init__(self, csv_path, camera_ids, target_emotions):
        self.csv_path = csv_path
        self.lock = threading.Lock()
        self.target_emotions = target_emotions
        self.camera_ids = camera_ids
        self.data_lock = threading.Lock()
        self.current_data = defaultdict(lambda: {'Number of People': 0, 'Happy': 0, 'Sad': 0, 'Neutral': 0, 'Angry': 0})
        self.csv_update_interval = 60  # seconds
        self.last_csv_update = time.time()

        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)

        # Initialize CSV with headers
        with self.lock:
            with open(self.csv_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                header = ['Timestamp', 'Source', 'Number of People', 'Happy', 'Sad', 'Neutral', 'Angry']
                writer.writerow(header)

    def record_emotions(self, camera_id, num_people, emotion_counts):
        """
        Records the latest emotion counts for a specific camera.
        """
        with self.data_lock:
            self.current_data[camera_id]['Number of People'] = num_people
            for emotion in self.target_emotions:
                # Capitalize the first letter to match CSV headers
                self.current_data[camera_id][emotion.capitalize()] = emotion_counts.get(emotion, 0)

    def periodic_update(self):
        """
        Periodically writes the collected data to the CSV file.
        """
        while True:
            time.sleep(1)
            current_time = time.time()
            if current_time - self.last_csv_update >= self.csv_update_interval:
                self.write_to_csv()
                self.last_csv_update = current_time

    def write_to_csv(self):
        """
        Writes the current data to the CSV and resets the data.
        """
        with self.data_lock:
            data_copy = dict(self.current_data)
            # Reset the data after copying
            self.current_data = defaultdict(lambda: {'Number of People': 0, 'Happy': 0, 'Sad': 0, 'Neutral': 0, 'Angry': 0})

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        rows = []
        for camera_id in self.camera_ids:
            camera_data = data_copy.get(camera_id, {
                'Number of People': 0,
                'Happy': 0,
                'Sad': 0,
                'Neutral': 0,
                'Angry': 0
            })
            row = [
                timestamp,
                camera_id,
                camera_data['Number of People'],
                camera_data['Happy'],
                camera_data['Sad'],
                camera_data['Neutral'],
                camera_data['Angry']
            ]
            rows.append(row)

        with self.lock:
            with open(self.csv_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(rows)
        logging.info(f"CSV updated at {timestamp}")

class ObjectTracking:
    """
    Handles object detection, face detection, emotion analysis, and data recording for a single camera stream.
    """
    def __init__(self, stream_url, display_queue, csv_handler, camera_id, headless_mode=False):
        self.stream_url = stream_url
        self.camera_id = camera_id
        self.display_queue = display_queue  # Shared display queue
        self.csv_handler = csv_handler
        self.headless_mode = headless_mode  # New attribute

        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Path to YOLOv8 model weights
        weights_path = r'C:\Users\KRYP MEDIA\Downloads\rkface\FInal_Codes\yolov8m.pt'

        # Ensure that torch uses CUDA if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device {self.device} for camera {self.camera_id}")

        # Initialize YOLO model for object detection
        self.model = YOLO(weights_path).to(self.device)
        self.model.fuse()

        try:
            # Path to YOLO model weights for face detection
            face_weights_path = r'C:\Users\KRYP MEDIA\Downloads\rkface\yolov8n-face.pt'
            self.face_model = YOLO(face_weights_path).to(self.device)
            logging.info(f"Face model initialized on {self.device} for camera {self.camera_id}")
        except Exception as e:
            logging.error(f"Error initializing face model for camera {self.camera_id}: {e}")
            raise

        # Emotion tracking parameters
        self.emotion_window_size = 5
        self.emotion_confidence_threshold = 0.80
        self.detected_persons = {}
        self.target_emotions = ['happy', 'sad', 'neutral', 'angry']

        # Flag to control the thread
        self.running = True

    def stop(self):
        """
        Stops the processing thread.
        """
        self.running = False

    def detect_emotion(self, face_crop, id):
        """
        Detects and tracks emotions for a given face crop.
        """
        try:
            # DeepFace does not use torch.cuda, so we leave it as is
            analysis = DeepFace.analyze(
                face_crop,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )

            if isinstance(analysis, list):
                analysis = analysis[0]

            emotions = {emotion: score for emotion, score in analysis['emotion'].items() if emotion in self.target_emotions}
            total = sum(emotions.values())
            if total > 0:
                emotions = {emotion: score / total for emotion, score in emotions.items()}

            if id not in self.detected_persons:
                self.detected_persons[id] = {
                    'emotion_history': deque(maxlen=self.emotion_window_size),
                    'current_emotion': None,
                    'confidence': 0.4
                }

            person = self.detected_persons[id]
            person['emotion_history'].append(emotions)

            averaged_emotions = defaultdict(float)
            for past_emotions in person['emotion_history']:
                for emotion, score in past_emotions.items():
                    averaged_emotions[emotion] += score

            num_frames = len(person['emotion_history'])
            if num_frames > 0:
                averaged_emotions = {emotion: score / num_frames for emotion, score in averaged_emotions.items()}

            high_confidence_emotions = {
                emotion: score 
                for emotion, score in averaged_emotions.items() 
                if score >= self.emotion_confidence_threshold
            }

            if high_confidence_emotions:
                current_emotion = max(high_confidence_emotions.items(), key=lambda x: x[1])
                person['current_emotion'] = current_emotion[0]
                person['confidence'] = current_emotion[1]

        except Exception as e:
            logging.error(f"Error detecting emotion for person {id}: {e}")

    def process_frame(self, frame):
        """
        Processes a single frame: detects persons, detects faces, analyzes emotions, and annotates the frame.
        """
        if frame is None or frame.size == 0:
            logging.warning(f"Received empty frame for Camera {self.camera_id}. Skipping processing.")
            return None

        display_frame = frame.copy()

        with torch.no_grad():
            try:
                # Detect persons in the frame
                results = self.model.track(frame, persist=True, classes=[0], device=self.device)

                if results and len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes.xyxy  # Bounding boxes
                    ids = results[0].boxes.id      # Object IDs

                    # Move boxes and ids to CPU for OpenCV operations
                    boxes_cpu = boxes.cpu().numpy().astype(int)
                    if ids is not None:
                        ids_cpu = ids.cpu().numpy().astype(int)
                    else:
                        ids_cpu = []

                    for box, id in zip(boxes_cpu, ids_cpu):
                        x1, y1, x2, y2 = box
                        person_crop = frame[y1:y2, x1:x2]

                        # Run face detection on person_crop using the face_model
                        face_results = self.face_model(person_crop, device=self.device)

                        if face_results and len(face_results[0].boxes) > 0:
                            face_box = face_results[0].boxes[0].xyxy.cpu().numpy().astype(int)
                            # face_box is a 2D array; get the first face
                            if len(face_box) > 0:
                                fx1, fy1, fx2, fy2 = face_box[0]
                                face_crop = person_crop[fy1:fy2, fx1:fx2]

                                if face_crop.shape[0] > 0 and face_crop.shape[1] > 0:
                                    # DeepFace does not use torch.cuda, so we can pass face_crop directly
                                    self.detect_emotion(face_crop, id)

                                    person_data = self.detected_persons[id]
                                    emotion = person_data.get('current_emotion', 'Unknown')
                                    color = (0, 255, 0) if emotion == 'happy' else (255, 0, 0)  # Green for happy, Blue otherwise

                                    # Draw bounding box and emotion label if not headless
                                    if not self.headless_mode:
                                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                                        cv2.putText(display_frame, f"{emotion}", (x1, y1 - 10),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            except Exception as e:
                logging.error(f"Error processing frame for camera {self.camera_id}: {e}")

        # Collect current emotion data
        num_people = len(self.detected_persons)
        emotion_counts = {emotion: 0 for emotion in self.target_emotions}

        for person in self.detected_persons.values():
            if person['current_emotion'] in emotion_counts:
                emotion_counts[person['current_emotion']] += 1

        # Update the shared CSV handler
        self.csv_handler.record_emotions(self.camera_id, num_people, emotion_counts)

        return display_frame

    def process_stream(self):
        """
        Continuously captures frames from the video stream and processes them.
        """
        cap = cv2.VideoCapture(self.stream_url)
        if not cap.isOpened():
            logging.error(f"Failed to open camera stream: {self.stream_url}")
            return

        while self.running:
            ret, frame = cap.read()
            if not ret:
                logging.warning(f"Failed to receive frame from {self.stream_url}. Retrying...")
                time.sleep(1)
                continue

            processed_frame = self.process_frame(frame)

            if not self.headless_mode and processed_frame is not None:
                self.display_queue.put((self.camera_id, processed_frame))
        cap.release()
        logging.info(f"Stream processing stopped for camera {self.camera_id}")

def main():
    """
    Main function to initialize and start all components.
    """
    # Argument Parsing
    parser = argparse.ArgumentParser(description="Face Recognition Attendance System")
    parser.add_argument(
        '--headless',
        action='store_true',
        help='Run the script in headless mode without GUI display.'
    )
    args = parser.parse_args()
    headless_mode = args.headless
    logging.info(f"Headless mode activated: {headless_mode}")

    # Define a shared display queue
    shared_display_queue = queue.Queue()

    # List of RTSP URLs
    rtsp_urls = [
        'rtsp://xsens:admin12345@192.168.0.100:554/Streaming/channels/201',
        'rtsp://xsens:admin12345@192.168.0.100:554/Streaming/channels/301'
        # Add more RTSP URLs as needed
    ]

    camera_ids = [url.split('/')[-1] for url in rtsp_urls]

    # Initialize shared CSV handler
    csv_directory = r'C:\Users\KRYP MEDIA\Downloads\rkface\Allcsv'
    os.makedirs(csv_directory, exist_ok=True)  # Ensure the directory exists
    csv_path = os.path.join(csv_directory, 'emotion_tracking_shared.csv')
    target_emotions = ['happy', 'sad', 'neutral', 'angry']
    csv_handler = SharedCSVHandler(csv_path, camera_ids, target_emotions)

    # Start the periodic CSV update thread
    csv_update_thread = threading.Thread(target=csv_handler.periodic_update, daemon=True)
    csv_update_thread.start()
    logging.info("CSV update thread started.")

    trackers = []
    threads = []

    for url in rtsp_urls:
        camera_id = url.split('/')[-1]
        tracker = ObjectTracking(url, shared_display_queue, csv_handler, camera_id, headless_mode)
        trackers.append(tracker)
        thread = threading.Thread(target=tracker.process_stream)
        thread.start()
        threads.append(thread)
        logging.info(f"Started processing thread for camera {camera_id}.")

    # Start the display_frames thread only if not headless
    if not headless_mode:
        display_thread = threading.Thread(target=display_frames, args=(shared_display_queue,))
        display_thread.daemon = True
        display_thread.start()
        logging.info("Display thread started.")

    # Keep the main thread alive to allow threads to run
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Stopping all threads...")
        for tracker in trackers:
            tracker.stop()
        for thread in threads:
            thread.join()
        logging.info("All processing threads stopped.")

        if not headless_mode:
            # To ensure the display_frames thread exits
            # Send a dummy frame to unblock the display queue
            shared_display_queue.put((None, None))
            display_thread.join()

        # Optionally, write remaining data to CSV before exiting
        csv_handler.write_to_csv()
        logging.info("Final CSV write completed. Exiting.")

if __name__ == "__main__":
    main()
