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

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def display_frames(display_queue):
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
                break
        except queue.Empty:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    for window in windows:
        cv2.destroyWindow(window)

class ObjectTracking:
    def __init__(self, stream_url):
        self.stream_url = stream_url
        self.camera_id = stream_url.split('/')[-1]
        self.display_queue = queue.Queue()

        base_dir = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(base_dir, r'C:\Users\KRYP MEDIA\Downloads\rkface\FInal_Codes\yolov8m.pt')
        self.bytetrack_yaml_path = r'bytetrack.yaml'

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(weights_path).to(self.device)
        self.model.fuse()

        try:
            self.face_model = YOLO(r'C:\Users\KRYP MEDIA\Downloads\rkface\yolov8n-face.pt').to(self.device)
            logging.info("Face model initialized")
        except Exception as e:
            logging.error(f"Error initializing face model: {e}")
            raise

        # Emotion tracking parameters
        self.emotion_window_size = 5
        self.emotion_confidence_threshold = 0.80
        self.detected_persons = {}
        self.target_emotions = ['happy', 'sad', 'neutral', 'angry']

        # Zone coordinates (adjust these based on the original frame resolution)
        self.zone_points = np.array([[308, 454], [1678, 580], [1733, 1005], [246, 1000], [308, 454]], dtype=np.int32)

        # CSV file initialization
        self.csv_file = os.path.join(base_dir, 'zone_people_emotions.csv')
        with open(self.csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Timestamp', 'Number of People', 'Happy', 'Sad', 'Neutral', 'Angry'])

        # Timer for CSV update
        self.last_csv_update = time.time()
        self.csv_update_interval = 60  # in seconds

    def is_inside_zone(self, box):
        """
        Check if the center of the bounding box is inside the defined zone
        """
        x_center = (box[0] + box[2]) // 2
        y_center = (box[1] + box[3]) // 2
        return cv2.pointPolygonTest(self.zone_points, (float(x_center), float(y_center)), False) >= 0

    def detect_emotion(self, face_crop, id):
        """
        Detect and track emotions with confidence scores
        """
        try:
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

    def update_csv(self):
        """
        Update CSV file with the current number of people and their emotions in the zone
        """
        num_people = len(self.detected_persons)
        emotion_counts = {emotion: 0 for emotion in self.target_emotions}

        for person in self.detected_persons.values():
            if person['current_emotion'] in emotion_counts:
                emotion_counts[person['current_emotion']] += 1

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, num_people, emotion_counts['happy'], emotion_counts['sad'], emotion_counts['neutral'], emotion_counts['angry']])

    def process_frame(self, frame):
        if frame is None or frame.size == 0:
            logging.warning(f"Received empty frame for Camera {self.camera_id}. Skipping processing.")
            return None

        # Ensure frame dimensions are divisible by 32
        height, width, _ = frame.shape
        new_width = (width // 32) * 32
        new_height = (height // 32) * 32
        resized_frame = cv2.resize(frame, (new_width, new_height))

        display_frame = resized_frame.copy()
        cv2.polylines(display_frame, [self.zone_points], isClosed=True, color=(0, 255, 255), thickness=2)

        frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb).to(self.device).float().permute(2, 0, 1).unsqueeze(0) / 255.0

        with torch.no_grad():
            try:
                results = self.model.track(
                    source=frame_tensor,
                    persist=True,
                    tracker=self.bytetrack_yaml_path,
                    classes=[0],
                    device=self.device
                )

                if results and len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                    ids = results[0].boxes.id
                    if ids is not None:
                        ids = ids.cpu().numpy().astype(int)
                    else:
                        ids = []

                    for box, id in zip(boxes, ids):
                        if not self.is_inside_zone(box):
                            continue

                        face_results = self.face_model(resized_frame[box[1]:box[3], box[0]:box[2]])
                        if face_results and len(face_results[0].boxes) > 0:
                            face_box = face_results[0].boxes[0].xyxy.cpu().numpy().astype(int)[0]
                            face_crop = resized_frame[
                                box[1] + face_box[1]:box[1] + face_box[3],
                                box[0] + face_box[0]:box[0] + face_box[2]
                            ]

                            if face_crop.shape[0] > 0 and face_crop.shape[1] > 0:
                                self.detect_emotion(face_crop, id)

                                if id in self.detected_persons:
                                    person_data = self.detected_persons[id]
                                    emotion = person_data.get('current_emotion', 'Unknown')
                                    confidence = person_data.get('confidence', 0.0)

                                    if emotion == 'happy':
                                        color = (0, 255, 0)
                                    elif emotion == 'sad':
                                        color = (255, 0, 0)
                                    elif emotion == 'angry':
                                        color = (0, 0, 255)
                                    elif emotion == 'neutral':
                                        color = (255, 255, 0)
                                    else:
                                        color = (200, 200, 200)

                                    cv2.rectangle(display_frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                                    if emotion != 'Unknown':
                                        confidence_pct = confidence * 100
                                        text = f"Id{id}: {emotion} ({confidence_pct:.1f}%)"
                                        cv2.putText(display_frame, text, (box[0], box[1] - 10),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            else:
                                logging.info(f"No valid face detected for ID {id}. Skipping emotion detection.")
                        else:
                            logging.info(f"No face detected for ID {id}. Skipping emotion detection.")
            except Exception as e:
                logging.error(f"Error processing frame for camera {self.camera_id}: {e}")
                return None

        current_time = time.time()
        if current_time - self.last_csv_update >= self.csv_update_interval:
            self.update_csv()
            self.last_csv_update = current_time

        return display_frame

    def process_stream(self):
        cap = cv2.VideoCapture(self.stream_url)

        if not cap.isOpened():
            logging.error(f"Failed to open camera stream for camera {self.camera_id}")
            return

        display_thread = threading.Thread(target=display_frames, args=(self.display_queue,))
        display_thread.daemon = True
        display_thread.start()

        frame_skip = 10  # Skip every 5 frames (adjust this value based on your performance needs)
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                logging.warning(f"Failed to receive frame from camera {self.camera_id}. Retrying...")
                time.sleep(1)
                continue

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            processed_frame = self.process_frame(frame)
            if processed_frame is not None:
                self.display_queue.put((self.camera_id, processed_frame))

        cap.release()

def main():
    rtsp_url = 'rtsp://xsens:admin12345@192.168.0.100:554/Streaming/channels/201'
    
    try:
        tracker = ObjectTracking(rtsp_url)
        tracker.process_stream()
    except KeyboardInterrupt:
        logging.info("Application stopped by user")
    except Exception as e:
        logging.error(f"Application error: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
