import os
import cv2
import mediapipe as mp
from deepface import DeepFace
import numpy as np
import threading
import time
import pygame
from collections import deque
from datetime import datetime

# 🔹 Configuration
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# 🔹 Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# 🔹 Capture video stream
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
max_width = 1280
max_height = 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, max_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, max_height)
cap.set(cv2.CAP_PROP_FPS, 60)

print(f"📷 Resolution: {max_width}x{max_height}")

# 🔹 Emotion analysis variables
detected_emotion = "Analyzing..."
emotion_history = deque(maxlen=30)
emotion_lock = threading.Lock()
last_analysis_time = time.time()
analysis_interval = 0.5  # seconds

# 🔹 Sound effects initialization
pygame.mixer.init()
sound_effects = {
    "happy": pygame.mixer.Sound(pygame.mixer.Sound(bytes(2000)) if False else None,
    "sad": pygame.mixer.Sound(pygame.mixer.Sound(bytes(1000)) if False else None
}

# 🔹 Performance metrics
fps_counter = 0
fps_last_time = time.time()
fps_deque = deque(maxlen=30)

# 🔹 Developer information
DEV_INFO = {
    "name": "Shayan Taherkhani",
    "website": "shayantaherkhani.ir",
    "github": "github.com/shayanthn",
    "email": "admin@shayantaherkhani.ir"
}

# 🔹 Emotion color mapping
EMOTION_COLORS = {
    "happy": (0, 255, 255),    # Yellow
    "sad": (255, 0, 0),        # Blue
    "angry": (0, 0, 255),      # Red
    "surprise": (255, 0, 255), # Magenta
    "fear": (0, 165, 255),     # Orange
    "disgust": (0, 255, 0),    # Green
    "neutral": (200, 200, 200) # Gray
}

# 🔹 Thread-safe emotion analysis
def analyze_emotion(face):
    global detected_emotion
    try:
        analysis = DeepFace.analyze(
            face, 
            actions=['emotion'], 
            enforce_detection=False, 
            detector_backend='opencv',
            silent=True
        )
        emotions = analysis[0]['emotion']
        emotion = max(emotions, key=emotions.get)
        
        with emotion_lock:
            detected_emotion = emotion
            emotion_history.append(emotion)
            
            # Trigger sound effect
            if emotion in sound_effects and sound_effects[emotion]:
                sound_effects[emotion].play()
                
    except Exception as e:
        print(f"⚠️ Analysis error: {str(e)}")

# 🔹 Draw developer info overlay
def draw_dev_info(frame):
    y_offset = 30
    for i, (key, value) in enumerate(DEV_INFO.items()):
        text = f"{key.capitalize()}: {value}"
        cv2.putText(frame, text, (10, y_offset + i*30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                   (0, 255, 0), 2)

# 🔹 Draw emotion history graph
def draw_emotion_graph(frame, history):
    if not history:
        return
    
    graph_height = 100
    graph_width = 300
    graph_x = frame.shape[1] - graph_width - 20
    graph_y = 20
    
    # Draw graph background
    cv2.rectangle(frame, (graph_x, graph_y), 
                 (graph_x + graph_width, graph_y + graph_height),
                 (40, 40, 40), -1)
    
    # Draw data points
    emotion_values = list(history)
    for i in range(1, len(emotion_values)):
        color = EMOTION_COLORS.get(emotion_values[i], (200, 200, 200))
        y1 = graph_y + graph_height - (i-1)*3
        y2 = graph_y + graph_height - i*3
        cv2.line(frame, 
                (graph_x + (i-1)*10, y1),
                (graph_x + i*10, y2),
                color, 2)

# 🔹 Main processing loop
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("🚨 Frame capture error")
            break

        # 🔹 Calculate FPS
        fps_counter += 1
        if time.time() - fps_last_time >= 1.0:
            fps = fps_counter
            fps_deque.append(fps)
            fps_counter = 0
            fps_last_time = time.time()
        
        avg_fps = sum(fps_deque) / len(fps_deque) if fps_deque else 0

        # 🔹 Process frame with Face Mesh
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape
                landmark_points = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]

                # 🔹 Draw face landmarks
                for point in landmark_points[::5]:  # Draw every 5th point for performance
                    cv2.circle(frame, point, 1, (0, 255, 255), -1)

                # 🔹 Get face bounding box
                x_coords = [p[0] for p in landmark_points]
                y_coords = [p[1] for p in landmark_points]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                # 🔹 Dynamic padding based on face size
                padding = int((x_max - x_min) * 0.15)
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(w, x_max + padding)
                y_max = min(h, y_max + padding)

                # 🔹 Draw face bounding box
                color = EMOTION_COLORS.get(detected_emotion, (200, 200, 200))
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 3)
                
                # 🔹 Analyze emotion at intervals
                current_time = time.time()
                if current_time - last_analysis_time > analysis_interval:
                    face_crop = frame[y_min:y_max, x_min:x_max]
                    if face_crop.size > 0:
                        threading.Thread(target=analyze_emotion, args=(face_crop,), daemon=True).start()
                        last_analysis_time = current_time

        # 🔹 Display emotion information
        color = EMOTION_COLORS.get(detected_emotion, (200, 200, 200))
        cv2.putText(frame, f"Emotion: {detected_emotion}", (20, 40), 
                   cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 2)
        
        # 🔹 Display FPS
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (frame.shape[1]-150, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # 🔹 Display emotion history graph
        draw_emotion_graph(frame, emotion_history)
        
        # 🔹 Display developer info
        draw_dev_info(frame)
        
        # 🔹 Show timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (frame.shape[1]-300, frame.shape[0]-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

        # 🔹 Display frame
        cv2.imshow("Advanced Emotion Analysis - Press 'ESC' to exit", frame)

        # 🔹 Exit on ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()

print("✅ Session ended successfully")
