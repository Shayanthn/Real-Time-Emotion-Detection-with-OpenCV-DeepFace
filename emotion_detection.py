import os
import cv2
import mediapipe as mp
from deepface import DeepFace
import numpy as np
import threading
import time
import requests
from collections import deque
from datetime import datetime
import pygame
from scipy.signal import savgol_filter
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# 🔹 Configuration
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# 🔹 Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.85,
    min_tracking_confidence=0.85
)

# 🔹 Capture high-resolution video stream
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print(f"📷 Resolution: {int(cap.get(3))}x{int(cap.get(4))}")

# 🔹 Emotion analysis variables
detected_emotion = "Analyzing..."
emotion_history = deque(maxlen=60)
emotion_probabilities = {}
emotion_lock = threading.Lock()
last_analysis_time = time.time()
analysis_interval = 0.3  # seconds

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

# 🔹 Online API configuration
API_ENDPOINT = "https://api.example.com/emotion"  # Replace with actual endpoint
API_KEY = "your_api_key_here"
USE_ONLINE_API = False  # Set to True to enable online analysis

# 🔹 Load additional models for ensemble analysis
try:
    FER_MODEL = load_model('fer2013_mini_XCEPTION.102-0.66.hdf5', compile=False)
    EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    print("✅ Additional models loaded successfully")
except:
    print("⚠️ Could not load additional models, using DeepFace only")
    FER_MODEL = None

# 🔹 Thread-safe emotion analysis with multiple models
def analyze_emotion(face):
    global detected_emotion, emotion_probabilities
    
    try:
        # Preprocess face for analysis
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (48, 48))
        
        # Ensemble analysis with multiple models
        results = []
        
        # Model 1: DeepFace
        try:
            analysis = DeepFace.analyze(
                face, 
                actions=['emotion'], 
                enforce_detection=False, 
                detector_backend='opencv',
                silent=True
            )
            deepface_emotions = analysis[0]['emotion']
            results.append(deepface_emotions)
        except Exception as e:
            print(f"DeepFace error: {str(e)}")
        
        # Model 2: FER Model
        if FER_MODEL is not None:
            try:
                fer_face = img_to_array(face_resized)
                fer_face = np.expand_dims(fer_face, axis=0)
                fer_face = fer_face.astype('float32') / 255.0
                fer_preds = FER_MODEL.predict(fer_face)[0]
                fer_emotions = {EMOTION_LABELS[i]: float(pred) * 100 for i, pred in enumerate(fer_preds)}
                results.append(fer_emotions)
            except Exception as e:
                print(f"FER model error: {str(e)}")
        
        # Model 3: Online API
        if USE_ONLINE_API:
            try:
                _, img_encoded = cv2.imencode('.jpg', face)
                response = requests.post(
                    API_ENDPOINT,
                    headers={"Authorization": f"Bearer {API_KEY}"},
                    files={"image": img_encoded.tobytes()},
                    timeout=1.5
                )
                if response.status_code == 200:
                    api_emotions = response.json().get('emotions', {})
                    results.append(api_emotions)
            except Exception as e:
                print(f"API error: {str(e)}")
        
        # Combine results from multiple models
        if results:
            combined_emotions = {}
            for emotion in results[0].keys():
                # Get predictions from all models that have this emotion
                predictions = [r[emotion] for r in results if emotion in r]
                
                # Apply weighted average (more weight to models with higher confidence)
                if predictions:
                    weights = [max(r.values()) for r in results if emotion in r]
                    total_weight = sum(weights)
                    if total_weight > 0:
                        weighted_avg = sum(p * w for p, w in zip(predictions, weights)) / total_weight
                        combined_emotions[emotion] = weighted_avg
            
            # Apply smoothing filter to emotion probabilities
            if emotion_history:
                last_probabilities = emotion_history[-1]
                smoothed_emotions = {}
                for emotion, prob in combined_emotions.items():
                    if emotion in last_probabilities:
                        # Apply exponential smoothing
                        smoothed_emotions[emotion] = 0.7 * prob + 0.3 * last_probabilities[emotion]
                    else:
                        smoothed_emotions[emotion] = prob
                combined_emotions = smoothed_emotions
            
            detected_emotion = max(combined_emotions, key=combined_emotions.get)
            
            with emotion_lock:
                emotion_probabilities = combined_emotions
                emotion_history.append(combined_emotions)
                
                # Trigger sound effect
                if detected_emotion in sound_effects and sound_effects[detected_emotion]:
                    sound_effects[detected_emotion].play()
                
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

# 🔹 Draw emotion probability bars
def draw_emotion_bars(frame, probabilities):
    if not probabilities:
        return
    
    # Setup bar chart parameters
    chart_x = frame.shape[1] - 300
    chart_y = 50
    bar_width = 250
    bar_height = 25
    max_value = max(probabilities.values()) if max(probabilities.values()) > 0 else 100
    
    # Draw each emotion bar
    for i, (emotion, prob) in enumerate(probabilities.items()):
        color = EMOTION_COLORS.get(emotion, (200, 200, 200))
        
        # Draw background bar
        cv2.rectangle(frame, 
                      (chart_x, chart_y + i*(bar_height+5)), 
                      (chart_x + bar_width, chart_y + i*(bar_height+5) + bar_height), 
                      (40, 40, 40), -1)
        
        # Draw filled bar
        bar_length = int(bar_width * (prob / max_value))
        cv2.rectangle(frame, 
                      (chart_x, chart_y + i*(bar_height+5)), 
                      (chart_x + bar_length, chart_y + i*(bar_height+5) + bar_height), 
                      color, -1)
        
        # Draw emotion label and percentage
        label = f"{emotion}: {prob:.1f}%"
        cv2.putText(frame, label, 
                   (chart_x + 5, chart_y + i*(bar_height+5) + int(bar_height/2) + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                   (255, 255, 255), 1)

# 🔹 Draw emotion history graph
def draw_emotion_graph(frame, history):
    if len(history) < 2:
        return
    
    graph_height = 150
    graph_width = 400
    graph_x = frame.shape[1] - graph_width - 20
    graph_y = frame.shape[0] - graph_height - 20
    
    # Draw graph background
    cv2.rectangle(frame, (graph_x, graph_y), 
                 (graph_x + graph_width, graph_y + graph_height),
                 (30, 30, 30), -1)
    
    # Draw grid lines
    for i in range(1, 5):
        y_pos = graph_y + i * (graph_height // 5)
        cv2.line(frame, (graph_x, y_pos), (graph_x + graph_width, y_pos), 
                (60, 60, 60), 1)
    
    # Draw data points for each emotion
    for emotion, color in EMOTION_COLORS.items():
        points = []
        for i, frame_probs in enumerate(history):
            if emotion in frame_probs:
                x = graph_x + int(i * (graph_width / len(history)))
                y = graph_y + graph_height - int(frame_probs[emotion] * graph_height / 100)
                points.append((x, y))
        
        # Apply smoothing to the line
        if len(points) > 2:
            x_vals = [p[0] for p in points]
            y_vals = [p[1] for p in points]
            
            # Apply Savitzky-Golay filter for smoothing
            try:
                window_size = min(len(y_vals), 15)
                if window_size > 5 and window_size % 2 == 1:
                    y_smooth = savgol_filter(y_vals, window_size, 3)
                    points = [(x, int(y)) for x, y in zip(x_vals, y_smooth)]
            except:
                pass
            
            # Draw the smoothed line
            for j in range(1, len(points)):
                cv2.line(frame, points[j-1], points[j], color, 2)
    
    # Draw graph border
    cv2.rectangle(frame, (graph_x, graph_y), 
                 (graph_x + graph_width, graph_y + graph_height),
                 (100, 100, 100), 1)

# 🔹 Detect micro-expressions using optical flow
def detect_micro_expressions(prev_frame, current_frame, face_region):
    if prev_frame is None or face_region is None:
        return None
    
    # Extract face region
    x, y, w, h = face_region
    prev_face = prev_frame[y:y+h, x:x+w]
    curr_face = current_frame[y:y+h, x:x+w]
    
    if prev_face.size == 0 or curr_face.size == 0:
        return None
    
    # Convert to grayscale
    prev_gray = cv2.cvtColor(prev_face, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_face, cv2.COLOR_BGR2GRAY)
    
    # Calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    # Calculate magnitude and angle of flow vectors
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Calculate average magnitude
    avg_magnitude = np.mean(magnitude)
    
    # Threshold for micro-expression detection
    if avg_magnitude > 3.0:  # Adjust based on sensitivity needs
        # Calculate direction of flow
        avg_angle = np.mean(angle)
        
        # Map angle to expression type
        if avg_angle < np.pi/4:
            return "Right Movement"
        elif avg_angle < np.pi/2:
            return "Down-Right Movement"
        elif avg_angle < 3*np.pi/4:
            return "Down Movement"
        elif avg_angle < np.pi:
            return "Down-Left Movement"
        elif avg_angle < 5*np.pi/4:
            return "Left Movement"
        elif avg_angle < 3*np.pi/2:
            return "Up-Left Movement"
        elif avg_angle < 7*np.pi/4:
            return "Up Movement"
        else:
            return "Up-Right Movement"
    
    return None

# 🔹 Main processing loop
try:
    prev_frame = None
    face_region = None
    micro_expression = None
    
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

        # 🔹 Detect micro-expressions
        if prev_frame is not None and face_region is not None:
            micro_expression = detect_micro_expressions(prev_frame, frame, face_region)
        prev_frame = frame.copy()

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape
                landmark_points = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]

                # 🔹 Get face bounding box with padding
                x_coords = [p[0] for p in landmark_points]
                y_coords = [p[1] for p in landmark_points]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                padding = int((x_max - x_min) * 0.2)
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(w, x_max + padding)
                y_max = min(h, y_max + padding)
                
                face_region = (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))

                # 🔹 Draw detailed facial landmarks
                for point in landmark_points[::2]:  # Draw every other point for performance
                    cv2.circle(frame, point, 1, (0, 255, 255), -1)

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
        
        # 🔹 Display micro-expression if detected
        if micro_expression:
            cv2.putText(frame, f"Micro: {micro_expression}", (20, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # 🔹 Display FPS
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (frame.shape[1]-150, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # 🔹 Display emotion probabilities
        with emotion_lock:
            draw_emotion_bars(frame, emotion_probabilities)
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
