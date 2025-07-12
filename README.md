# 🎯 Advanced Real-Time Emotion Analysis & Micro-Expression Detection

> 🚀 Built with ❤️ by [**Shayan Taherkhani**](https://shayantaherkhani.ir)  
> 🎓 Combining AI, computer vision, and neuroscience into one powerful platform.

---

## 🌟 What’s Inside?

🔹 **Multi-Model Emotion Analysis**  
Uses a hybrid of `DeepFace`, a custom `FER2013 Mini-XCEPTION` model, and an optional online API. The results are **fused using weighted averaging** for robustness and accuracy.

🔹 **Micro-Expression Detection**  
Tracks **tiny facial movements** using optical flow and direction mapping (e.g. `↑`, `↘`, `←`) for detecting emotional shifts **that last milliseconds**.

🔹 **📊 Real-Time Visualization Overlays**
- 🎨 Emotion bars with live probability breakdowns
- 📈 Emotion history graph with smoothing
- 🎥 FPS counter and timestamps
- 🧑‍💻 Developer credits on-screen

🔹 **🔊 Audio Feedback**
- Play emotion-specific sound effects using `pygame.mixer`  
  (e.g. 😄 → happy.wav, 😢 → sad.wav)

🔹 **🧵 Multithreaded Design**
- Keeps emotion analysis in a **non-blocking thread**
- Maintains 60 FPS on full HD stream

---

## ⚙️ Installation

🧩 Required Packages:

```bash
pip install opencv-python mediapipe deepface pygame tensorflow requests scipy
````

🧠 Download Emotion Model:
📥 [`fer2013_mini_XCEPTION.hdf5`](https://github.com/oarriaga/face_classification/blob/master/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5)

> Save it in the same folder as `emotion_detection.py`

---

## 🧪 Optional: Online API Support

To activate:

```python
USE_ONLINE_API = True
API_ENDPOINT = "https://api.example.com/emotion"
API_KEY = "your_api_key_here"
```

---

## ▶️ Run the Application

```bash
python emotion_detection.py
```

📸 Default camera resolution: `1920x1080` @ `60 FPS`
🧠 Emotion inference every `0.3 seconds`

---

## 🔧 Fine-Tuning Parameters

| Purpose            | Setting                          | Recommended |
| ------------------ | -------------------------------- | ----------- |
| Detection accuracy | `min_detection_confidence`       | 0.85 – 0.95 |
| Tracking accuracy  | `min_tracking_confidence`        | 0.85 – 0.95 |
| Analysis speed     | `analysis_interval` (in seconds) | 0.3         |
| Micro-expression   | `avg_magnitude` threshold        | 3.0         |
| Model smoothing    | `Savitzky-Golay filter` on graph | window=15   |

---

## 📊 Visual Highlights

✅ Emotion bars dynamically update in real-time
✅ Smoothed emotion graph for each detected class
✅ Thread-safe drawing and analysis
✅ Auto-coloring using:

```python
EMOTION_COLORS = {
  "happy":    (0, 255, 255),   # 🌞 Yellow
  "sad":      (255, 0, 0),     # 💧 Blue
  "angry":    (0, 0, 255),     # 🔥 Red
  "surprise": (255, 0, 255),   # 💥 Magenta
  "fear":     (0, 165, 255),   # 🧡 Orange
  "disgust":  (0, 255, 0),     # 💚 Green
  "neutral":  (200, 200, 200)  # ⚪ Gray
}
```

---

## 🧑‍💻 Developer Info

| 🔹 Name   | [**Shayan Taherkhani**](https://shayantaherkhani.ir)          |
| --------- | ------------------------------------------------------------- |
| 🌐 Site   | [shayantaherkhani.ir](https://shayantaherkhani.ir)            |
| 🐙 GitHub | [@shayanthn](https://github.com/shayanthn)                    |
| 📧 Email  | [admin@shayantaherkhani.ir](mailto:admin@shayantaherkhani.ir) |

---

## 💭 Why Use This Project?

✅ Real-time deep analysis
✅ Thread-safe & high-FPS optimized
✅ Rich UI + sound + graph overlays
✅ Custom micro-expression support
✅ Fully modular and extensible for research

---

## ✅ Final Words

Emotion recognition meets neuroscience.
AI meets visual aesthetics.
Performance meets precision.

**Welcome to the next generation of human-computer interaction.**

> 💡 *“Machines that understand how you feel — that’s not science fiction anymore.”*
