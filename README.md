# 🎯 Advanced Real-Time Emotion Analysis & Micro-Expression Detection

<div align="center">
  <img src="https://media.giphy.com/media/kGCuRgmbnO5pK/giphy.gif" width="500" alt="Emotion AI"/>
</div>

👋 Welcome to this **AI-powered emotion recognition and micro-expression detection system**, meticulously crafted by **Shayan Taherkhani**.
This is **not just a simple face detector** — it’s a **multi-threaded, ensemble-based analysis platform** designed for **research-grade real-time emotion insights**.

---

## ✨ Key Features

✅ **Hybrid Ensemble Emotion Detection**

* Leverages **DeepFace**, a custom **FER2013 Mini-XCEPTION** model, and an **optional online API**.
* Weighted probability fusion for accurate and resilient predictions.

✅ **Real-Time Micro-Expression Tracking**

* Uses **optical flow analysis** to detect subtle micro-movements.
* Classifies motion direction (*Up Movement*, *Down-Left Movement*, etc.) for deeper emotional context.

✅ **Dynamic Visual Feedback**

* Live overlay of:

  * Emotion probability bars
  * Smoothed emotion history graphs
  * Real-time FPS and timestamps
  * Developer branding

✅ **Auditory Feedback**

* Plays instant sounds when specific emotions are detected.

✅ **High-Performance Video Processing**

* Captures **1920×1080 @60FPS** with optimized buffering and threading.

✅ **Customizable and Extensible**

* Adjustable sensitivity, detection intervals, and thresholds.
* Easy integration of new models or APIs.

---

## 🛠️ Quick Start Guide

### 1️⃣ Install dependencies

```bash
pip install opencv-python mediapipe deepface pygame tensorflow requests scipy
```

---

### 2️⃣ Download FER2013 model

[📥 Download here](https://github.com/oarriaga/face_classification/blob/master/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5)

> **Place the file in the project directory.**

---

### 3️⃣ (Optional) Configure Online API

Edit `API_ENDPOINT` and `API_KEY` in the code:

```python
API_ENDPOINT = "https://api.example.com/emotion"
API_KEY = "your_api_key_here"
USE_ONLINE_API = True
```

---

### 4️⃣ Run the application

```bash
python emotion_detection.py
```

---

## ⚙️ How It Works

<div align="center">
  <img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*Yur5G_0Z_1eJG0wFLSRtKQ.png" width="600" alt="Emotion Flow"/>
</div>

1. **Video Capture**

   * Opens high-resolution webcam stream.
   * Configures FPS and buffer for minimal latency.

2. **Face Mesh Landmark Detection**

   * Mediapipe extracts 468 landmarks in real-time.

3. **Bounding Box & Cropping**

   * Dynamically crops the detected face region.

4. **Multi-Model Emotion Analysis**

   * DeepFace and FER2013 model predictions are fused.
   * Optionally queries an online API.

5. **Probability Smoothing**

   * Applies exponential and Savitzky-Golay filters.

6. **Visualization & Feedback**

   * Draws overlays, bars, graphs, and text.
   * Plays emotion-specific sounds.

7. **Micro-Expression Detection**

   * Optical flow tracks subtle frame-to-frame movements.
   * Directional motion classified as micro-expressions.

---

## 🎨 Example Visual Output

<div align="center">
  <img src="https://i.imgur.com/PLxDScN.png" width="600" alt="Emotion Display"/>
</div>

---

## 🎯 Customization Options

**Sensitivity & Interval**

* `analysis_interval`: Default `0.3s` between detections.
* `min_detection_confidence`: Recommended `0.85`.

**Micro-Expression Threshold**

* `avg_magnitude`: Adjust `>3.0` for more/less sensitivity.

**Performance**

* Reduce resolution or FPS for slower machines.
* Disable online API (`USE_ONLINE_API = False`) to save bandwidth.

**Precision**

* Increase confidence threshold.
* Add more ensemble models.

---

## 🔊 Sound Effects

The system plays **instant auditory cues** when emotions are detected.
Easily customize by replacing:

```python
sound_effects = {
    "happy": pygame.mixer.Sound("sounds/happy.wav"),
    "sad": pygame.mixer.Sound("sounds/sad.wav"),
}
```

---

## 💡 Advanced Tips

✅ **Threaded Analysis**

* Emotion detection runs in a **background thread**.
* Prevents blocking video display.

✅ **Smoothing & Filtering**

* Emotion probabilities blended with **exponential smoothing**.
* History curves smoothed with **Savitzky-Golay filter**.

✅ **Extensible Design**

* Plug in new models or APIs with minimal changes.

---

## 👨‍💻 Developer Information

<table>
<tr>
<td><strong>Name</strong></td>
<td>Shayan Taherkhani</td>
</tr>
<tr>
<td><strong>🌐 Website</strong></td>
<td><a href="https://shayantaherkhani.ir">shayantaherkhani.ir</a></td>
</tr>
<tr>
<td><strong>🐙 GitHub</strong></td>
<td><a href="https://github.com/shayanthn">@shayanthn</a></td>
</tr>
<tr>
<td><strong>✉️ Email</strong></td>
<td><a href="mailto:admin@shayantaherkhani.ir">admin@shayantaherkhani.ir</a></td>
</tr>
</table>

---

## 🧠 Why Use This Project?

✅ **Research-Grade Precision**
✅ **Beautiful Real-Time Visualizations**
✅ **Fully Threaded, No Lag**
✅ **Customizable & Extensible**
✅ **Micro-Expression Analysis Beyond Standard APIs**

---

<div align="center">
  <img src="https://media.giphy.com/media/LmNwrBhejkK9EFP504/giphy.gif" width="400" alt="AI Analysis"/>
</div>

**Explore the uncharted world of real-time emotion AI.
Push the limits of perception.
Make your applications truly aware.**

✨ **Let’s build the future together.** ✨
