# ✨🚀 Advanced Real-Time Emotion Analysis & Micro-Expression Detection

<div align="center">
  <img src="https://media.giphy.com/media/LmNwrBhejkK9EFP504/giphy.gif" width="500" alt="Emotion Detection"/>
</div>

Hey there! 👋 Welcome to this **cutting-edge project by [Shayan Taherkhani](https://shayantaherkhani.ir)** that elevates real-time emotion analysis to a whole new level.
This isn’t just another face detection script — it’s a **sophisticated AI system** combining:

✅ Multi-model ensemble detection
✅ Micro-expression recognition
✅ Rich visualizations
✅ Real-time performance

…and much more. 🚀

---

## ✨ Features That Will Amaze You

| ⚡ Feature                              | 💡 Description                                                                                                                   |
| -------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| 🎭 **Hybrid Emotion Analysis**         | An ensemble of **DeepFace** and **FER2013 Mini-XCEPTION** models deliver robust, accurate detection. No single point of failure! |
| 🌐 **Optional Online API Integration** | Seamlessly connect to external APIs for richer insights (just set `USE_ONLINE_API = True`).                                      |
| 🔬 **Micro-Expression Detection**      | Uses **optical flow** to track fleeting facial micro-expressions like *Up Movement* and *Down-Left Movement*.                    |
| 📊 **Dynamic Visualizations**          | Live probability bars, emotion history graphs (with **Savitzky-Golay smoothing**), and FPS counters.                             |
| 🔊 **Auditory Feedback**               | Instant audio cues for detected emotions (e.g., cheerful sounds for happiness).                                                  |
| 📸 **High-Resolution Video**           | Capture in **1920×1080 @60FPS** for crystal-clear analysis.                                                                      |
| 🔒 **Thread-Safe Processing**          | Analysis runs in a separate thread for smooth video.                                                                             |
| 🧑‍💻 **Developer Showcase**           | Your info elegantly displayed on the video feed.                                                                                 |

<div align="center">
  <img src="https://i.imgur.com/PLxDScN.png" width="600" alt="Feature Showcase"/>
</div>

---

## 🛠️ Quick Setup

Ready to get started? Follow these steps:

---

### ✅ Install dependencies:

```bash
pip install opencv-python mediapipe deepface pygame tensorflow requests scipy
```

---

### 📥 Download the FER2013 Model

➡️ [**Download FER2013 Mini-XCEPTION**](https://github.com/oarriaga/face_classification/blob/master/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5)

**Place it in the same directory as your script.**

---

### 🌐 Enable Online API (Optional)

* Replace `API_ENDPOINT` with your actual endpoint.
* Set `USE_ONLINE_API = True`.
* Add your `API_KEY`.

---

### 🎛️ Adjust Sensitivity

* Tweak `min_detection_confidence` and `min_tracking_confidence` (recommended: `0.85`).
* Change the **optical flow threshold** (`3.0` by default).

---

### ▶️ Run the Application

```bash
python emotion_detection.py
```

---

## 🎯 Performance Tuning Tips

**Optimize the experience to your needs:**

| 🛠️ Goal              | 🔧 What to Adjust                                                                                                |
| --------------------- | ---------------------------------------------------------------------------------------------------------------- |
| 🧠 Higher Sensitivity | Decrease `analysis_interval` (e.g., `0.2–0.3s`), raise optical flow threshold (`5.0+`), lower smoothing factors. |
| ⚡ Better Performance  | Lower camera resolution (720p), set `USE_ONLINE_API=False`, disable FER model if not needed.                     |
| 🎯 Higher Precision   | Increase `min_detection_confidence` (`0.9+`), add more ensemble models, use a high-quality camera.               |

---

## 🙋‍♂️ About the Developer

Made with ❤️ by **Shayan Taherkhani**

<table>
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

<div align="center">
  <img src="https://media.giphy.com/media/3o7btPCcdNniyf0ArS/giphy.gif" width="400" alt="AI Emotion"/>
</div>

**This advanced system provides unprecedented detail in emotion analysis — multiple verification layers, micro-expression detection, and a beautiful real-time visualization of human emotions.**

> **Get ready to explore the fascinating world of emotion AI!** 🧠👁️✨
