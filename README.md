# Real-Time Emotion Detection with OpenCV & DeepFace

This project is a real-time facial emotion recognition system using OpenCV, Mediapipe, and DeepFace. It captures video from a webcam, detects facial landmarks, and analyzes emotions in real-time using deep learning models.

## 📌 Features

- **Real-time Face Detection**: Uses Mediapipe's FaceMesh to detect facial landmarks.
- **Emotion Analysis**: Leverages DeepFace to analyze emotions from facial expressions.
- **Live Visualization**: Displays the detected emotion on the video feed and a separate Turtle-based UI.
- **Multi-threaded Processing**: Enhances performance by analyzing emotions in a separate thread.
- **Custom Font Support**: Uses B Nazanin font for Persian-language support (if available on the system).

---

## 📥 Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/emotion-detection.git
cd emotion-detection
```

### 2️⃣ Install Dependencies

Ensure you have Python 3.7+ installed. Install the required libraries using:

```bash
pip install opencv-python mediapipe deepface numpy pillow
```

---

## 🚀 Usage

Run the following command to start the application:

```bash
python main.py
```

Press `q` to exit the application.

---

## 🛠️ Dependencies

- **OpenCV**: For capturing and processing video frames.
- **Mediapipe**: For facial landmark detection.
- **DeepFace**: For emotion analysis.
- **NumPy**: For numerical operations.
- **Pillow**: For rendering text with custom fonts.

---

## 🖥️ System Requirements

- A computer with a webcam.
- Windows/Linux/MacOS.
- Python 3.7+.

---

## 🔧 Troubleshooting

1. **Webcam not detected?**
   - Ensure your webcam is connected and not being used by another application.
2. **Emotion analysis slow?**
   - The first analysis may take longer due to model loading. Subsequent analyses will be faster.
3. **Font rendering issues?**
   - Ensure `B Nazanin` font is installed in `C:/Windows/Fonts/` (Windows users).

---

## 💡 Contributing

Contributions are welcome! If you have ideas on how to improve the project, feel free to contribute:

1. **Suggest Enhancements**: Open an issue describing your idea for improving the project.
2. **Implement a New Feature**: If you have a feature in mind, fork the repository and submit a pull request.
3. **Bug Fixes**: Help us identify and fix bugs to make the project more stable.
4. **Documentation Improvements**: If you think the documentation can be clearer, feel free to update it.

We appreciate any contributions to make this project even better! 😊

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 📧 Contact

👨‍💻 **Developer**: Shayan Taherkhani\
📧 **Email**: [shayanthn78@gmail.com](mailto:shayanthn78@gmail.com)\
🌎 **GitHub**: [github.com/shayanthn](https://github.com/shayanthn)

---

Enjoy real-time emotion analysis! 😊

