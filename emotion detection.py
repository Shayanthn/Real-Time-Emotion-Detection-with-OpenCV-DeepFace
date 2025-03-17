import os
import cv2
import mediapipe as mp
from deepface import DeepFace
import numpy as np
import threading
import turtle
from PIL import ImageFont, ImageDraw, Image
import ctypes

# 🔹 غیرفعال کردن هشدارهای TensorFlow برای افزایش سرعت
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# 🔹 راه‌اندازی Mediapipe برای تحلیل جزئیات چهره
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.7)

# 🔹 دریافت رزولوشن بیشینه‌ی پشتیبانی شده توسط وب‌کم
cap = cv2.VideoCapture(0)
max_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
max_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30

print(f"📷 بالاترین رزولوشن پشتیبانی شده: {max_width}x{max_height} @ {fps} FPS")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, max_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, max_height)
cap.set(cv2.CAP_PROP_FPS, fps)

# 🔹 متغیر برای ذخیره احساس تشخیص داده‌شده
detected_emotion = "در حال تحلیل..."

# 🔹 تنظیم پنجره‌ی نمایش احساسات با `Turtle`
emotion_screen = turtle.Screen()
emotion_screen.setup(400, 200)
emotion_screen.bgcolor("black")
emotion_screen.title("تحلیل احساسات")
emotion_turtle = turtle.Turtle()
emotion_turtle.hideturtle()
emotion_turtle.penup()
emotion_turtle.color("white")

# 🔹 تنظیم پنجره‌ی اطلاعات توسعه‌دهنده
info_screen = turtle.Screen()
info_screen.setup(400, 200)
info_screen.bgcolor("darkgreen")
info_screen.title("اطلاعات توسعه‌دهنده")
info_turtle = turtle.Turtle()
info_turtle.hideturtle()
info_turtle.penup()
info_turtle.color("white")
info_turtle.goto(-180, 50)
info_turtle.write("👨‍💻 توسعه‌دهنده: شایان طاهرخانی\n📧 ایمیل: shayanthn78@gmail.com\n🌎 گیت‌هاب: github.com/shayanthn", 
                  align="left", font=("B Nazanin", 14, "bold"))

# 🔹 دریافت مسیر فونت `B Nazanin` از ویندوز
def get_b_nazanin_font():
    font_dir = "C:/Windows/Fonts"
    font_files = ["BNAZANIN.TTF", "B_NAZANIN.TTF", "BNazanin.ttf", "b_nazanin.ttf"]
    for font_file in font_files:
        font_path = os.path.join(font_dir, font_file)
        if os.path.exists(font_path):
            return font_path
    return None  # اگر فونت پیدا نشد

b_nazanin_font = get_b_nazanin_font()

# 🔹 تابع برای تحلیل احساسات در `Thread`
def analyze_emotion(face):
    global detected_emotion
    try:
        analysis = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
        emotions = analysis[0]['emotion']
        detected_emotion = max(emotions, key=emotions.get)
    except Exception as e:
        detected_emotion = "خطا در تشخیص"

# 🔹 حلقه‌ی اصلی برای دریافت و پردازش فریم‌ها
while True:
    ret, frame = cap.read()
    if not ret:
        print("🚨 خطا در دریافت تصویر از وب‌کم!")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape
            landmark_points = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]

            for point in landmark_points[:468]:  
                cv2.circle(frame, point, 1, (255, 0, 0), -1)  

            x_min, y_min = min(landmark_points)
            x_max, y_max = max(landmark_points)

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)

            face_crop = frame[y_min:y_max, x_min:x_max]

            threading.Thread(target=analyze_emotion, args=(face_crop,)).start()

    # نمایش احساسات در پنجره‌ی OpenCV با فونت `B Nazanin`
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    if b_nazanin_font:
        try:
            font = ImageFont.truetype(b_nazanin_font, 32)
            draw.text((20, 60), f"احساس: {detected_emotion}", font=font, fill=(0, 255, 0))
            frame = np.array(img_pil)
        except:
            cv2.putText(frame, f"احساس: {detected_emotion}", (20, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # بروزرسانی احساسات در پنجره‌ی `Turtle`
    emotion_turtle.clear()
    emotion_turtle.goto(-80, 0)
    emotion_turtle.write(f"احساس: {detected_emotion}", align="center", font=("B Nazanin", 24, "bold"))

    cv2.imshow("تحلیل احساسات - دکمه 'q' برای خروج", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# بستن منابع
cap.release()
cv2.destroyAllWindows()
emotion_screen.bye()
info_screen.bye()

## https://github.com/Shayanthn
## shayanthn78@gmail.com
""" 🚀 About the Developer
Python Expert | Advanced English Instructor | Creative Thinker

Passionate Python developer with expertise in web development, AI, automation, and data science. Skilled in problem-solving, clean code, and scalable solutions. Also an advanced English instructor with strong communication and technical documentation abilities.

💡 Innovator & Idea Generator | Always seeking to create efficient, cutting-edge projects.
📌 Proficient in Django, Flask, FastAPI, TensorFlow, Pandas, Selenium, and more.
🚀 Open to collaboration on GitHub and tech-driven projects.
 """

