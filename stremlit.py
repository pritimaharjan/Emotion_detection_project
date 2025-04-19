import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import numpy as np
import time

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load emotion detection model (trained on face crops)
model = YOLO('new_best .pt')

# Function to detect emotion from face crops
def detect_emotion(video_path):
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()
    st.info("Processing video... Please wait.")

    if not cap.isOpened():
        st.error("Error opening video file.")
        return

    prev_time = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame
        frame = cv2.resize(frame, (640, 640))

        # Detect faces using Haar Cascade
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_crop = frame[y:y+h, x:x+w]
            if face_crop.size == 0:
                continue

            # Predict emotion on cropped face
            results = model.predict(face_crop, conf=0.25)

            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        cls = int(box.cls[0])
                        label = model.names[cls]
                        conf = box.conf[0].item()

                        if label in ["Anger", "Contempt", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]:
                            # Draw bounding box and label on original frame
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            cv2.putText(frame, f"{label} {conf:.2f}", (x, y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # FPS calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Show frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB")

    cap.release()
    st.success("Video processing completed!")

# Streamlit UI
st.title("🎭 Emotion Detection (Face-focused) using YOLOv8 + Haar Cascade")

uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov", "mkv"])

if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    detect_emotion(tfile.name)
