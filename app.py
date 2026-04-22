from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import numpy as np
import mediapipe as mp
import math
import os
import gc

app = Flask(__name__)
CORS(app)

# ---------------------------
# MediaPipe Setup (lazy loading)
# ---------------------------
model_path = 'face_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE
)

landmarker = None  # Lazy init

# ---------------------------
# Helper Function
# ---------------------------
def get_dist(p1, p2, w, h):
    x1, y1 = p1.x * w, p1.y * h
    x2, y2 = p2.x * w, p2.y * h
    return math.hypot(x2 - x1, y2 - y1)

# ---------------------------
# Routes
# ---------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global landmarker

    try:
        if 'image' not in request.files:
            return jsonify({"emotion": "No image provided"}), 400

        file = request.files['image']
        if not file:
            return jsonify({"emotion": "Empty file"}), 400

        img = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(img, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"emotion": "Invalid image"}), 400

        # Resize for speed + stability
        image = cv2.resize(image, (640, 480))

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # 🔥 Lazy load model (fixes Render crash)
        if landmarker is None:
            landmarker = FaceLandmarker.create_from_options(options)

        results = landmarker.detect(mp_image)

        if results is None or len(results.face_landmarks) == 0:
            return jsonify({"emotion": "No face detected"})

        landmarks = results.face_landmarks[0]
        h, w, _ = image.shape

        face_width = get_dist(landmarks[234], landmarks[454], w, h)

        emotion = "Neutral"

        if face_width > 0:
            mouth_open = get_dist(landmarks[13], landmarks[14], w, h) / face_width

            bottom_lip_y = landmarks[17].y * h
            left_corner_y = landmarks[61].y * h
            right_corner_y = landmarks[291].y * h

            avg_corner_y = (left_corner_y + right_corner_y) / 2.0
            smile_curve = (bottom_lip_y - avg_corner_y) / face_width

            # Improved thresholds
            if mouth_open > 0.08:
                if smile_curve < 0.035:
                    emotion = "Cry / Anguish"
                elif smile_curve > 0.075:
                    emotion = "Laughing"
                else:
                    emotion = "Surprise"
            else:
                if smile_curve > 0.07:
                    emotion = "Happy"
                elif smile_curve < 0.035:
                    emotion = "Sad"
                else:
                    emotion = "Neutral"

        print("Emotion:", emotion)

        gc.collect()  # free memory

        return jsonify({"emotion": emotion})

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"emotion": "Error processing image"}), 500

# ---------------------------
# Run App (Render compatible)
# ---------------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)