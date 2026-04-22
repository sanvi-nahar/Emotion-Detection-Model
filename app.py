from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import numpy as np
import mediapipe as mp
import math
import os
import urllib.request

app = Flask(__name__)
CORS(app)

# Download model if not exists
model_path = 'face_landmarker.task'
if not os.path.exists(model_path):
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
        model_path
    )

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE
)

landmarker = FaceLandmarker.create_from_options(options)

def get_dist(p1, p2, w, h):
    x1, y1 = p1.x * w, p1.y * h
    x2, y2 = p2.x * w, p2.y * h
    return math.hypot(x2 - x1, y2 - y1)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided", "emotion": "Error"}), 400
        
    file = request.files['image']
    if not file:
        return jsonify({"error": "Empty file", "emotion": "Error"}), 400

    img = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(img, cv2.IMREAD_COLOR)
    
    if image is None:
        return jsonify({"error": "Invalid image format", "emotion": "Error"}), 400

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    results = landmarker.detect(mp_image)

    emotion = "No face detected"

    if len(results.face_landmarks) > 0:
        landmarks = results.face_landmarks[0]
        h, w, _ = image.shape

        face_width = get_dist(landmarks[234], landmarks[454], w, h)

        if face_width > 0:
            mouth_open = get_dist(landmarks[13], landmarks[14], w, h) / face_width

            bottom_lip_y = landmarks[17].y * h
            left_corner_y = landmarks[61].y * h
            right_corner_y = landmarks[291].y * h

            avg_corner_y = (left_corner_y + right_corner_y) / 2.0
            smile_curve = (bottom_lip_y - avg_corner_y) / face_width

            if mouth_open > 0.05:
                if smile_curve < 0.035:
                    emotion = "Cry / Anguish"
                elif smile_curve > 0.065:
                    emotion = "Laughing"
                else:
                    emotion = "Surprise"
            else:
                if smile_curve > 0.060:
                    emotion = "Happy"
                elif smile_curve < 0.035:
                    emotion = "Sad"
                else:
                    emotion = "Neutral"

    print("Emotion:", emotion)  # debug

    return jsonify({"emotion": emotion})

if __name__ == '__main__':
    app.run(debug=True)