from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import numpy as np
import os

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
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

        image = cv2.resize(image, (640, 480))

        # 🔥 ONLY DeepFace
        from deepface import DeepFace

        result = DeepFace.analyze(
            image,
            actions=['emotion'],
            enforce_detection=False
        )

        if isinstance(result, list):
            result = result[0]

        print("DeepFace result:", result)

        emotion = result.get('dominant_emotion', 'No emotion detected')

        return jsonify({"emotion": emotion})

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"emotion": "Error"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)