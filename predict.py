import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
from collections import Counter

checkpoint = torch.load('sign_model.pth', weights_only=False)
classes = checkpoint['classes']

class ASLModel(nn.Module):
    def __init__(self, num_classes):
        super(ASLModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = ASLModel(len(classes))
model.load_state_dict(checkpoint['model'])
model.eval()
print(f"Model loaded. Classes: {classes}")

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

roi_top, roi_bottom, roi_left, roi_right = 100, 364, 250, 514
prediction_history = []
last_prediction = "-"

def process_roi(roi):
    # Convert to grayscale and apply threshold to get white hand on white background 
    # like the dataset (which is white background with dark hand)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive threshold - makes hand white, background darker
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # Morphological operations to clean up
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Invert to match dataset format (white background, dark hand)
    thresh = 255 - thresh
    
    # Convert back to BGR for model
    result = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    return result

print("Show hand signs in the box. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    roi = frame[roi_top:roi_bottom, roi_left:roi_right]
    
    # Process ROI to look like dataset
    processed = process_roi(roi)
    
    img = Image.fromarray(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
    tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, predicted = probs.max(1)
    
    predicted_letter = classes[predicted.item()]
    confidence = conf.item()
    
    if confidence > 0.3:
        prediction_history.append(predicted_letter)
        if len(prediction_history) > 20:
            prediction_history.pop(0)
        if len(prediction_history) >= 5:
            most_common = Counter(prediction_history).most_common(1)[0][0]
            last_prediction = most_common

    cv2.rectangle(frame, (roi_left, roi_top), (roi_right, roi_bottom), (0, 255, 0), 2)
    cv2.putText(frame, f"Letter: {last_prediction}", (10, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.putText(frame, f"Conf: {confidence:.2f}", (10, 80), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show processed version
    preview = cv2.resize(processed, (200, 200))
    cv2.putText(preview, "Processed", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)
    cv2.imshow("Processed", preview)
    
    cv2.imshow("Sign Language Translator", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done.")