import os
import base64
import cv2
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
from collections import Counter

def base64_to_image(base64_string):
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        img_bytes = base64.b64decode(base64_string)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        print(f"Image conversion error: {e}")
        raise

def enhance_image(image):
    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, 50)
        v = np.clip(v, 0, 255)
        enhanced_hsv = cv2.merge((h, s, v))
        enhanced_image = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
        return enhanced_image
    except Exception as e:
        print(f"Image enhancement error: {e}")
        return image

app = Flask(__name__)
CORS(app)
model = YOLO("yolov8n.pt")

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.json:
        return jsonify({"error": "No image provided"}), 400
    try:
        base64_image = request.json['image']
        image = base64_to_image(base64_image)
        enhanced_image = enhance_image(image)
        results = model(enhanced_image)
        object_counts = Counter()
        bounding_boxes, confidences, class_ids = [], [], []
        colors = np.random.randint(0, 255, size=(len(results[0].names), 3), dtype="uint8")
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                label = result.names[int(box.cls[0])]
                if conf > 0.4:
                    bounding_boxes.append([x1, y1, x2, y2])
                    confidences.append(conf)
                    class_ids.append(label)
        indices = cv2.dnn.NMSBoxes(bounding_boxes, confidences, 0.5, 0.4)
        for i in indices.flatten():
            x1, y1, x2, y2 = bounding_boxes[i]
            label = class_ids[i]
            color = [int(c) for c in colors[i % len(colors)]]
            object_counts[label] += 1
            cv2.rectangle(enhanced_image, (x1, y1), (x2, y2), color, 3)
            text = f"{label} {confidences[i]:.2f}"
            cv2.putText(enhanced_image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        _, buffer = cv2.imencode('.jpg', enhanced_image)
        processed_base64 = base64.b64encode(buffer).decode('utf-8')
        return jsonify({"processed_image": processed_base64, "object_counts": dict(object_counts)})
    except Exception as e:
        print(f"Detection error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
