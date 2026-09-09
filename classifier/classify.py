import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import numpy as np
from PIL import Image
import json
import sys

# ── PATHS ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'lettuce_model.tflite')
LABELS_PATH = os.path.join(BASE_DIR, 'model', 'lettuce_labels.json')

# ── LOAD MODEL ───────────────────────────────────────────
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open(LABELS_PATH, 'r') as f:
    labels = json.load(f)

print("✅ Model loaded successfully")
print(f"   Categories: {list(labels.values())}\n")

# ── PREPROCESS ───────────────────────────────────────────
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

# ── INFERENCE ────────────────────────────────────────────
def classify(image_path):
    print(f"Classifying: {image_path}")

    img_array = preprocess_image(image_path)
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    probabilities = output[0]

    predicted_index = np.argmax(probabilities)
    predicted_label = labels[str(predicted_index)]
    confidence = probabilities[predicted_index] * 100

    print("\n── Results ──────────────────────────────────")
    for i, prob in enumerate(probabilities):
        label = labels[str(i)]
        bar = '█' * int(prob * 30)
        marker = ' ← PREDICTION' if i == predicted_index else ''
        print(f"  {label:<25} {prob*100:5.1f}%  {bar}{marker}")

    print(f"\n🌿 Prediction : {predicted_label}")
    print(f"📊 Confidence : {confidence:.1f}%")
    print("─────────────────────────────────────────────")

    return predicted_label, confidence

# ── MAIN ─────────────────────────────────────────────────
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage  : python3 classify.py <image_path>")
        print("Example: python3 classify.py leaf.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        sys.exit(1)

    classify(image_path)
