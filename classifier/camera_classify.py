"""
Live Camera Inference Script
Supports: Pi Camera v1/v2 (picamera), Pi Camera v3 (picamera2), USB webcam (opencv)
Auto-detects available camera library.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import numpy as np
from PIL import Image
import json
import time

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

print("✅ Model loaded")

# ── AUTO-DETECT CAMERA ───────────────────────────────────
camera_mode = None

try:
    from picamera2 import Picamera2
    camera_mode = 'picamera2'
    print("📷 Camera: Pi Camera v3 (picamera2)")
except ImportError:
    pass

if not camera_mode:
    try:
        import picamera
        camera_mode = 'picamera'
        print("📷 Camera: Pi Camera v1/v2 (picamera)")
    except ImportError:
        pass

if not camera_mode:
    try:
        import cv2
        camera_mode = 'opencv'
        print("📷 Camera: USB Webcam (OpenCV)")
    except ImportError:
        pass

if not camera_mode:
    print("❌ No camera library found.")
    print("   Install one of:")
    print("   pip install picamera2   (Pi Camera v3)")
    print("   pip install picamera    (Pi Camera v1/v2)")
    print("   pip install opencv-python (USB webcam)")
    exit(1)

# ── INFERENCE ────────────────────────────────────────────
def classify_array(img_array):
    arr = np.array(img_array.resize((224, 224)), dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    interpreter.set_tensor(input_details[0]['index'], arr)
    interpreter.invoke()
    probs = interpreter.get_tensor(output_details[0]['index'])[0]
    idx = np.argmax(probs)
    return labels[str(idx)], probs[idx] * 100, probs

def print_result(pred, conf, probs):
    os.system('clear')
    print("=" * 50)
    print("  🌿 Lettuce Disease Classifier — LIVE")
    print("=" * 50)
    for i, prob in enumerate(probs):
        name = labels[str(i)]
        bar = '█' * int(prob * 30)
        marker = ' ◄' if i == np.argmax(probs) else ''
        print(f"  {name:<25} {prob*100:5.1f}%  {bar}{marker}")
    print("=" * 50)
    emoji = "✅" if pred == "Healthy" else "⚠️"
    print(f"  {emoji}  {pred}  ({conf:.1f}%)")
    print("=" * 50)
    print("  Press Ctrl+C to stop")

# ── MAIN LOOP ────────────────────────────────────────────
print("\nStarting live classification... (Ctrl+C to stop)\n")
CAPTURE_PATH = '/tmp/capture.jpg'

try:
    if camera_mode == 'picamera2':
        cam = Picamera2()
        config = cam.create_still_configuration(main={"size": (640, 480)})
        cam.configure(config)
        cam.start()
        time.sleep(2)
        while True:
            cam.capture_file(CAPTURE_PATH)
            img = Image.open(CAPTURE_PATH).convert('RGB')
            pred, conf, probs = classify_array(img)
            print_result(pred, conf, probs)
            time.sleep(1)

    elif camera_mode == 'picamera':
        import picamera
        with picamera.PiCamera() as cam:
            cam.resolution = (640, 480)
            time.sleep(2)
            while True:
                cam.capture(CAPTURE_PATH)
                img = Image.open(CAPTURE_PATH).convert('RGB')
                pred, conf, probs = classify_array(img)
                print_result(pred, conf, probs)
                time.sleep(1)

    elif camera_mode == 'opencv':
        import cv2
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Could not open USB camera")
            exit(1)
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to capture frame")
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            pred, conf, probs = classify_array(img)
            print_result(pred, conf, probs)
            time.sleep(1)
        cap.release()

except KeyboardInterrupt:
    print("\n\n👋 Stopped.")
