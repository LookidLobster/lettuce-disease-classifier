import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import numpy as np
from PIL import Image, ImageTk
import json
import tkinter as tk
from tkinter import filedialog
import threading

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

# ── THEME ────────────────────────────────────────────────
COLORS = {
    'Bacterial':            '#E74C3C',
    'Fungal':               '#E67E22',
    'Healthy':              '#27AE60',
    'Pest damage':          '#8E44AD',
    'Shepherd_purse_weeds': '#2980B9',
}
BG       = '#1A1A2E'
CARD     = '#16213E'
ACCENT   = '#0F3460'
FG       = '#EAEAEA'
MUTED    = '#A0A0A0'

# ── INFERENCE ────────────────────────────────────────────
def classify(image_path):
    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize((224, 224))
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    interpreter.set_tensor(input_details[0]['index'], arr)
    interpreter.invoke()
    probs = interpreter.get_tensor(output_details[0]['index'])[0]
    idx = np.argmax(probs)
    return labels[str(idx)], probs[idx] * 100, probs, img

# ── APP ──────────────────────────────────────────────────
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("🌿 Lettuce Disease Classifier")
        self.root.configure(bg=BG)
        self.root.geometry("900x650")
        self.root.resizable(False, False)
        self._build()

    def _build(self):
        # Header
        hdr = tk.Frame(self.root, bg=ACCENT, pady=15)
        hdr.pack(fill='x')
        tk.Label(hdr, text="🌿 Lettuce Disease Classifier",
                 font=('Helvetica', 22, 'bold'), bg=ACCENT, fg='white').pack()
        tk.Label(hdr, text="MobileNetV2 + TFLite on Raspberry Pi",
                 font=('Helvetica', 10), bg=ACCENT, fg=MUTED).pack()

        # Content
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill='both', expand=True, padx=20, pady=20)

        # Left panel — image
        left = tk.Frame(body, bg=CARD, width=380, height=420)
        left.pack(side='left', padx=(0, 10), fill='y')
        left.pack_propagate(False)
        tk.Label(left, text="INPUT IMAGE", font=('Helvetica', 10, 'bold'),
                 bg=CARD, fg=MUTED).pack(pady=(15, 5))
        self.img_lbl = tk.Label(left, bg=CARD, text="No image loaded",
                                fg=MUTED, font=('Helvetica', 12))
        self.img_lbl.pack(expand=True)

        # Right panel
        right = tk.Frame(body, bg=BG)
        right.pack(side='left', fill='both', expand=True)

        # Prediction card
        pred_card = tk.Frame(right, bg=CARD, pady=15)
        pred_card.pack(fill='x', pady=(0, 15))
        tk.Label(pred_card, text="PREDICTION", font=('Helvetica', 10, 'bold'),
                 bg=CARD, fg=MUTED).pack()
        self.pred_lbl = tk.Label(pred_card, text="—",
                                  font=('Helvetica', 28, 'bold'), bg=CARD, fg='white')
        self.pred_lbl.pack()
        self.conf_lbl = tk.Label(pred_card, text="Confidence: —",
                                  font=('Helvetica', 13), bg=CARD, fg=MUTED)
        self.conf_lbl.pack()

        # Bars card
        bars_card = tk.Frame(right, bg=CARD, pady=10, padx=15)
        bars_card.pack(fill='both', expand=True)
        tk.Label(bars_card, text="CLASS PROBABILITIES",
                 font=('Helvetica', 10, 'bold'), bg=CARD, fg=MUTED).pack(anchor='w', pady=(0, 10))

        self.bars = {}
        self.pct_lbls = {}
        for i in range(5):
            name = labels[str(i)]
            color = COLORS.get(name, '#FFFFFF')
            row = tk.Frame(bars_card, bg=CARD)
            row.pack(fill='x', pady=4)
            tk.Label(row, text=name.replace('_', ' '), font=('Helvetica', 10),
                     bg=CARD, fg=FG, width=20, anchor='w').pack(side='left')
            bg_bar = tk.Frame(row, bg='#2C2C2C', height=18, width=200)
            bg_bar.pack(side='left', padx=5)
            bg_bar.pack_propagate(False)
            fill = tk.Frame(bg_bar, bg=color, height=18, width=0)
            fill.place(x=0, y=0, height=18)
            self.bars[i] = (bg_bar, fill)
            pct = tk.Label(row, text="0.0%", font=('Helvetica', 10, 'bold'),
                           bg=CARD, fg=color, width=6)
            pct.pack(side='left')
            self.pct_lbls[i] = pct

        # Buttons
        btn_row = tk.Frame(self.root, bg=BG, pady=10)
        btn_row.pack()
        tk.Button(btn_row, text="📂  Load Image", font=('Helvetica', 13, 'bold'),
                  bg=ACCENT, fg='white', padx=20, pady=8, relief='flat',
                  cursor='hand2', command=self.load).pack(side='left', padx=10)
        tk.Button(btn_row, text="🔄  Clear", font=('Helvetica', 13),
                  bg='#2C2C2C', fg='white', padx=20, pady=8, relief='flat',
                  cursor='hand2', command=self.clear).pack(side='left', padx=10)

        # Status bar
        self.status = tk.Label(self.root, text="Ready — Load an image to classify",
                               font=('Helvetica', 9), bg=ACCENT, fg=MUTED, pady=5)
        self.status.pack(fill='x', side='bottom')

    def load(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return
        self.status.config(text=f"Classifying: {os.path.basename(path)}...")
        self.root.update()
        threading.Thread(target=self._infer, args=(path,), daemon=True).start()

    def _infer(self, path):
        pred, conf, probs, img = classify(path)
        color = COLORS.get(pred, 'white')

        # Update image preview
        img.thumbnail((340, 340))
        photo = ImageTk.PhotoImage(img)
        self.img_lbl.config(image=photo, text='')
        self.img_lbl.image = photo

        # Update prediction
        self.pred_lbl.config(text=pred.replace('_', ' '), fg=color)
        self.conf_lbl.config(text=f"Confidence: {conf:.1f}%")

        # Update bars
        for i, prob in enumerate(probs):
            _, fill = self.bars[i]
            fill.place(x=0, y=0, height=18, width=int(prob * 200))
            self.pct_lbls[i].config(text=f"{prob*100:.1f}%")

        emoji = "✅" if pred == "Healthy" else "⚠️"
        self.status.config(
            text=f"{emoji}  {pred} ({conf:.1f}%) | {os.path.basename(path)}")

    def clear(self):
        self.img_lbl.config(image='', text='No image loaded')
        self.img_lbl.image = None
        self.pred_lbl.config(text='—', fg='white')
        self.conf_lbl.config(text='Confidence: —')
        for i in range(5):
            _, fill = self.bars[i]
            fill.place(x=0, y=0, height=18, width=0)
            self.pct_lbls[i].config(text='0.0%')
        self.status.config(text='Ready — Load an image to classify')

if __name__ == '__main__':
    root = tk.Tk()
    App(root)
    root.mainloop()
