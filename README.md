# 🌿 Lettuce Disease Classifier
### AI-Powered Plant Health Detection on Raspberry Pi

> Internship Project — EETAR Laboratory, Department of Electrical Engineering, Chulalongkorn University  
> Supervisor: Assoc. Prof. Suwit Kiravittaya

---

## 📌 Overview

A lightweight AI system that classifies lettuce leaf health status in real-time using a camera connected to a Raspberry Pi 5. The model is trained on a dataset of 4,052 images across 5 categories and deployed as a TFLite model for efficient edge inference.

This project was developed as part of an IoT & Embedded Systems internship, motivated by the lab's ongoing research in smart plant factory technology and precision agriculture.

---

## 🎯 Categories

| Class | Description |
|---|---|
| ✅ Healthy | Normal, healthy lettuce leaf |
| 🦠 Bacterial | Bacterial infection symptoms |
| 🍄 Fungal | Fungal disease symptoms |
| 🐛 Pest damage | Physical damage from pests |
| 🌿 Shepherd_purse_weeds | Weed contamination |

---

## 📊 Results

| Metric | Value |
|---|---|
| Training Accuracy | 95.84% |
| Validation Accuracy | 93.56% |
| Model Size (TFLite) | 2.4 MB |
| Inference Speed (Pi 5) | ~0.5-1s per image |
| Training Platform | Google Colab (T4 GPU) |
| Deployment Platform | Raspberry Pi 5 |

### Training History
![Training History](results/training_history.png)

---

## 🔍 Key Finding: Domain Gap

One of the most interesting findings of this project was the **domain gap** between dataset images and real-world images:

| Image Source | Healthy Accuracy | Notes |
|---|---|---|
| Dataset images | ~100% | Controlled conditions, close-up leaf shots |
| Google Images | ~0% | Studio product photos, whole lettuce head |
| Real market lettuce | TBD | Tested at lab |

**Conclusion**: The model performs near-perfectly on images similar to its training distribution (close-up leaf photos in field/greenhouse conditions) but struggles with images from different visual domains (studio photography, different angles, different framing). This is a known challenge in agricultural AI deployment and highlights the importance of collecting locally representative training data.

---

## 🏗️ Architecture

```
Input Image (any size)
        ↓
Resize to 224×224, normalize to [0,1]
        ↓
MobileNetV2 (pretrained on ImageNet, frozen)
        ↓
GlobalAveragePooling2D
        ↓
Dropout (0.3)
        ↓
Dense(5, softmax)
        ↓
5 class probabilities
```

**Transfer Learning Strategy**:
- Phase 1: Freeze MobileNetV2 base, train classification head only (15 epochs)
- Phase 2: Unfreeze last 30 layers, fine-tune with lr=1e-5 (10 epochs)

---

## 📁 Repository Structure

```
lettuce-disease-classifier/
├── README.md
├── requirements.txt
├── train/
│   └── train_colab.ipynb       ← Full training notebook (run on Google Colab)
├── classifier/
│   ├── classify.py             ← Terminal inference script
│   ├── gui.py                  ← Tkinter GUI application
│   └── camera_classify.py      ← Live Pi Camera inference
├── model/
│   ├── lettuce_labels.json     ← Class label mapping
│   └── lettuce_model.tflite   ← Trained TFLite model (download separately)
├── results/
│   ├── training_history.png    ← Accuracy/loss curves
│   └── domain_gap_analysis.md ← Domain gap findings
└── docs/
    └── report.md               ← Internship report
```

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/lettuce-disease-classifier.git
cd lettuce-disease-classifier
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run terminal classifier
```bash
python3 classifier/classify.py path/to/your/leaf_image.jpg
```

### 4. Run GUI
```bash
python3 classifier/gui.py
```

### 5. Run live camera (Raspberry Pi only)
```bash
python3 classifier/camera_classify.py
```

---

## 🛠️ Setup (Raspberry Pi)

```bash
# Create project folder
mkdir ~/lettuce_classifier
cd ~/lettuce_classifier

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install tensorflow numpy pillow

# Run classifier
python3 classifier/classify.py sample.jpg
```

---

## 📦 Dataset

- **Source**: [Lettuce Disease Dataset — Kaggle](https://www.kaggle.com/datasets/iqrapervez2000/lettuce-disease-dataset)
- **Total images**: 4,052
- **Split**: 80% training (3,244) / 20% validation (808)
- **Augmentation**: rotation, horizontal flip, zoom, shear, width/height shift

---

## 🔗 References

- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
- [Safe Lettuce — Nvidia Developer Forums](https://forums.developer.nvidia.com/t/safe-lettuce-re-training-resnet-18-on-lettuce-diseases-dataset/295559)
- [Frontiers in Plant Science — Lettuce Weight Estimation](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2022.980581/full)
- [EETAR Laboratory — Chulalongkorn University](https://suwitkiravittaya.eng.chula.ac.th)

---

## 👨‍💻 Author

**Nodri** — EE Undergraduate, Chulalongkorn University  
Internship at EETAR Laboratory under Assoc. Prof. Suwit Kiravittaya  
