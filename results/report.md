# Internship Report
## AI-Powered Lettuce Disease Classifier on Raspberry Pi

**Student**: Nodri  
**Supervisor**: Assoc. Prof. Suwit Kiravittaya  
**Laboratory**: EETAR (Electrical Engineering Technology for Agricultural Resources)  
**Department**: Electrical Engineering, Chulalongkorn University  
**Duration**: 4 weeks  

---

## 1. Introduction

This report documents the development of an AI-powered lettuce disease classification system deployed on a Raspberry Pi 5. The project was motivated by the EETAR laboratory's ongoing research in smart plant factory technology and precision agriculture, with the goal of developing a practical, low-cost tool for automated plant health monitoring.

Thailand's agricultural sector increasingly relies on controlled-environment agriculture (smart farms, plant factories, hydroponic systems), creating demand for automated monitoring tools that can detect plant diseases early without requiring specialized expertise from farmers.

---

## 2. Objectives

1. Train a deep learning model to classify lettuce leaf health status from images
2. Deploy the model efficiently on Raspberry Pi 5 for edge inference
3. Evaluate real-world performance and identify deployment challenges
4. Document findings for future research and improvement

---

## 3. Methodology

### 3.1 Dataset

- **Source**: Lettuce Disease Dataset (Kaggle)
- **Total images**: 4,052
- **Classes**: Bacterial, Fungal, Healthy, Pest damage, Shepherd_purse_weeds
- **Split**: 80% training (3,244 images) / 20% validation (808 images)

### 3.2 Model Architecture

Transfer learning approach using MobileNetV2:

```
Input (224×224×3)
    ↓
MobileNetV2 backbone (pretrained on ImageNet, ~3.4M parameters)
    ↓
GlobalAveragePooling2D
    ↓
Dropout (rate=0.3)
    ↓
Dense (5 units, softmax activation)
    ↓
Output: 5 class probabilities
```

MobileNetV2 was chosen over ResNet for deployment efficiency — its depthwise separable convolutions make it significantly faster on CPU-only devices like the Raspberry Pi.

### 3.3 Training Strategy

**Phase 1 — Classification head training:**
- Freeze all MobileNetV2 layers
- Train only the new Dense layer
- Learning rate: 0.001, Epochs: 15 (early stopping at epoch 10)
- Result: 92.70% validation accuracy

**Phase 2 — Fine-tuning:**
- Unfreeze last 30 layers of MobileNetV2
- Train with reduced learning rate: 1e-5
- Epochs: 10 (best at epoch 7)
- Result: 93.56% validation accuracy

**Data augmentation** applied during training:
- Random rotation (±20°)
- Horizontal flip
- Zoom range (±20%)
- Width/height shift (±20%)
- Shear transformation

### 3.4 Deployment

- Model converted to TFLite format with INT8 quantization
- Final model size: 2.4 MB (vs ~14MB+ for full Keras model)
- Inference time on Pi 5: ~0.5-1 second per image
- Runtime: TensorFlow 2.21.0 (tflite-runtime unavailable for Python 3.13)

---

## 4. Results

### 4.1 Training Performance

| Metric | Phase 1 | Phase 2 (Final) |
|---|---|---|
| Training Accuracy | 95.84% | 95.93% |
| Validation Accuracy | 92.70% | **93.56%** |
| Training Loss | 0.1030 | 0.1063 |
| Validation Loss | 0.2139 | 0.2372 |

### 4.2 Per-Sample Inference Results

**Dataset images (same distribution as training):**

| Sample | Predicted | Confidence | Result |
|---|---|---|---|
| sample_healthy.jpg | Healthy | 100.0% | ✅ |
| sample_bacterial.jpg | Bacterial | 99.3% | ✅ |
| sample_fungal.jpg | Fungal | 98.5% | ✅ |

**Real-world images (Google Images):**

| Sample | Predicted | Confidence | Result |
|---|---|---|---|
| healthyLettuce.jpg | Fungal | 63.4% | ❌ |
| fungalLettuce.jpg | Fungal | 90.5% | ✅ |

---

## 5. Key Finding: Domain Gap

The most significant finding of this project was the identification of a substantial **domain gap** between dataset images and real-world images.

### Root Cause

Visual inspection revealed that:
- **Dataset images**: close-up leaf photos in natural greenhouse conditions
- **Google Images tested**: whole lettuce head, white studio background, commercial photography

These visually distinct image styles cause the model to perform near-perfectly on dataset-style images but poorly on images from different visual domains — particularly the Healthy class, which scored 0% confidence on studio-style photographs.

### Significance

This finding has important implications for real-world deployment:
1. Image acquisition protocol matters as much as model accuracy
2. Locally collected Thai lettuce images would significantly improve real-world performance
3. A standardized camera setup (fixed distance, angle, lighting) is essential for consistent results

---

## 6. Discussion

### What worked well
- Transfer learning with MobileNetV2 achieved strong accuracy (93.56%) efficiently
- TFLite quantization successfully reduced model size by ~85% with minimal accuracy loss
- Two-phase training (freeze → fine-tune) provided meaningful accuracy improvement
- Deployment on Pi 5 was smooth with ~1 second inference time

### Challenges encountered
- `tflite-runtime` package incompatible with Python 3.13 on Raspberry Pi OS (trixie) — resolved by using full TensorFlow
- Network instability on Pi's built-in WiFi during large package downloads — resolved via USB phone tethering
- Domain gap between training dataset and real-world images — identified and documented as a research finding

### Limitations
- Dataset consists of hydroponic lettuce from controlled environments — may not represent Thai field conditions
- No real Thai lettuce images available for testing during this internship period
- Live camera inference not tested due to equipment constraints at time of writing

---

## 7. Conclusion

This project successfully developed and deployed an AI-powered lettuce disease classifier on Raspberry Pi 5, achieving 93.56% validation accuracy with a lightweight 2.4MB TFLite model capable of ~1 second inference on edge hardware.

The identification of the domain gap between controlled dataset conditions and real-world deployment scenarios is the most valuable finding — it provides a clear roadmap for future improvement: collecting locally representative Thai lettuce images and fine-tuning the existing model would likely bridge most of the performance gap.

This work demonstrates the viability of deploying computer vision AI on low-cost edge devices for agricultural applications, directly supporting the EETAR laboratory's broader mission of developing practical AI tools for Thai smart farming systems.

---

## 8. Future Work

1. **Local dataset collection**: Photograph Thai lettuce varieties in actual greenhouse/farm conditions with consistent camera setup
2. **Fine-tuning**: Retrain model on locally collected images to close domain gap
3. **Live camera integration**: Complete the real-time inference pipeline with Pi Camera
4. **Multi-task model**: Add growth stage classification alongside disease detection
5. **IoT integration**: Connect to MQTT broker, log results to dashboard over time
6. **Mobile deployment**: Convert to Android/iOS app for farmer smartphone use

---

## References

1. Sandler, M. et al. "MobileNetV2: Inverted Residuals and Linear Bottlenecks." CVPR 2018.
2. Zhang, X. et al. "Lettuce Fresh Weight Estimation Using RGB-D Images." Frontiers in Plant Science, 2022.
3. Kaggle: Lettuce Disease Dataset — iqrapervez2000
4. Nvidia Developer Forums: Safe Lettuce Project — xxzhang17, 2024
5. Kiravittaya, S. et al. "Smart Indoor Cultivation with AIThaiGen IoT Platform." ECTI-CON 2025.
