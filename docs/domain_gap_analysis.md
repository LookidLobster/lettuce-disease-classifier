# Domain Gap Analysis
## Lettuce Disease Classifier — Real-World Deployment Evaluation

---

## Overview

After successfully training the model to 93.56% validation accuracy on the benchmark dataset, we evaluated its performance on images from different sources to assess real-world deployability.

---

## Test Results

### Dataset Images (same distribution as training)

| Image | Predicted | Confidence | Correct? |
|---|---|---|---|
| sample_healthy.jpg | Healthy | 100.0% | ✅ |
| sample_bacterial.jpg | Bacterial | 99.3% | ✅ |
| sample_fungal.jpg | Fungal | 98.5% | ✅ |

**Average confidence on dataset images: ~99%**

---

### Real-World Google Images

| Image | Predicted | Confidence | Correct? |
|---|---|---|---|
| healthyLettuce.jpg (studio) | Fungal | 63.4% | ❌ |
| fungalLettuce.jpg | Fungal | 90.5% | ✅ |

**Healthy class score on all Google images: 0.0%**

---

## Root Cause Analysis

Visual inspection of the images revealed a clear explanation:

### Dataset Images (what the model was trained on):
- Close-up photos of **individual leaves** still attached to the plant
- Natural greenhouse/field environment (soil, stems visible in background)
- Varied lighting, slightly blurry, handheld camera style
- Consistent framing: leaf fills most of the frame

### Google Images (what we tested with):
- **Whole lettuce head** photographed from above
- White studio background, professional product photography
- Perfect lighting, high resolution, commercial quality
- Very different visual structure from training data

These two image types look **completely different** to a neural network even though both represent "healthy lettuce" to a human eye.

---

## Key Finding

> The model performs near-perfectly (98-100%) on images from the same distribution as the training data, but performance degrades significantly on images from different visual domains — particularly the Healthy class, which scored 0% on studio-style whole-head photographs.

This is a well-documented phenomenon in machine learning called **domain gap** or **distribution shift** — the difference between the statistical properties of training data and real-world deployment data.

---

## Implications

1. **For deployment**: The model should be used with images that match the training data style — close-up leaf photos taken in natural/greenhouse conditions, not commercial photography.

2. **For improvement**: Collecting locally representative images (Thai lettuce varieties, local greenhouse conditions, consistent camera angle/distance) and retraining would significantly improve real-world performance.

3. **For future work**: Fine-tuning on a small set of locally collected images (even 50-100 per class) could bridge much of the domain gap through transfer learning.

---

## Recommendation

Before deployment in a real Thai plant factory or agricultural setting:
- Collect 100-200 images per class from the actual deployment environment
- Use consistent camera setup (fixed distance, controlled lighting)
- Fine-tune the existing model on this local dataset
- Re-evaluate accuracy on a held-out local test set

This would likely improve Healthy class accuracy from ~0% on real-world images to near-dataset-level performance.

---

*Analysis conducted during internship at EETAR Laboratory, Chulalongkorn University*
*Supervisor: Assoc. Prof. Suwit Kiravittaya*
