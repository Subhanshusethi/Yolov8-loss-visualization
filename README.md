
# 📦 Bounding Box Regression Losses – From Scratch

This repository contains a complete training pipeline to analyze and compare different **bounding box regression loss functions** in the context of object detection. The primary focus is on understanding how loss functions influence training dynamics when regressing perturbed YOLO-format bounding boxes back to their ground truth counterparts.

## 🔍 Key Features

- **Losses Implemented from Scratch**:
  - MAE (L1 Loss)
  - IoU Loss
  - Full SIoU Loss (Angle, Distance, Shape components)
  
- **Perturbation Generator**:
  - Applies controlled noise to ground truth boxes
  - Visualizes ground-truth vs perturbed boxes

- **Simple Neural Regressor**:
  - MLP-based model that learns to correct noisy bounding boxes

- **Training & Validation Loops**:
  - Tracks predictions across epochs
  - Saves best model based on validation loss

- **Interactive Visualization Tool**:
  - Scroll through epochs and watch how predictions improve over time

---

## 🗂️ Repository Structure

```

.
├── datagen.py              # Perturb YOLO labels + visualize GT & noisy boxes
├── dataset.py              # Custom Dataset to load GT and perturbed boxes
├── simple\_model.py         # MLP-based regression model
├── losses.py               # MAE, IoU, SIoU loss functions implemented from scratch
├── main.py                 # Full training pipeline with logging
├── evaluate.py             # Evaluate model using IoU, precision, recall
├── visulaization.py        # Track predicted boxes across epochs using OpenCV

````

---


## 🚀 Quick Start

### 1. Prepare Data

Place your YOLO-format dataset under `archive/coco128/labels/train2017` and images under `archive/coco128/images/train2017`.

Run:

```bash
python datagen.py
```

This will generate perturbed labels and annotated image overlays.

---

### 2. Train Model

```bash
python main.py
```

Trains the model using `Siou_loss_full()` and logs predictions per epoch.

---

### 3. Evaluate

```bash
python evaluate.py
```

Prints IoU, accuracy, precision, and recall over the validation set.

---

### 4. Visualize Bounding Box Evolution

```bash
python visulaization.py
```

Use the OpenCV trackbar to scroll through epochs and observe how predicted boxes get refined.

---

## 📈 Example: Loss Functions Compared

| Loss Function | Spatial Awareness      | Gradient Stability | Accuracy |
| ------------- | ---------------------- | ------------------ | -------- |
| MAE           | ❌                      | ✅                  | Low      |
| IoU           | ✅                      | ❌ (no overlap)     | Medium   |
| SIoU          | ✅ (Angle, Dist, Shape) | ✅                  | High     |

---
