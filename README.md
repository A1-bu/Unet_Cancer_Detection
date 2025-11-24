# **UNet for Multi-Class Breast Cancer Ultrasound Segmentation**

![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch\&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.9+-3776ab?logo=python\&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![W\&B](https://img.shields.io/badge/Weights%20&%20Biases-Logged-yellow?logo=weightsandbiases\&logoColor=white)

A complete deep-learning pipeline for **multi-class semantic segmentation** of breast cancer ultrasound images using a custom **U-Net** architecture.
The model performs pixel-level classification of **background**, **benign**, and **malignant** tumor regions.

---

## **Table of Contents**

* [Overview](#overview)
* [Dataset](#dataset)
* [Features](#features)
* [Project Structure](#project-structure)
* [Installation](#installation)
* [Training](#training)
* [Visualization](#visualization)
* [Model Architecture](#model-architecture)
* [Results](#results)
* [WandB Integration](#wandb-integration)
* [Future Work](#future-work)
* [License](#license)

---

## **Overview**

This project implements a high-quality **U-Net segmentation model** for breast ultrasound tumor detection. The pipeline includes:

* Custom dataset loader
* Class-weighted training
* On-the-fly real-time visualization
* Evaluation with Dice score per class
* Optional Weights & Biases logging
* Prediction visualization utilities

---

## **Dataset**

This project uses the **BUSI (Breast Ultrasound Images)** dataset, containing:

| Class | Meaning             |
| ----- | ------------------- |
| **0** | Background / Normal |
| **1** | Benign Tumor        |
| **2** | Malignant Tumor     |

Each sample includes an ultrasound image and a segmentation mask.
Dataset is automatically downloaded using:

```bash
pip install wldhx.yadisk-direct
curl -L $(yadisk-direct <URL>) -o data.zip
unzip data.zip
```

---

## **Features**

* ✔ Custom **MultiClassBUSIDataset** loader
* ✔ Stratified train/val split
* ✔ Albumentations-based augmentation
* ✔ Full **U-Net encoder–decoder architecture**
* ✔ Class-weighted cross-entropy
* ✔ Dice score for each class
* ✔ Real-time visual training dashboard
* ✔ Prediction visualization
* ✔ W&B logging (optional)

---

## **Project Structure**

```
├── data/                       # BUSI dataset
├── unet_model.py               # UNet architecture
├── dataset.py                  # Custom dataset loader
├── train.py                    # Training loop
├── visualize.py                # Visualization utilities
├── notebook.ipynb             # Colab notebook version
└── README.md
```

---

## **Installation**

Clone the repo:

```bash
git clone https://github.com/your-username/unet-breast-cancer-segmentation.git
cd unet-breast-cancer-segmentation
```

Install dependencies:

```bash
pip install -r requirements.txt
```

(If you need a generated `requirements.txt`, I can create one.)

---

## **Training**

Train the model:

```python
from unet_model import Unet
from train import train_multiclass_unet

model = Unet(in_channels=1, out_channels=3)

history = train_multiclass_unet(
    model,
    train_loader,
    val_loader,
    num_epochs=100,
    learning_rate=1e-4,
    class_weights=class_weights,
)
```

---

## **Visualization**

### **During Training**

The script displays:

* Loss curves
* LR schedule
* Dice per class
* Live predictions

### **After Training**

Generate side-by-side predictions:

```python
from visualize import visualize_multiclass_prediction
visualize_multiclass_prediction(model, val_loader, num_samples=20)
```

---

## **Model Architecture**

A standard U-Net with:

* Contracting encoder
* Bottleneck (1024 filters)
* Expanding decoder with skip connections
* Final 1×1 conv → 3-class mask

Diagram available upon request.

---

## 📈 **Results**

Typical performance on BUSI:

| Class      | Dice Score |
| ---------- | ---------- |
| Background | ~0.97      |
| Benign     | ~0.42      |
| Malignant  | ~0.22      |

(The malignant class is significantly smaller and harder to segment.)

If you want, I can generate a **results.png** template for your repo.

---

## **WandB Integration**

Enable logging:

```python
run = wandb.init(project="unet-segmentation")
```

Logs include:

* Loss curves
* Dice curves
* Prediction grids
* Learning rate schedule

Example run from your notebook:
`project: unet-segmentation, run: su430bz4`

---

## **Future Work**

* Add **Focal Loss**
* Try **DeepLabV3+** for comparison
* Add test-time augmentation
* Improve malignant segmentation with region priors
* Export model to ONNX for deployment

---

## **License**

This project is licensed under the **MIT License**.

---
