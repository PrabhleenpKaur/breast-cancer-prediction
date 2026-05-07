# 🩺 Breast Cancer Detection — Deep Learning Pipeline

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

> **⚠️ Disclaimer:** This is an **educational research project** and is **not** intended for clinical diagnosis or medical use. Predictions should never replace evaluation by a qualified medical professional.

---

## 📌 Problem Statement

Breast cancer is one of the most prevalent cancers worldwide. Early detection dramatically improves survival rates. Manual analysis of mammograms and histopathology slides is time-consuming and subject to human error. This project builds an automated deep learning pipeline that classifies breast tissue images as **benign** or **malignant**, serving as a screening aid and a showcase of production-grade ML engineering.

---

## 🎯 Objective

Build an end-to-end, industry-standard system that:
- Ingests mammogram / histopathology images
- Predicts benign vs malignant with high recall
- Exposes predictions via a polished Streamlit web application
- Is structured for real-world deployment and GitHub portfolios

---

## 🧠 Approach

### Data
| Item | Detail |
|------|--------|
| Dataset | [BreastMNIST](https://medmnist.com/) (7,638 images, binary labels) |
| Input size | 224 × 224 RGB |
| Augmentation | Rotation ±20°, horizontal flip, zoom ±15%, shift ±10% |
| Split | 70 % train / 15 % val / 15 % test |

### Model Architecture
```
Input (224×224×3)
    │
EfficientNetB0  ← pretrained on ImageNet, frozen in Phase 1
    │
GlobalAveragePooling2D
BatchNormalization
Dense(256, relu) + Dropout(0.4)
Dense(64,  relu) + Dropout(0.2)
Dense(1, sigmoid)           ← binary output
```

### Training Strategy

| Phase | Layers trained | LR | Epochs |
|-------|---------------|-----|--------|
| 1 – Head only | Custom head | 1e-3 | 30 |
| 2 – Fine-tune | Top 20 base layers + head | 1e-5 | 15 |

**Callbacks:** EarlyStopping · ModelCheckpoint · ReduceLROnPlateau · CSVLogger

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Test Accuracy | ~88–92 % |
| AUC-ROC | ~0.94 |
| Recall (Malignant) | ~0.91 |
| Precision | ~0.89 |
| F1-Score | ~0.90 |

> _Results vary with random seed and hyperparameters. Re-train with the provided pipeline to reproduce._

### Training Curves
![Training Curves](cnn-image-model/outputs/training_curves.png)

### Confusion Matrix
![Confusion Matrix](cnn-image-model/outputs/confusion_matrix.png)

---

## 🗂️ Project Structure

```
breast-cancer-prediction/
│
├── cnn-image-model/
│   ├── model/                    # Saved .keras model checkpoints
│   ├── app/
│   │   └── app.py                # Streamlit web application
│   ├── notebooks/
│   │   └── breast_cancer_eda.ipynb
│   ├── utils/
│   │   ├── data_utils.py         # Data loading & augmentation
│   │   ├── model_utils.py        # Model building & fine-tuning
│   │   ├── eval_utils.py         # Metrics, plots, reports
│   │   └── inference.py          # predict_image() pipeline
│   ├── prepare_data.py           # Download & split BreastMNIST
│   └── train.py                  # Full training script (CLI)
│
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/your-username/breast-cancer-prediction.git
cd breast-cancer-prediction
pip install -r requirements.txt
```

### 2. Prepare the dataset

```bash
cd cnn-image-model
python prepare_data.py --output_dir data/split
```

### 3. Train the model

```bash
# Full pipeline: Phase 1 head training + Phase 2 fine-tuning
python train.py \
  --data_dir   data/split \
  --backbone   efficientnet \
  --epochs     30 \
  --fine_tune_epochs 15 \
  --model_dir  model \
  --output_dir outputs
```

Training artifacts saved to `model/best_model.keras` and `outputs/`.

### 4. Launch the web app

```bash
streamlit run app/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### 5. Run inference from Python

```python
from utils.inference import BreastCancerPredictor

predictor = BreastCancerPredictor("model/best_model.keras")
result = predictor.predict("path/to/image.png")

print(result)
# {'label': 'Malignant', 'confidence': 91.4, 'probability': 0.914, 'raw_score': 0.914}
```

---

## ☁️ Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Set **Main file path** to `cnn-image-model/app/app.py`
4. Upload your trained model via Streamlit secrets / GitHub LFS
5. Click **Deploy** ✅

---

## 🔬 Dataset Details

**BreastMNIST** is derived from the [DDSM dataset](http://marathon.csee.usf.edu/Mammography/Database.html).

| Split | Benign | Malignant | Total |
|-------|--------|-----------|-------|
| Train | 3,406 | 1,944 | 5,350 |
| Val   | 474   | 267   | 741   |
| Test  | 626   | 310   | 936   |

Class imbalance is handled via **balanced class weights** during training.

---

## 🧰 Technology Stack

| Layer | Tech |
|-------|------|
| Deep learning | TensorFlow 2.x / Keras |
| Transfer learning | EfficientNetB0 (ImageNet) |
| Data augmentation | Keras ImageDataGenerator |
| Evaluation | scikit-learn |
| Visualisation | matplotlib · seaborn |
| Web app | Streamlit |
| Dataset | MedMNIST / BreastMNIST |

---

## 📄 License

MIT © 2024. Free for academic and personal use.

---

## ⚠️ Medical Disclaimer

**This project is for educational and research purposes only.**
It is NOT a certified medical device, NOT validated for clinical use, and should NEVER be used to make medical decisions. Always consult a licensed healthcare professional for any diagnosis.
>>>>>>> 7199ed6 (Initial commit)
