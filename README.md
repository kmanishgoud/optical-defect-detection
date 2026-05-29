# Automated Optical Defect Detection System

AAn AI-powered visual inspection system that classifies industrial components
as defective or non-defective using Computer Vision and Machine Learning.

Built as an end-to-end deployable pipeline — from raw image preprocessing
to a live web application — with a research focus on **CNN spatial attention
patterns** and their connection to human visual inspection behaviour.

## Key Research Finding

> The CNN develops **distinct spatial attention strategies per class** —
> focusing on the lower-center region for defective components and the
> upper-center region for OK components. This spatial divergence reflects
> the actual geometry of casting defects in pump impellers.
>
> This raises an open research question: **do these machine attention patterns
> correlate with where a human expert inspector would look at the same images?**
> If so, human eye-tracking data during inspection could be used to supervise
> and improve CNN attention — connecting directly to semantic gaze target
> detection research (Tafasca et al., NeurIPS 2024).
>
> See [`notebooks/attention_analysis.ipynb`](notebooks/attention_analysis.ipynb)
> for the full analysis.

---

## Project Overview

This system replicates the core challenge in manufacturing quality control:
automatically detecting surface defects in industrial components from images,
replacing manual inspection with a fast, consistent, and explainable AI pipeline.

The project bridges two domains:
- **Manufacturing QC** — automated accept/reject classification (relevant to optical inspection in diagnostics manufacturing)
- **Visual Attention Research** — Grad-CAM heatmaps reveal *where* the model focuses, connecting to gaze and attention research in computer vision

---

## Results

| Model | Accuracy | ROC-AUC |
|---|---|---|
| SVM | 100.0% | 1.000 |
| Random Forest | 99.3% | 0.999 |
| Gradient Boosting | 98.7% | 0.999 |
| CNN (TensorFlow) | 93.8% | 0.984 |

---

## Pipeline Architecture
Raw Images (casting components)
↓
Preprocessing (OpenCV)

Grayscale conversion
Gaussian blur + Otsu thresholding
Contour detection
↓
Feature Extraction (26 features)
Statistical: mean, std, skewness, kurtosis
Texture: Laplacian variance, edge density
Morphological: contour count, area
Histogram: 16-bin pixel distribution
↓
┌─────────────────┬──────────────────────┐
│  Classical ML   │    Deep Learning     │
│  Scikit-learn   │    TensorFlow/Keras  │
│  SVM / RF / GB  │    Custom CNN        │
└─────────────────┴──────────────────────┘
↓
Grad-CAM Attention Heatmaps
Visualises where CNN focuses
Connects model decisions to spatial features
↓
Streamlit Web Application
Upload any component image
Get instant defect classification
View attention heatmaps + feature analysis

---

## 🗂️ Dataset

**Real-Life Industrial Casting Product Dataset**  
- 7,348 grayscale images of pump impeller castings
- Binary labels: `ok_front` (non-defective) / `def_front` (defective)
- Train: 6,633 images | Test: 715 images
- Source: [Kaggle](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product)

---

## Tech Stack

- **Python 3.11**
- **OpenCV** — image preprocessing and contour detection
- **Scikit-learn** — SVM, Random Forest, Gradient Boosting
- **TensorFlow/Keras** — CNN architecture
- **Grad-CAM** — visual attention heatmaps
- **Streamlit** — web application
- **Matplotlib/Seaborn** — statistical visualization

---

## Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/kmanishgoud/optical-defect-detection.git
cd optical-defect-detection
```

### 2. Create virtual environment
```bash
py -3.11 -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download dataset
```bash
python -m kaggle datasets download -d ravirajsinh45/real-life-industrial-dataset-of-casting-product -p data/raw --unzip
```

### 5. Run the pipeline
```bash
python src/preprocessing.py   # Preprocess images
python src/features.py        # Extract features
python src/train_ml.py        # Train ML models
python src/train_cnn.py       # Train CNN
python src/gradcam.py         # Generate heatmaps
```

### 6. Launch the app
```bash
streamlit run app.py
```

---

## 📁 Project Structure
optical-defect-detection/
├── src/
│   ├── preprocessing.py   # OpenCV pipeline
│   ├── features.py        # Feature extraction
│   ├── train_ml.py        # Scikit-learn models
│   ├── train_cnn.py       # TensorFlow CNN
│   ├── gradcam.py         # Attention heatmaps
│   └── evaluate.py        # Evaluation utilities
├── app.py                 # Streamlit web app
├── requirements.txt
└── README.md

---

## Grad-CAM Visual Attention

Grad-CAM (Gradient-weighted Class Activation Mapping) visualises which 
regions of an image the CNN uses to make its classification decision.

This connects to active research in visual attention and gaze estimation —
understanding not just *what* a model decides, but *where* it looks to 
decide it.

---

## 👤 Author

**Kamareddy Manish Goud**  
[GitHub](https://github.com/kmanishgoud)