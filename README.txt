## eNDR: Extended Nomological Deductive Reasoning for Heart Disease Risk Prediction

This repository contains a PyTorch implementation of the **eNDR model**, a hybrid deep learning architecture that combines neural networks with NDR-based interpretable decision rules to predict cardiovascular disease (CVD) risk based on the UCI Heart Disease dataset. The core result is a self-explainable neural network with domain-grounded law-based explanation for each prediction. 

![GitHub repo size](https://img.shields.io/github/repo-size/HakizimanaGedeon/eNDR-heart-risk-model)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c?logo=pytorch)
![License](https://img.shields.io/github/license/HakizimanaGedeon/eNDR-heart-risk-model)

---

## Features

- Deep Neural Network (DNN) with BatchNorm and Dropout
- Learnable medical rule modules (LawModules) for explainability
- Confidence-aware prediction calibration
- Threshold optimization for best classification performance
- Human-readable risk explanation reports
- Model saving & loading with `.pth` files
- Ready for PyTorch Lightning conversion

---

## Model Overview

The **eNDR architecture** consists of:

- A base DNN trained on clinical features
- A set of **Law Modules** that encode interpretable medical rules (e.g., "age > 50", "cholesterol > 240")
- A calibrated output combining learned model predictions and rule activations

> Example Rule:
> - **Age > 50** → increases risk based on learned weight and confidence level (sigmoid gate)

---

## Dataset

- **Source**: [UCI Heart Disease Dataset (Cleveland subset)](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
- **Features used**:
  - Numerical: `age`, `trestbps`, `chol`, `thalach`
  - Categorical: `sex`, `cp`, `restecg`
  - Target: Binary indicator of presence of heart disease

---

## Results (on validation set)

| Metric        | Score    |
|---------------|----------|
| Accuracy      | 0.83     |
| Precision     | 0.85     |
| Recall        | 0.86     |
| F1-score      | 0.86     |
| Brier Score   | 0.1292   |
| Optimal Threshold | 0.636 (TPR: 87.8%, FPR: 21.6%) |

**Learned Law Weights:**
age_over_50 : weight = 0.7791
trestbps_high : weight = 0.7935
chol_high : weight = 0.8178
thalach_low : weight = 0.7665
sex_male : weight = 0.8077
cp_typical_angina : weight = 0.8184
restecg_abnormal : weight = 0.7672


---

## Installation

```bash
# Clone the repo
git clone https://github.com/HakizimanaGedeon/eNDR-heart-risk-model.git
cd eNDR-heart-risk-model

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

---
## Usage
Train the model: 
python train.py

Generate human-readable risk reports:
python explain.py

The report is saved to:
human_readable_cvd_reports.txt (In your working directory)

---
## Project structure
eNDR-heart-risk-model/
├── train.py                     # Main training loop
├── explain.py                   # Generates human-readable explanations
├── model.py                     # Model definition: BaseModel + LawModules + eNDR
├── utils.py                     # Helper functions: scaling, evaluation, thresholding
├── requirements.txt             # Python dependencies
├── heart_disease_uci.csv        # UCI dataset (cleaned)
├── human_readable_cvd_reports.txt
└── README.md

---
## Example Explanation (Report)
Sample 1:
Model prediction: High risk of CVD (probability: 0.783)
True label: Has CVD
Explanation includes uncertainty via law activation levels:
  1. Cholesterol (value: 260.3 mg/dl) is over the threshold 240 with activation 0.923 contributing 0.754 (24.8% of total risk)
  2. Age (value: 63.4 years) is over the threshold 50 with activation 0.879 contributing 0.683 (22.5% of total risk)

---
## Citation
@misc{endr2025,
  author = {Hakizimana Gedeon},
  title = {eNDR: Explainable Deep Neural Rules for Heart Disease Risk},
  year = {2025},
  url = {https://github.com/HakizimanaGedeon/eNDR-heart-risk-model}
}

---

## Contributing
Pull requests are welcome! If you spot an issue or want to extend the framework (e.g. PyTorch Lightning, GUI), feel free to open an issue or PR.
