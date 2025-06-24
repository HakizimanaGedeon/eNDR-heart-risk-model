import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, brier_score_loss
import joblib
import numpy as np

from models.endr_model import eNDRModel
from utils.metrics import tune_threshold

def main():
    preprocessor = joblib.load("preprocessor.joblib")
    model = eNDRModel(input_dim=15, preprocessor=preprocessor)
    model.load_state_dict(torch.load("endr_model_weights.pth"))
    model.eval()

    # Load test data, example: reuse train for demo, adjust path and split for real eval
    import pandas as pd
    df = pd.read_csv("data/heart_disease_uci.csv")
    df = df.drop(columns=["id", "dataset", "fbs", "exang", "oldpeak", "slope", "ca", "thal"])
    df["target"] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"]).replace("?", np.nan).dropna()

    numerical = ["age", "trestbps", "chol", "thalch"]
    categorical = ["sex", "cp", "restecg"]
    X = preprocessor.transform(df[numerical + categorical])
    y_true = df["target"].values

    X_tensor = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        y_probs, _ = model(X_tensor)

    y_probs = y_probs.numpy()
    threshold, tpr, fpr = tune_threshold(y_true, y_probs)
    y_pred = (y_probs > threshold).astype(int)

    print(f"Optimal Threshold: {threshold:.3f} (TPR: {tpr:.3f}, FPR: {fpr:.3f})")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    print(f"Brier Score: {brier_score_loss(y_true, y_probs):.4f}")

if __name__ == "__main__":
    main()
