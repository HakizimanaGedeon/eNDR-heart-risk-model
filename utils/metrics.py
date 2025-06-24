import numpy as np
from sklearn.metrics import brier_score_loss, classification_report, roc_curve

def tune_threshold(y_true, y_probs):
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return thresholds[best_idx], tpr[best_idx], fpr[best_idx]

def evaluate_classification(y_true, y_pred):
    return classification_report(y_true, y_pred)

def compute_brier_score(y_true, y_probs):
    return brier_score_loss(y_true, y_probs)
