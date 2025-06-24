from models.law_module import inverse_scale
from models.law_module import scale_threshold

law_display = {
    "age_over_50": ("Age", 0, 50, "years", "greater"),
    "trestbps_high": ("Resting blood pressure", 1, 140, "mmHg", "greater"),
    "chol_high": ("Cholesterol", 2, 240, "mg/dl", "greater"),
    "thalach_low": ("Max heart rate", 3, 120, "bpm", "less"),
    "sex_male": ("Gender", 5, 1, "male", "equals"),
    "cp_typical_angina": ("Chest pain type", 9, 1, "typical angina", "equals"),
    "restecg_abnormal": ("Resting ECG", 12, 1, "abnormal", "equals")
}

def generate_human_readable_report(model, X, y_true, preprocessor, threshold=0.5):
    model.eval()
    reports = []
    import torch
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        probs, gates = model(X_tensor)
        probs = probs.numpy()
        gates = gates.numpy()
        total_activation = sum(law(X_tensor)[0].sum() for law in model.laws.values())

        for i in range(len(X)):
            x_sample = X_tensor[i:i+1]
            pred_prob = probs[i]
            pred_label = "High risk of CVD" if pred_prob > threshold else "Low risk of CVD"
            true_label = "Has CVD" if y_true[i] == 1 else "No CVD"
            explanation = model.explain(x_sample)
            law_acts = sorted(explanation.items(), key=lambda x: abs(x[1]["weighted_contrib"]), reverse=True)

            report = f"Sample {i+1}:\n"
            report += f"Model prediction: {pred_label} (probability: {pred_prob:.3f})\n"
            report += f"True label: {true_label}\n"
            report += f"Explanation includes uncertainty via law activation levels:\n"

            for idx, (law_key, val) in enumerate(law_acts, 1):
                activation = val["activation"]
                weighted_contrib = val["weighted_contrib"]
                if abs(weighted_contrib) < 0.01:
                    continue
                label, feat_idx, thresh, units, direction = law_display[law_key]
                val_scaled = x_sample[0, feat_idx].item()
                percentage_contribution = (weighted_contrib / total_activation) * 100

                if feat_idx < 4:
                    val_actual = inverse_scale(val_scaled, feat_idx, preprocessor)
                    dir_text = "over" if direction == "greater" else "under" if direction == "less" else "equals"
                    report += (f"  {idx}. {label} (value: {val_actual:.1f} {units}) is {dir_text} the threshold "
                               f"{thresh} {units} with activation {activation:.3f} contributing "
                               f"{weighted_contrib:.3f} ({percentage_contribution:.2f}% of total risk)\n")
                else:
                    report += (f"  {idx}. {label} indicates {units} with activation {activation:.3f} "
                               f"contributing {weighted_contrib:.3f} ({percentage_contribution:.2f}% of total risk)\n")
            report += "\n"
            reports.append(report)
    return reports
