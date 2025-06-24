import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)
    df = df.drop(columns=["id", "dataset", "fbs", "exang", "oldpeak", "slope", "ca", "thal"])
    df["target"] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"])
    df = df.replace("?", np.nan).dropna()

    categorical = ["sex", "cp", "restecg"]
    numerical = ["age", "trestbps", "chol", "thalch"]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numerical),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical)
    ])

    X = preprocessor.fit_transform(df[numerical + categorical])
    y = df["target"].values.astype(np.float32)

    return X, y, preprocessor
