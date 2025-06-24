import torch
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
import numpy as np

from utils.preprocess import load_and_preprocess
from lightning.lightning_module import eNDRLightningModule

def main():
    X, y, preprocessor = load_and_preprocess("data/heart_disease_uci.csv")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    model = eNDRLightningModule(X.shape[1], preprocessor)

    trainer = pl.Trainer(max_epochs=30, accelerator='auto')
    trainer.fit(model, train_loader, val_loader)

    # Save model weights & preprocessor for inference
    torch.save(model.model.state_dict(), "endr_model_weights.pth")
    import joblib
    joblib.dump(preprocessor, "preprocessor.joblib")

if __name__ == "__main__":
    main()
