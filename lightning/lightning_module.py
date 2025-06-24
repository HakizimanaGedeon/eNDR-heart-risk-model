import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torchmetrics import Accuracy
from models.endr_model import eNDRModel
from utils.metrics import tune_threshold

class eNDRLightningModule(pl.LightningModule):
    def __init__(self, input_dim, preprocessor, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = eNDRModel(input_dim, preprocessor)
        self.criterion = torch.nn.BCELoss()
        self.lr = lr
        self.accuracy = Accuracy()
        self.best_val_acc = 0.0

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds, _ = self.model(x)
        loss = self.criterion(preds, y)
        self.log('train_loss', loss, prog_bar=True)
        preds_labels = (preds > 0.5).float()
        acc = self.accuracy(preds_labels, y.int())
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds, _ = self.model(x)
        loss = self.criterion(preds, y)
        preds_labels = (preds > 0.5).float()
        acc = self.accuracy(preds_labels, y.int())
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return {"val_loss": loss, "val_acc": acc, "preds": preds, "targets": y}

    def validation_epoch_end(self, outputs):
        preds = torch.cat([x["preds"] for x in outputs]).cpu().numpy()
        targets = torch.cat([x["targets"] for x in outputs]).cpu().numpy()
        threshold, tpr, fpr = tune_threshold(targets, preds)
        self.log("optimal_threshold", threshold)
        # Save threshold for later use if needed
        self.optimal_threshold = threshold

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
