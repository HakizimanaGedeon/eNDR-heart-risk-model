import torch
import torch.nn as nn

def scale_threshold(orig_val, feature_idx, preprocessor):
    scaler = preprocessor.named_transformers_['num']
    mean = scaler.mean_[feature_idx]
    scale = scaler.scale_[feature_idx]
    return (orig_val - mean) / scale

def inverse_scale(scaled_val, feature_idx, preprocessor):
    scaler = preprocessor.named_transformers_['num']
    mean = scaler.mean_[feature_idx]
    scale = scaler.scale_[feature_idx]
    return scaled_val * scale + mean

class LawModule(nn.Module):
    def __init__(self, index, threshold, direction="greater"):
        super().__init__()
        self.index = index
        self.threshold = threshold
        self.direction = direction
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        val = x[:, self.index]
        if self.direction == "greater":
            gate = torch.sigmoid(10 * (val - self.threshold))
        elif self.direction == "less":
            gate = torch.sigmoid(10 * (self.threshold - val))
        elif self.direction == "equals":
            gate = torch.sigmoid(10 * (val - self.threshold))
        else:
            raise ValueError("Unknown direction")
        return self.weight * gate, gate
