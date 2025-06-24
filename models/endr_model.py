import torch
import torch.nn as nn
from .base_model import BaseModel
from .law_module import LawModule, scale_threshold

class eNDRModel(nn.Module):
    def __init__(self, input_dim, preprocessor):
        super().__init__()
        self.base = BaseModel(input_dim)
        self.laws = nn.ModuleDict({
            "age_over_50": LawModule(0, scale_threshold(50, 0, preprocessor), "greater"),
            "trestbps_high": LawModule(1, scale_threshold(140, 1, preprocessor), "greater"),
            "chol_high": LawModule(2, scale_threshold(240, 2, preprocessor), "greater"),
            "thalach_low": LawModule(3, scale_threshold(120, 3, preprocessor), "less"),
            "sex_male": LawModule(5, 0.5, "equals"),
            "cp_typical_angina": LawModule(9, 0.5, "equals"),
            "restecg_abnormal": LawModule(12, 0.5, "equals"),
        })
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        base_out = self.base(x)
        law_contribs = []
        law_gates = []
        for law in self.laws.values():
            weighted_contrib, gate = law(x)
            law_contribs.append(weighted_contrib)
            law_gates.append(gate)
        law_sum = torch.stack(law_contribs, dim=0).sum(dim=0)
        out = base_out + law_sum
        prob = self.sigmoid(out)
        return prob, torch.stack(law_gates, dim=0)

    def explain(self, x):
        self.eval()
        with torch.no_grad():
            explanations = {}
            for name, law in self.laws.items():
                weighted_contrib, gate = law(x)
                explanations[name] = {
                    "weighted_contrib": weighted_contrib.item(),
                    "activation": gate.item()
                }
            return explanations
