import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(temperature))

    def forward(self, feat1, feat2):
        labels = torch.arange(feat1.shape[0], device=feat1.device, dtype=torch.int64)
        logits = self.logit_scale.exp() * F.normalize(feat1, dim=1) @ F.normalize(feat2, dim=1).T
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        return loss
