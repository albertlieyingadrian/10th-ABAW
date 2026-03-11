import torch
from torch import nn
import torch.nn.functional as F
    
    
class CCC_MSE_Loss(nn.Module):
    def __init__(self, alpha=0.8):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        pred = pred.float()
        target = target.float()

        pred_mean = torch.mean(pred)
        target_mean = torch.mean(target)

        cov = torch.mean((pred - pred_mean) * (target - target_mean))
        pred_var = torch.var(pred)
        target_var = torch.var(target)

        ccc = (2 * cov) / (pred_var + target_var + (pred_mean - target_mean)**2 + 1e-8)
        ccc = ccc.float()

        ccc_loss = 1 - ccc
        mse_loss = self.mse(pred, target)

        return self.alpha * ccc_loss + (1 - self.alpha) * mse_loss