from torch import nn
from typing import Tuple, Dict
import torch
from torch.nn import functional as F

from config.config import TrainingConfig
from src.core.gaussian_model import GaussianModel

class SSIMLoss(nn.Module):
    def __init__(self, window_size: int = 11, size_average: bool = True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:        
        K = self.window_size
        pad = K // 2
        device = pred.device
        x = torch.arange(K, device=device).float() - (K - 1)/2
        g1d = torch.exp(-x**2 / (2 * (K/6) **2)) # sigma /approx K/6
        g1d = (g1d / g1d.sum()).view(1, 1, K)
        
        def blur(img: torch.Tensor) -> torch.Tensor:
            out = F.conv2d(img, g1d.unsqueeze(3), padding=(0, pad), groups=img.shape[1])
            out = F.conv2d(out, g1d.unsqueeze(2), padding=(pad, 0), groups=img.shape[1])
            return out

        mu_x = blur(pred)
        mu_y = blur(target)
        
        sigma_x = blur(pred**2) - mu_x**2
        sigma_y = blur(target**2) - mu_y**2
        sigma_xy = blur(pred * target) - mu_x * mu_y
        
        ssim = ((2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)) / (
       (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2))
        ssim = ssim.clamp(0, 1)

class GaussianLoss(nn.Module):
    def __init__(self, lambda_dssim: float = 0.2):
        super().__init__()
        self.lambda_dssim = lambda_dssim
        self.ssim_loss = SSIMLoss()    
    def forward(self,
                rendered: torch.Tensor,
                target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        """
        计算总损失 = (1-λ)*L1 + λ*D-SSIM
        
        Returns:
            (total_loss, loss_dict)
        """
        l1 = (rendered - target).abs().mean()
        dssim = self.ssim_loss(rendered, target)
        total_loss = (1 - self.lambda_dssim) * l1 + self.lambda_dssim * dssim
        return total_loss, {
            "l1": float(l1.detach().cpu()),
            "dssim": float(dssim.detach().cpu()),
            "total_loss": float(total_loss.detach().cpu())
            }
    
    
if __name__ == "__main__":
    ssim_loss = SSIMLoss()
    gaussian_loss = GaussianLoss()
    pred = torch.randn(1, 3, 256, 256) # batch=1, RGB, 32x32
    target = torch.randn(1, 3, 256, 256)
    print("=== Running SSIMLoss ===")
    loss_val = ssim_loss(pred, target)
    print("SSIMLoss output:", loss_val)

    print("\n=== Running GaussianLoss ===")
    total, logs = gaussian_loss(pred, target)
    print("GaussianLoss total:", total)
    print("Details:", logs)
        