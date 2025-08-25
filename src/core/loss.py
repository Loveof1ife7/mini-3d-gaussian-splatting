from torch import nn
from types import Tuple, Dict
import torch

from config.config import TrainingConfig
from core.gaussian_model import GaussianModel

class SSIMLoss(nn.Module):
    def __init__(self, window_size: int = 11, size_average: bool = True):
        pass
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pass
        
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
        pass
    
    

        