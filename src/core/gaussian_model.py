
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

from config.config import TrainingConfig

class GaussianModel(nn.Module):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.max_sh_degree = 3
        
        self._xyz = nn.Parameter(torch.empty(0, 3))           # 位置[N, 3]
        self._features_dc = nn.Parameter(torch.empty(0, 1, 3))  # DC分量[N, 1, 3]
        self._features_rest = nn.Parameter(torch.empty(0, 15, 3))  # 高阶SH分量 [N, 15, 3]
        self._scaling = nn.Parameter(torch.empty(0, 3))       # 缩放 [N, 3]
        self._rotation = nn.Parameter(torch.empty(0, 4))      # 旋转(四元数) [N, 4]
        self._opacity = nn.Parameter(torch.empty(0, 1))       # 不透明度 [N, 1]
        
        # 优化相关属性
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        
        self.setup_functions()
        
    @staticmethod
    def inverse_sigmoid(x):
        return torch.log(x/(1-x))
    def setup_functions(self):
        """设置激活函数和其他函数"""
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.opacity_activation = torch.sigmoid
        self.opacity_inverse_activation = self.inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize
        
    def create_from_pcd(self, pcd_path: str, spatial_lr_scale: float = 1.0) -> None:
        pass
    def load_ply(self, path: str):
        pass

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def density_and_split(self, grad_threshold: float, scene_extent: float) -> None:
        pass
    
    def density_and_clone(self, grad_threshold: float, scene_extent: float) -> None:
        pass
    
    def prune_points(self, mask: torch.Tensor) -> None:
        pass    
    
    def compute_3d_covariance(self) -> torch.Tensor:
        pass
    
    def get_num_points(self) -> int:
        return self._xyz.shape[0]
    
    def reset_opacity(self) -> None:
        self._opacity.data.fill_(0.0)   
        
        
