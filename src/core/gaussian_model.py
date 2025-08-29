import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

from config.config import TrainingConfig
from src.utils.io_utils import IOUtils
from src.utils.math_utils import MathUtils


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
        
        # 优化辅助张量
        self.register_buffer("xyz_gradient_accum", torch.zeros(0, 3))
        self.register_buffer("denom", torch.zeros(0, 1))
        self.register_buffer("max_radii2D", torch.zeros(0))
        
        self.setup_functions()
    def setup_functions(self):
        """设置激活函数和其他函数"""
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.opacity_activation = torch.sigmoid
        self.opacity_inverse_activation = self.inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize
    # ------------------------------ 初始化 ------------------------------
    @torch.no_grad()    
    def create_from_pcd(self, pcd_path: str, spatial_lr_scale: float = 1.0) -> None:
        points, colors = IOUtils.load_pcd(pcd_path)
        
        if points.size == 0:
            raise ValueError("No points found in the PCD file.")
        N = points.shape[0]
        device = self._infer_device()

        xyz = torch.from_numpy(points).float().to(device)

        if colors is None:
            colors = np.ones((N, 3), dtype=np.float32)
        
        colors = torch.from_numpy(colors).float().to(device)

        features_dc = colors[:, None, :].contiguous() #(N, 3) → (N, 1, 3)
        features_rest = torch.zeros(N, 15, 3, device=device) #(N, 15, 3)

        extent = (xyz.max(dim = 0).values - xyz.min(dim = 0).values).mean().item()
        base_scale = 0.01 * max(extent, 1e-2) * spatial_lr_scale
        scaling = torch.full((N, 3), math.log(base_scale), device=device) # 存储 log-sigma
        rot = F.normalize(torch.randn(N, 4, device=device), dim=-1)
        opacity = torch.full((N, 1), 0.5, device=device)

        self._xyz = nn.Parameter(xyz)
        self._features_dc = nn.Parameter(features_dc)
        self._features_rest = nn.Parameter(features_rest)
        self._scaling = nn.Parameter(scaling)
        self._rotation = nn.Parameter(rot)
        self._opacity = nn.Parameter(opacity)
    
        self.xyz_gradient_accum = torch.zeros_like(self.xyz)
        self.denom = torch.ones_like(self._opacity)
        self.max_radii2D = torch.zeros(N, device=device)

    @torch.no_grad()
    def create_from_random(self, num_points: int, scene_extent: float = 1.0) -> None:
        """随机初始化高斯参数"""
        device = self._infer_device()
        xyz = (torch.rand(num_points, 3, device=device) - 0.5) * (2.0 * scene_extent)
        features_dc = torch.rand(num_points, 1, 3, device=device)
        features_rest = torch.zeros(num_points, 15, 3, device=device)
        scaling = torch.full((num_points, 3), math.log(0.02 * scene_extent), device=device)
        rot = F.normalize(torch.randn(num_points, 4, device=device), dim=-1)
        opacity = torch.full((num_points, 1), -2.0, device=device)

        self._xyz = nn.Parameter(xyz)
        self._features_dc = nn.Parameter(features_dc)
        self._features_rest = nn.Parameter(features_rest)
        self._scaling = nn.Parameter(scaling)
        self._rotation = nn.Parameter(rot)
        self._opacity = nn.Parameter(opacity)

        self.xyz_gradient_accum = torch.zeros_like(self._xyz)
        self.denom = torch.zeros_like(self._opacity)
        self.max_radii2D = torch.zeros(num_points, device=device)

    # ------------------------------ 属性访问器 ------------------------------
    @property
    def get_xyz(self) -> torch.Tensor:
        return self._xyz

    @property
    def get_features(self) -> torch.Tensor:
        """获取颜色特征 (DC + 高阶SH) -> [N, 16, 3] (degree<=3)
        注意：本实现的渲染仅使用 DC，SH 评估函数提供接口。"""
        if self._features_rest.numel() == 0:
            return self._features_dc
        return torch.cat([self._features_dc, self._features_rest], dim=1)
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    # ------------------------------ 模型操作 ------------------------------
    @torch.no_grad()
    def density_and_split(self, grad_threshold: float, scene_extent: float) -> None:
        """分裂大高斯（简化：对大且高梯度的点，按主轴一分为二）"""
        if self._xyz.grad is None:
            return
        size = self.get_scaling.mean(dim=-1)
        grad = self._xyz.grad.norm(dim=-1)
        mask = (grad > grad_threshold ) & (size > 0.03 * scene_extent)
        if mask.sum() == 0:
            return
        idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        xyz = self._xyz.data[idx]
        scale = self.get_scaling.data[idx]
        rot = self.get_rotation.data[idx]
        dir_vec = MathUtils.build_rotation_matrix(rot)[:, :, 0] # 主轴x方向
        offset = dir_vec * (scale.mean(dim=-1, keepdim=True) * 0.5)

        # 生成两个子高斯
        new_xyz = torch.cat([xyz - offset, xyz + offset], dim = 0)
        new_feat_dc = self._features_dc.data[idx].repeat(2, 1, 1)
        new_feat_rest = self._features_rest.data[idx].repeat(2, 1, 1)
        new_scale = torch.log(scale * 0.75).repeat(2, 1)
        new_rot = rot.repeat(2, 1)
        new_op = torch.clamp(torch.logit(self.get_opacity.data[idx])[:, 0:1],-6, 6).repeat(2, 1)
        
        self.prune_points(~mask)
        self._append_points(new_xyz, new_feat_dc, new_feat_rest, new_scale, new_rot, new_op)

        
    @torch.no_grad()
    def density_and_clone(self, grad_threshold: float, scene_extent: float) -> None:
        """克隆小高斯（简化：对小且高梯度的点复制一份并略微抖动）"""
        if self._xyz.grad == None:
            return
        size = self.get_scaling.mean(dim = -1)
        grad = self._xyz.grad.norm(dim = -1)
        mask = (grad > grad_threshold) & (size < 0.01 * scene_extent)
        if mask.sum() == 0:
            return
        idx = torch.nonzero(mask, as_tuple=False).squeeze(dim = -1)
        jitter = torch.randn_like(self._xyz[idx]) * (self.get_scaling.data[idx].mean(dim=-1, keepdim=True) * 0.5) 
        new_xyz = self._xyz.data[idx] + jitter
        self._append_points(
            new_xyz,
            self._features_dc.data[idx],
            self._features_rest.data[idx],
            self._scaling.data[idx],
            self._rotation.data[idx],
            self._opacity.data[idx],
        )

    def prune_points(self, mask: torch.Tensor) -> None:
        """根据布尔 mask (保留=mask) 过滤点集并重建参数""" 
        device = self._xyz.device

        def filter(input_tensor: torch.Tensor) -> torch.Tensor:
            if input_tensor.numel() == 0:
                return input_tensor
            return input_tensor[mask]

        self._xyz = nn.Parameter(filter(self._xyz.data))
        self._features_dc = nn.Parameter(filter(self._features_dc.data))
        self._features_rest = nn.Parameter(filter(self._features_rest.data))
        self._scaling = nn.Parameter(filter(self._scaling.data))
        self._rotation = nn.Parameter(filter(self._rotation.data))
        self._opacity = nn.Parameter(filter(self._opacity.data))
        self.register_buffer("xyz_gradient_accum", torch.zeros_like(self._xyz))
        self.register_buffer("denom", torch.zeros(self._xyz.shape[0], 1, device=device))

    # ------------------------------ 工具方法 ------------------------------
    def compute_3d_covariance(self) -> torch.Tensor:
        """计算3D协方差矩阵：R * diag(sigma^2) * R^T -> [N,3,3]"""
        sigma = self.get_scaling
        rot = self.get_rotation
        Rm = MathUtils.build_rotation_matrix(rot)
        D = torch.diag_embed(sigma ** 2)
        cov = Rm @ D @ Rm.transpose(-1, -2)
        return cov
    def get_num_points(self) -> int:
        return int(self._xyz.shape[0])
    @torch.no_grad()    
    def reset_opacity(self, new_opacity: float = 0.01) -> None:
        val = torch.clamp(torch.tensor(new_opacity, device=self._opacity.device), 1e-4, 1-1e-4)
        self._opacity.data[:] = self.inverse_sigmoid(val)
    @staticmethod
    def inverse_sigmoid(x):
        return torch.log(x/(1-x))
    
    # ------------------------------ 内部辅助 ------------------------------
    def _infer_device(self) -> torch.device:
        for p in self.parameters():
            return p.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _append_points(self, xyz: torch.Tensor, fdc: torch.Tensor, frest: torch.Tensor,
                        scaling_log: torch.Tensor, rot: torch.Tensor, op: torch.Tensor) -> None:
        self._xyz = nn.Parameter(torch.cat([self._xyz.data, xyz], dim = 0))
        self._features_dc = nn.Parameter(torch.cat([self._features_dc.data, fdc], dim = 0))
        self._features_rest = nn.Parameter(torch.cat([self._features_rest.data, frest], dim = 0))
        self._scaling = nn.Parameter(torch.cat([self._scaling_log.data, scaling_log], dim = 0))
        self._rotation = nn.Parameter(torch.cat([self._rotation.data, rot], dim = 0))
        self._opacity = nn.Parameter(torch.cat([self._opacity.data, op], dim = 0))

        device = self._xyz.device
        self.xyz_gradient_accum = torch.zeros_like(self._xyz, device = device)
        self.denom = torch.zeros(self._xyz.shape[0], 1, device = device)
        self.max_radii2D = torch.zeros(self._xyz.shape[0], 1, device = device)        
    
        
