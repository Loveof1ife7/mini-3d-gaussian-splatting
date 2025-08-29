import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

class MathUtils:
    """数学工具"""
    @staticmethod
    def build_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
        """四元数转旋转矩阵: quaternion: [N, 4] -> [N, 3, 3]; q=[w,x,y,z]"""
        q = F.normalize(quaternion, dim=-1)\
        
        w, x, y, z = q.unbind(-1)

        ww, xx, yy, zz = w*w, x*x, y*y, z*z 
        wx, wy, wz = w*x, w*y, w*z
        xy, xz, yz = x*y, x*z, y*z

        R = torch.stack([
            1-2*(yy+zz), 2*(xy-wz), 2*(xz+wy),
            2*(xy+wz), 1-2*(xx+zz), 2*(yz-wx),
            2*(xz-wy), 2*(yz+wx), 1-2*(xx+yy)
        ], dim = -1).view(-1, 3, 3)
        
        return R
    
    @staticmethod
    def build_covariance_3d(scaling: torch.Tensor, 
                           rotation: torch.Tensor) -> torch.Tensor:
        """构建3D协方差矩阵"""
        R = MathUtils.build_rotation_matrix(rotation)
        D = torch.diag_embed(scaling ** 2)
        return R @ D @ R.transpose(-1, -2)
    
    @staticmethod
    def project_covariance_2d(cov3d: torch.Tensor,
                             viewmatrix: torch.Tensor,
                             projmatrix: torch.Tensor) -> torch.Tensor:
        """投影3D协方差到2D"""
        var = cov3d.diagonal(dim1=-2, dim2=-1).mean(dim=-1) #[N]
        return torch.diag_embed(torch.stack[var, var],dim=-1) #[N,2,2]
    
    @staticmethod
    def spherical_harmonics_eval(degrees: int,
                                dirs: torch.Tensor,
                                coeffs: torch.Tensor) -> torch.Tensor:
        """球谐函数求值 (极简): 仅 DC（degree=0）返回 coeffs[:,0]"""
        return coeffs[:,0]
        