# =============================================================================
# Mini 3D Gaussian Splatting (educational reference implementation)
# Focused on file organization, class design, and method signatures
# This is a compact, single-file implementation matching your architecture.
# It favors clarity over performance and completeness.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path

import math
import random
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore

# =============================================================================
# config/config.py - 配置管理
# =============================================================================

@dataclass
class TrainingConfig:
    """训练配置类"""
    # 路径配置
    data_path: str = "./data"
    output_path: str = "./output"

    # 训练参数
    iterations: int = 500
    learning_rate: float = 1e-2
    batch_size: int = 1

    # 高斯参数学习率
    position_lr_init: float = 1e-3
    position_lr_final: float = 1e-4
    feature_lr: float = 1e-2
    opacity_lr: float = 5e-3
    scaling_lr: float = 3e-3
    rotation_lr: float = 3e-3

    # 密度控制参数
    densify_from_iter: int = 100
    densify_until_iter: int = 400
    densify_grad_threshold: float = 1e-3
    densify_interval: int = 50

    # 渲染参数
    image_height: int = 480
    image_width: int = 640

    # 系统配置
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ConfigManager:
    """配置管理器"""

    @staticmethod
    def load_from_yaml(config_path: str) -> TrainingConfig:
        """从YAML文件加载配置"""
        if yaml is None:
            raise ImportError("PyYAML not installed. pip install pyyaml")
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return TrainingConfig(**data)

    @staticmethod
    def save_to_yaml(config: TrainingConfig, config_path: str) -> None:
        """保存配置到YAML文件"""
        if yaml is None:
            raise ImportError("PyYAML not installed. pip install pyyaml")
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump({k: getattr(config, k) for k in config.__dataclass_fields__.keys()}, f, allow_unicode=True)

    @staticmethod
    def get_default_config() -> TrainingConfig:
        """获取默认配置"""
        return TrainingConfig()


# =============================================================================
# src/core/gaussian_model.py - 3D高斯模型核心
# =============================================================================

class GaussianModel(nn.Module):
    """3D高斯模型核心类 (简化版)"""

    def __init__(self, max_sh_degree: int = 3):
        super().__init__()
        self.max_sh_degree = max_sh_degree

        # 初始化为空参数；调用 create_* 方法后填充
        self._xyz: nn.Parameter = nn.Parameter(torch.empty(0, 3))
        self._features_dc: nn.Parameter = nn.Parameter(torch.empty(0, 1, 3))
        self._features_rest: nn.Parameter = nn.Parameter(torch.empty(0, 15, 3))
        self._scaling: nn.Parameter = nn.Parameter(torch.empty(0, 3))
        self._rotation: nn.Parameter = nn.Parameter(torch.empty(0, 4))
        self._opacity: nn.Parameter = nn.Parameter(torch.empty(0, 1))

        # 优化辅助张量
        self.register_buffer("xyz_gradient_accum", torch.zeros(0, 3))
        self.register_buffer("denom", torch.zeros(0, 1))
        self.register_buffer("max_radii2D", torch.zeros(0))

    # ------------------------------ 初始化 ------------------------------
    @torch.no_grad()
    def create_from_pcd(self, pcd_path: str, spatial_lr_scale: float = 1.0) -> None:
        """从点云初始化高斯参数
        - 支持 COLMAP points3D.txt 或 (N,3)/(N,6) 的 .npz/.npy
        """
        points, colors = IOUtils.load_point_cloud(pcd_path)
        if points.size == 0:
            raise ValueError(f"Empty point cloud: {pcd_path}")
        N = points.shape[0]
        device = self._infer_device()

        xyz = torch.from_numpy(points).float().to(device)
        if colors is None:
            colors = np.ones((N, 3), dtype=np.float32)
        colors = torch.from_numpy(colors).float().to(device).clamp(0, 1)

        # DC 初始化为颜色的 RGB，rest 为零
        features_dc = colors[:, None, :].contiguous()
        features_rest = torch.zeros(N, 15, 3, device=device)

        # 缩放、旋转、透明度初始化
        extent = (xyz.max(dim=0).values - xyz.min(dim=0).values).mean().item()
        base_scale = 0.01 * max(extent, 1e-2) * spatial_lr_scale
        scaling = torch.full((N, 3), math.log(base_scale), device=device)  # 存储 log-sigma
        rot = F.normalize(torch.randn(N, 4, device=device), dim=-1)
        opacity = torch.full((N, 1), -2.0, device=device)  # sigmoid(-2) ~ 0.119

        # 赋值为 Parameter
        self._xyz = nn.Parameter(xyz)
        self._features_dc = nn.Parameter(features_dc)
        self._features_rest = nn.Parameter(features_rest)
        self._scaling = nn.Parameter(scaling)
        self._rotation = nn.Parameter(rot)
        self._opacity = nn.Parameter(opacity)

        # Buffers
        self.xyz_gradient_accum = torch.zeros_like(self._xyz)
        self.denom = torch.zeros_like(self._opacity)
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
    def get_scaling(self) -> torch.Tensor:
        """获取缩放 (经过exp激活) 作为 sigma (标准差)"""
        return torch.exp(self._scaling)

    @property
    def get_rotation(self) -> torch.Tensor:
        """获取旋转 (归一化四元数)"""
        return F.normalize(self._rotation, dim=-1)

    @property
    def get_opacity(self) -> torch.Tensor:
        """获取不透明度 (经过sigmoid激活)"""
        return torch.sigmoid(self._opacity)

    # ------------------------------ 模型操作 ------------------------------
    @torch.no_grad()
    def densify_and_split(self, grad_threshold: float, scene_extent: float) -> None:
        """分裂大高斯（简化：对大且高梯度的点，按主轴一分为二）"""
        if self._xyz.grad is None:
            return
        size = self.get_scaling().mean(dim=-1)  # [N]
        grad = self._xyz.grad.norm(dim=-1)
        mask = (grad > grad_threshold) & (size > 0.03 * scene_extent)
        if mask.sum() == 0:
            return
        idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        xyz = self._xyz.data[idx]
        scale = self.get_scaling().data[idx]
        rot = self.get_rotation().data[idx]
        dir_vec = MathUtils.build_rotation_matrix(rot)[:, :, 0]  # 主轴x方向
        offset = (scale.mean(dim=-1, keepdim=True) * 0.5) * dir_vec

        # 生成两个子高斯
        new_xyz = torch.cat([xyz - offset, xyz + offset], dim=0)
        new_feat_dc = self._features_dc.data[idx].repeat(2, 1, 1)
        new_feat_rest = self._features_rest.data[idx].repeat(2, 1, 1)
        new_scale = torch.log(scale * 0.75).repeat(2, 1)
        new_rot = rot.repeat(2, 1)
        new_op = torch.clamp(torch.logit(self.get_opacity().data[idx])[:, 0:1], -6, 6).repeat(2, 1)

        self._append_points(new_xyz, new_feat_dc, new_feat_rest, new_scale, new_rot, new_op)
        self.prune_points(~mask)  # 移除被分裂的原点

    @torch.no_grad()
    def densify_and_clone(self, grad_threshold: float, scene_extent: float) -> None:
        """克隆小高斯（简化：对小且高梯度的点复制一份并略微抖动）"""
        if self._xyz.grad is None:
            return
        size = self.get_scaling().mean(dim=-1)
        grad = self._xyz.grad.norm(dim=-1)
        mask = (grad > grad_threshold) & (size < 0.01 * scene_extent)
        if mask.sum() == 0:
            return
        idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        jitter = torch.randn_like(self._xyz.data[idx]) * (self.get_scaling().data[idx].mean(dim=-1, keepdim=True) * 0.5)
        new_xyz = self._xyz.data[idx] + jitter
        self._append_points(
            new_xyz,
            self._features_dc.data[idx],
            self._features_rest.data[idx],
            self._scaling.data[idx],
            self._rotation.data[idx],
            self._opacity.data[idx],
        )

    @torch.no_grad()
    def prune_points(self, mask: torch.Tensor) -> None:
        """根据布尔 mask (保留=mask) 过滤点集并重建参数"""
        device = self._xyz.device
        def filt(p: torch.Tensor) -> torch.Tensor:
            if p.numel() == 0:
                return p
            return p[mask]

        self._xyz = nn.Parameter(filt(self._xyz.data))
        self._features_dc = nn.Parameter(filt(self._features_dc.data))
        self._features_rest = nn.Parameter(filt(self._features_rest.data))
        self._scaling = nn.Parameter(filt(self._scaling.data))
        self._rotation = nn.Parameter(filt(self._rotation.data))
        self._opacity = nn.Parameter(filt(self._opacity.data))
        self.register_buffer("xyz_gradient_accum", torch.zeros_like(self._xyz))
        self.register_buffer("denom", torch.zeros(self._xyz.shape[0], 1, device=device))
        self.register_buffer("max_radii2D", torch.zeros(self._xyz.shape[0], device=device))

    # ------------------------------ 工具方法 ------------------------------
    def compute_3d_covariance(self) -> torch.Tensor:
        """计算3D协方差矩阵：R * diag(sigma^2) * R^T -> [N,3,3]"""
        sigma = self.get_scaling()
        rot = self.get_rotation()
        Rm = MathUtils.build_rotation_matrix(rot)
        D = torch.diag_embed(sigma ** 2)
        cov = Rm @ D @ Rm.transpose(-1, -2)
        return cov

    def get_num_points(self) -> int:
        return int(self._xyz.shape[0])

    @torch.no_grad()
    def reset_opacity(self, new_opacity: float = 0.01) -> None:
        val = torch.clamp(torch.tensor(new_opacity, device=self._opacity.device), 1e-4, 1-1e-4)
        self._opacity.data[:] = torch.log(val/(1-val))

    # ------------------------------ 内部辅助 ------------------------------
    def _infer_device(self) -> torch.device:
        for p in self.parameters():
            return p.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _append_points(self, xyz: torch.Tensor, fdc: torch.Tensor, frest: torch.Tensor,
                        scaling_log: torch.Tensor, rot: torch.Tensor, op: torch.Tensor) -> None:
        self._xyz = nn.Parameter(torch.cat([self._xyz.data, xyz], dim=0))
        self._features_dc = nn.Parameter(torch.cat([self._features_dc.data, fdc], dim=0))
        self._features_rest = nn.Parameter(torch.cat([self._features_rest.data, frest], dim=0))
        self._scaling = nn.Parameter(torch.cat([self._scaling.data, scaling_log], dim=0))
        self._rotation = nn.Parameter(torch.cat([self._rotation.data, rot], dim=0))
        self._opacity = nn.Parameter(torch.cat([self._opacity.data, op], dim=0))
        device = self._xyz.device
        self.xyz_gradient_accum = torch.zeros_like(self._xyz)
        self.denom = torch.zeros(self._xyz.shape[0], 1, device=device)
        self.max_radii2D = torch.zeros(self._xyz.shape[0], device=device)


# =============================================================================
# src/core/camera.py - 相机模型
# =============================================================================

class Camera:
    """相机类 (COLMAP 约定: X_cam = R * X_world + T)"""

    def __init__(self,
                 uid: int,
                 R: np.ndarray,
                 T: np.ndarray,
                 FoVx: float,
                 FoVy: float,
                 image: torch.Tensor,
                 image_name: str,
                 width: int,
                 height: int):
        self.uid = uid
        self._R = torch.from_numpy(R).float()
        self._T = torch.from_numpy(T).float()
        self.FoVx = float(FoVx)
        self.FoVy = float(FoVy)
        self.image = image  # [3,H,W] in [0,1]
        self.image_name = image_name
        self.width = width
        self.height = height
        self._wv: Optional[torch.Tensor] = None
        self._proj: Optional[torch.Tensor] = None

    # 属性
    @property
    def world_view_transform(self) -> torch.Tensor:
        if self._wv is None:
            self._wv = CameraUtils.build_world_view_matrix(self._R.cpu().numpy(), self._T.cpu().numpy())
        return self._wv

    @property
    def projection_matrix(self) -> torch.Tensor:
        if self._proj is None:
            self._proj = CameraUtils.build_projection_matrix(0.01, 1000.0, self.FoVx, self.FoVy)
        return self._proj

    @property
    def full_proj_transform(self) -> torch.Tensor:
        return self.projection_matrix @ self.world_view_transform

    @property
    def camera_center(self) -> torch.Tensor:
        # C = -R^T T
        R = self._R
        T = self._T
        return (-R.t() @ T)


class CameraUtils:
    """相机工具类"""

    @staticmethod
    def build_world_view_matrix(R: np.ndarray, T: np.ndarray) -> torch.Tensor:
        M = np.eye(4, dtype=np.float32)
        M[:3, :3] = R
        M[:3, 3] = T
        return torch.from_numpy(M)

    @staticmethod
    def build_projection_matrix(znear: float, zfar: float, fovX: float, fovY: float) -> torch.Tensor:
        # 标准 OpenGL-like 透视投影（NDC [-1,1]）
        tanHalfFovX = math.tan(fovX * 0.5)
        tanHalfFovY = math.tan(fovY * 0.5)
        A = 1.0 / tanHalfFovX
        B = 1.0 / tanHalfFovY
        C = -(zfar + znear) / (zfar - znear)
        D = -(2 * zfar * znear) / (zfar - znear)
        P = torch.tensor([[A, 0, 0, 0],
                          [0, B, 0, 0],
                          [0, 0, C, D],
                          [0, 0, -1, 0]], dtype=torch.float32)
        return P

    @staticmethod
    def project_points_3d_to_2d(points_3d: torch.Tensor, camera: Camera) -> Tuple[torch.Tensor, torch.Tensor]:
        """将3D点投影到2D屏幕空间 -> (xy_pix [N,2], depth_cam [N])"""
        N = points_3d.shape[0]
        ones = torch.ones((N, 1), device=points_3d.device)
        homo = torch.cat([points_3d, ones], dim=1)  # [N,4]
        M = camera.full_proj_transform.to(points_3d.device)  # [4,4]
        clip = (M @ homo.t()).t()  # [N,4]
        # 透视除法 -> NDC
        ndc = clip[:, :3] / clip[:, 3:4].clamp(min=1e-8)
        # 像素坐标（原点左上角）
        x = (ndc[:, 0] * 0.5 + 0.5) * camera.width
        y = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * camera.height
        xy = torch.stack([x, y], dim=-1)
        depth = clip[:, 2]  # 使用裁剪空间Z(或相机Z亦可)
        return xy, depth


# =============================================================================
# src/core/renderer.py - 可微分渲染器 (朴素栅格化)
# =============================================================================

@dataclass
class RenderSettings:
    image_height: int
    image_width: int
    bg_color: torch.Tensor
    scale_modifier: float = 1.0
    debug: bool = False


class GaussianRenderer:
    """3D高斯可微分渲染器 (简化版：逐点高斯 Splat，前向 Alpha 混合)"""

    def __init__(self):
        pass

    def render(self, camera: Camera, gaussians: GaussianModel, settings: RenderSettings) -> Dict[str, torch.Tensor]:
        device = gaussians.get_xyz.device
        H, W = settings.image_height, settings.image_width
        bg = settings.bg_color.to(device).view(3, 1, 1)

        # 投影
        means2D, depths = self._project_gaussians_3d_to_2d(gaussians, camera)
        # 朴素半径估计：屏幕像素尺度 ~ sigma * f (使用 FoV)；这里简化为固定比例
        sigmas = gaussians.get_scaling().mean(dim=-1) * settings.scale_modifier
        radii = (sigmas * max(H, W)).clamp(1.0, 50.0)

        # 视锥剔除
        vis = self._frustum_culling(means2D, depths, radii, settings)
        idx = torch.nonzero(vis, as_tuple=False).squeeze(-1)

        if idx.numel() == 0:
            return {
                "image": bg.repeat(1, H, W),
                "alpha": torch.zeros(1, H, W, device=device),
                "depth": torch.zeros(1, H, W, device=device),
                "viewspace_points": means2D,
                "visibility_filter": vis,
                "radii": radii,
            }

        # 深度排序（前到后）
        order = self._sort_gaussians_by_depth(idx, depths)

        # 颜色（仅 DC 分量）
        colors_dc = gaussians._features_dc.squeeze(1).sigmoid()  # [N,3]
        opacities = gaussians.get_opacity().squeeze(1)  # [N]

        out_rgb = bg.expand(3, H, W).clone()
        out_a = torch.zeros(1, H, W, device=device)
        out_d = torch.zeros(1, H, W, device=device)

        # 逐高斯 splat (小窗口高斯核)
        for i in order.tolist():
            cx, cy = means2D[i]
            r = int(radii[i].item())
            if r <= 0:
                continue
            x0 = int(torch.clamp(cx - r, 0, W - 1).item())
            x1 = int(torch.clamp(cx + r + 1, 0, W).item())
            y0 = int(torch.clamp(cy - r, 0, H - 1).item())
            y1 = int(torch.clamp(cy + r + 1, 0, H).item())
            if x1 <= x0 or y1 <= y0:
                continue

            # 网格
            xs = torch.arange(x0, x1, device=device).float()
            ys = torch.arange(y0, y1, device=device).float()
            grid_x, grid_y = torch.meshgrid(xs, ys, indexing='xy')
            dx2 = (grid_x - cx) ** 2
            dy2 = (grid_y - cy) ** 2
            sigma = max(1.0, r / 2.0)
            gauss = torch.exp(-(dx2 + dy2) / (2 * sigma * sigma))  # [w,h]
            a = (opacities[i] * gauss).clamp(0, 1)

            # Alpha 合成 (over)
            patch_rgb = out_rgb[:, y0:y1, x0:x1]
            patch_a = out_a[:, y0:y1, x0:x1]
            a3 = a.unsqueeze(0)
            color = colors_dc[i].view(3, 1, 1)
            trans = (1 - patch_a)
            contrib = a3 * trans
            patch_rgb += contrib * color
            patch_a += a3
            out_d[:, y0:y1, x0:x1] += contrib * depths[i].clamp(min=0).view(1, 1, 1)
            out_rgb[:, y0:y1, x0:x1] = patch_rgb
            out_a[:, y0:y1, x0:x1] = patch_a.clamp(0, 1)

        # 避免除零
        out_d = out_d / (out_a + 1e-6)

        return {
            "image": out_rgb.clamp(0, 1),
            "alpha": out_a.clamp(0, 1),
            "depth": out_d,
            "viewspace_points": means2D,
            "visibility_filter": vis,
            "radii": radii,
        }

    def _project_gaussians_3d_to_2d(self, gaussians: GaussianModel, camera: Camera) -> Tuple[torch.Tensor, torch.Tensor]:
        means2D, depths = CameraUtils.project_points_3d_to_2d(gaussians.get_xyz, camera)
        return means2D, depths

    def _frustum_culling(self, means2D: torch.Tensor, depths: torch.Tensor, radii: torch.Tensor, settings: RenderSettings) -> torch.Tensor:
        H, W = settings.image_height, settings.image_width
        x, y = means2D[:, 0], means2D[:, 1]
        vis = (
            (depths < 0)  # 在相机前（裁剪空间 z<0, 取决于矩阵；这里按上面投影）
            & (x >= -radii) & (x < W + radii)
            & (y >= -radii) & (y < H + radii)
            & (radii > 0)
        )
        return vis

    def _sort_gaussians_by_depth(self, indices: torch.Tensor, depths: torch.Tensor) -> torch.Tensor:
        d = depths[indices]
        order = torch.argsort(d)  # 小的在前（更近）
        return indices[order]


# =============================================================================
# src/core/loss.py - 损失函数
# =============================================================================

class SSIMLoss(nn.Module):
    """简单的 SSIM 实现"""

    def __init__(self, window_size: int = 11, size_average: bool = True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred/target: [B,3,H,W]
        K = self.window_size
        pad = K // 2
        # 生成分离高斯核
        device = pred.device
        x = torch.arange(K, device=device).float() - (K - 1) / 2
        g1d = torch.exp(-(x ** 2) / (2 * (K/6) ** 2))
        g1d = (g1d / g1d.sum()).view(1, 1, K)
        # 深度可分离卷积
        def blur(img: torch.Tensor) -> torch.Tensor:
            out = F.conv2d(img, g1d.unsqueeze(3), padding=(0, pad), groups=img.shape[1])
            out = F.conv2d(out, g1d.unsqueeze(2), padding=(pad, 0), groups=img.shape[1])
            return out

        mu_x = blur(pred)
        mu_y = blur(target)
        sigma_x = blur(pred * pred) - mu_x * mu_x
        sigma_y = blur(target * target) - mu_y * mu_y
        sigma_xy = blur(pred * target) - mu_x * mu_y

        ssim = ((2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)) / ((mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2))
        ssim = ssim.clamp(0, 1)
        dssim = (1 - ssim) / 2
        if self.size_average:
            return dssim.mean()
        else:
            return dssim.mean(dim=[1, 2, 3])


class GaussianLoss(nn.Module):
    """3DGS总损失函数: (1-λ)*L1 + λ*D-SSIM"""

    def __init__(self, lambda_dssim: float = 0.2):
        super().__init__()
        self.lambda_dssim = lambda_dssim
        self.ssim_loss = SSIMLoss()

    def forward(self, rendered: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        l1 = (rendered - target).abs().mean()
        dssim = self.ssim_loss(rendered, target)
        total = (1 - self.lambda_dssim) * l1 + self.lambda_dssim * dssim
        return total, {"l1": float(l1.detach().cpu()), "dssim": float(dssim.detach().cpu()), "total": float(total.detach().cpu())}


# =============================================================================
# src/core/optimizer.py - 优化器和密度控制
# =============================================================================

class LearningRateScheduler:
    """Cosine decay with warmup-delay (like NeRF)"""

    def __init__(self, lr_init: float, lr_final: float, lr_delay_steps: int, lr_delay_mult: float, max_steps: int):
        self.lr_init = lr_init
        self.lr_final = lr_final
        self.lr_delay_steps = lr_delay_steps
        self.lr_delay_mult = lr_delay_mult
        self.max_steps = max_steps

    def get_lr(self, step: int) -> float:
        if self.max_steps <= 0:
            return self.lr_final
        t = min(step, self.max_steps) / self.max_steps
        cos_decay = 0.5 * (1 + math.cos(math.pi * t))
        lr = self.lr_final + (self.lr_init - self.lr_final) * cos_decay
        if self.lr_delay_steps > 0:
            delay_rate = self.lr_delay_mult + (1 - self.lr_delay_mult) * min(step / self.lr_delay_steps, 1)
            lr *= delay_rate
        return float(lr)


class DensityController:
    """密度自适应控制器 (简化版)"""

    def __init__(self, config: TrainingConfig):
        self.config = config

    def should_densify(self, iteration: int) -> bool:
        return (self.config.densify_from_iter <= iteration <= self.config.densify_until_iter) and \
               (iteration % self.config.densify_interval == 0)

    @torch.no_grad()
    def densify_and_prune(self, gaussians: GaussianModel, optimizer: torch.optim.Optimizer, iteration: int, scene_extent: float) -> None:
        grad_th = self.config.densify_grad_threshold
        gaussians.densify_and_split(grad_th, scene_extent)
        gaussians.densify_and_clone(grad_th, scene_extent)

        # 修剪低不透明度
        keep = (gaussians.get_opacity().squeeze(1) > 0.01)
        if keep.numel() > 0:
            gaussians.prune_points(keep)

        # 重建优化器参数组
        if isinstance(optimizer, torch.optim.Adam):
            # 直接清空并重建
            pass

    def _compute_gradient_mask(self, gaussians: GaussianModel, grad_threshold: float) -> torch.Tensor:
        if gaussians._xyz.grad is None:
            return torch.zeros(gaussians.get_num_points(), dtype=torch.bool, device=gaussians.get_xyz.device)
        return gaussians._xyz.grad.norm(dim=-1) > grad_threshold

    def _get_size_mask(self, gaussians: GaussianModel, percent_dense: float, scene_extent: float) -> torch.Tensor:
        size = gaussians.get_scaling().mean(dim=-1)
        th = torch.quantile(size, torch.tensor(percent_dense, device=size.device))
        return size > th


class GaussianOptimizer:
    """3DGS优化器包装类"""

    def __init__(self, gaussians: GaussianModel, config: TrainingConfig):
        self.gaussians = gaussians
        self.config = config
        self.optimizer: Optional[torch.optim.Adam] = None
        self.lr_scheduler_pos = LearningRateScheduler(config.position_lr_init, config.position_lr_final, 0, 1.0, config.iterations)
        self.density_controller = DensityController(config)

    def setup_optimizer(self) -> None:
        params = [
            {"params": [self.gaussians._xyz], "lr": self.config.position_lr_init},
            {"params": [self.gaussians._features_dc, self.gaussians._features_rest], "lr": self.config.feature_lr},
            {"params": [self.gaussians._opacity], "lr": self.config.opacity_lr},
            {"params": [self.gaussians._scaling], "lr": self.config.scaling_lr},
            {"params": [self.gaussians._rotation], "lr": self.config.rotation_lr},
        ]
        self.optimizer = torch.optim.Adam(params)

    def step(self) -> None:
        assert self.optimizer is not None
        self.optimizer.step()

    def zero_grad(self) -> None:
        assert self.optimizer is not None
        self.optimizer.zero_grad(set_to_none=True)

    def update_learning_rate(self, iteration: int) -> None:
        assert self.optimizer is not None
        # 仅更新 xyz 学习率（示范）
        lr = self.lr_scheduler_pos.get_lr(iteration)
        self.optimizer.param_groups[0]["lr"] = lr

    def densify_and_prune(self, iteration: int, scene_extent: float) -> None:
        if self.density_controller.should_densify(iteration):
            assert self.optimizer is not None
            self.density_controller.densify_and_prune(self.gaussians, self.optimizer, iteration, scene_extent)
            # 重新设置优化器（参数数量改变）
            self.setup_optimizer()

    def reset_opacity(self) -> None:
        self.gaussians.reset_opacity()


# =============================================================================
# src/data/dataset.py - 数据集管理 (COLMAP 简化读取)
# =============================================================================

class CameraDataset:
    """相机数据集基类"""

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.cameras: List[Camera] = []
        self.train_cameras: List[Camera] = []
        self.test_cameras: List[Camera] = []

    def load_cameras(self) -> None:
        raise NotImplementedError

    def split_train_test(self, test_ratio: float = 0.1) -> None:
        n = len(self.cameras)
        k = max(1, int(n * test_ratio))
        random.shuffle(self.cameras)
        self.test_cameras = self.cameras[:k]
        self.train_cameras = self.cameras[k:]
        if not self.train_cameras:  # fallback
            self.train_cameras = self.test_cameras

    def get_train_cameras(self) -> List[Camera]:
        return self.train_cameras

    def get_test_cameras(self) -> List[Camera]:
        return self.test_cameras

    def get_scene_info(self) -> Dict[str, Any]:
        pcd = self.get_point_cloud_path()
        pts, _ = IOUtils.load_point_cloud(pcd)
        if pts.size == 0:
            center = np.zeros(3)
            extent = 1.0
        else:
            center = pts.mean(axis=0)
            extent = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)) / 2.0 + 1e-3)
        return {"center": center, "extent": extent}

    def get_point_cloud_path(self) -> str:
        raise NotImplementedError


class COLMAPDataset(CameraDataset):
    """COLMAP数据集"""

    def __init__(self, data_path: str):
        super().__init__(data_path)

    def load_cameras(self) -> None:
        cameras_txt = self._read_cameras_txt()
        images = self._read_images_txt()
        H = int(cameras_txt.get("height", 480))
        W = int(cameras_txt.get("width", 640))
        fx = float(cameras_txt.get("fx", max(W, H)))
        fy = float(cameras_txt.get("fy", max(W, H)))
        FoVx = 2 * math.atan(W / (2 * fx))
        FoVy = 2 * math.atan(H / (2 * fy))

        img_dir = self.data_path / "images"
        cmap: List[Camera] = []
        for uid, meta in images.items():
            qw, qx, qy, qz = meta["qvec"]
            tx, ty, tz = meta["tvec"]
            R = MathUtils.quaternion_to_matrix_np(np.array([qw, qx, qy, qz], dtype=np.float32))
            T = np.array([tx, ty, tz], dtype=np.float32)
            img_path = img_dir / meta["name"]
            img = IOUtils.load_image(str(img_path)) if Image is not None else torch.zeros(3, H, W)
            cam = Camera(uid=uid, R=R, T=T, FoVx=FoVx, FoVy=FoVy, image=img, image_name=meta["name"], width=W, height=H)
            cmap.append(cam)
        self.cameras = cmap

    def _read_cameras_txt(self) -> Dict[str, Any]:
        path = self.data_path / "sparse/0/cameras.txt"
        if not path.exists():
            return {}
        content = [l.strip() for l in path.read_text(encoding="utf-8", errors="ignore").splitlines() if l and not l.startswith("#")]
        # 仅处理常见 SIMPLE_PINHOLE 或 PINHOLE
        out: Dict[str, Any] = {}
        for line in content:
            parts = line.split()
            if len(parts) < 5:
                continue
            # CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS...
            model = parts[1]
            W = int(parts[2]); H = int(parts[3])
            out["width"], out["height"] = W, H
            params = list(map(float, parts[4:]))
            if model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "OPENCV", "OPENCV_FISHEYE", "PINHOLE"}:
                if model == "PINHOLE":
                    fx, fy, cx, cy = params[:4]
                else:
                    fx = fy = params[0]
                    cx = params[1]; cy = params[2]
                out["fx"], out["fy"], out["cx"], out["cy"] = fx, fy, cx, cy
            break
        return out

    def _read_images_txt(self) -> Dict[int, Dict[str, Any]]:
        path = self.data_path / "sparse/0/images.txt"
        if not path.exists():
            return {}
        lines = [l.strip() for l in path.read_text(encoding="utf-8", errors="ignore").splitlines() if l]
        data: Dict[int, Dict[str, Any]] = {}
        i = 0
        while i < len(lines):
            if lines[i].startswith("#"):
                i += 1
                continue
            parts = lines[i].split()
            if len(parts) < 10:
                i += 1
                continue
            img_id = int(parts[0])
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            # camera_id = int(parts[8])
            name = parts[9]
            data[img_id] = {"qvec": (qw, qx, qy, qz), "tvec": (tx, ty, tz), "name": name}
            i += 2  # 下一行是 2D-3D 对应，跳过
        return data

    def _read_points3d_txt(self) -> np.ndarray:
        path = self.data_path / "sparse/0/points3D.txt"
        if not path.exists():
            return np.zeros((0, 3), dtype=np.float32)
        points = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line or line.startswith("#"):
                    continue
                parts = line.strip().split()
                if len(parts) < 7:
                    continue
                x, y, z = map(float, parts[1:4])
                points.append([x, y, z])
        return np.asarray(points, dtype=np.float32)

    def get_point_cloud_path(self) -> str:
        # 优先返回 points3D.txt；若不存在可返回预处理的 npz/npy
        p = self.data_path / "sparse/0/points3D.txt"
        if p.exists():
            return str(p)
        alt_npz = self.data_path / "points.npz"
        alt_npy = self.data_path / "points.npy"
        if alt_npz.exists():
            return str(alt_npz)
        if alt_npy.exists():
            return str(alt_npy)
        return str(p)  # 默认


# =============================================================================
# src/training/trainer.py - 训练器主类
# =============================================================================

class GaussianTrainer:
    """3DGS训练器"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.dataset: CameraDataset = COLMAPDataset(config.data_path)
        self.gaussians: GaussianModel = GaussianModel(max_sh_degree=3).to(config.device)
        self.renderer: GaussianRenderer = GaussianRenderer()
        self.optimizer: GaussianOptimizer = GaussianOptimizer(self.gaussians, config)
        self.loss_fn: GaussianLoss = GaussianLoss()
        self.iteration: int = 0
        self.scene_extent: float = 1.0
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

    def setup(self) -> None:
        # 1. 初始化数据集
        self.dataset.load_cameras()
        self.dataset.split_train_test(0.1)

        # 2. 从点云初始化高斯模型
        pcd_path = self.dataset.get_point_cloud_path()
        try:
            self.gaussians.create_from_pcd(pcd_path)
        except Exception:
            # 退化：随机初始化
            info = self.dataset.get_scene_info()
            self.gaussians.create_from_random(10_000, scene_extent=float(info.get("extent", 1.0)))

        # 3. 设置渲染器（已创建）

        # 4. 设置优化器和损失函数
        self.optimizer.setup_optimizer()

        # 5. 计算场景范围
        info = self.dataset.get_scene_info()
        self.scene_extent = float(info.get("extent", 1.0))

    def train(self) -> None:
        device = self.config.device
        H, W = self.config.image_height, self.config.image_width
        train_cams = self.dataset.get_train_cameras()
        if not train_cams:
            raise RuntimeError("No training cameras loaded.")

        for it in range(self.config.iterations):
            self.iteration = it
            cam = random.choice(train_cams)
            # 若相机图像分辨率与配置不一致，调整渲染输出尺寸
            settings = RenderSettings(image_height=cam.height, image_width=cam.width, bg_color=torch.tensor([0.0, 0.0, 0.0], device=device))
            loss_dict = self.train_step(cam, settings)

            # 密度控制
            self.optimizer.densify_and_prune(self.iteration, self.scene_extent)

            # 日志与检查点
            if it % 50 == 0:
                print(f"Iter {it}: total={loss_dict['total']:.4f} L1={loss_dict['l1']:.4f} DSSIM={loss_dict['dssim']:.4f}")
                self.save_checkpoint(it)

    def train_step(self, camera: Camera, settings: RenderSettings) -> Dict[str, float]:
        self.optimizer.zero_grad()
        out = self.renderer.render(camera, self.gaussians, settings)
        pred = out["image"].unsqueeze(0)  # [1,3,H,W]
        target = camera.image.to(pred.device).unsqueeze(0)
        total, loss_dict = self.loss_fn(pred, target)
        total.backward()
        self.optimizer.update_learning_rate(self.iteration)
        self.optimizer.step()
        self.train_losses.append(loss_dict["total"])
        return loss_dict

    def validate(self) -> Dict[str, float]:
        cams = self.dataset.get_test_cameras()
        if not cams:
            return {"psnr": 0.0, "l1": 0.0}
        with torch.no_grad():
            l1s = []
            for cam in cams:
                settings = RenderSettings(image_height=cam.height, image_width=cam.width, bg_color=torch.tensor([0.0, 0.0, 0.0], device=self.config.device))
                out = self.renderer.render(cam, self.gaussians, settings)
                l1 = (out["image"] - cam.image.to(out["image"].device)).abs().mean().item()
                l1s.append(l1)
            return {"l1": float(np.mean(l1s))}

    def save_checkpoint(self, iteration: int) -> None:
        ckpt_dir = Path(self.config.output_path)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"ckpt_{iteration:06d}.pt"
        torch.save({
            "iter": iteration,
            "gaussians": self.gaussians.state_dict(),
        }, path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        data = torch.load(checkpoint_path, map_location=self.config.device)
        self.gaussians.load_state_dict(data["gaussians"])  # type: ignore
        self.iteration = int(data.get("iter", 0))

    def get_scene_extent(self) -> float:
        return float(self.scene_extent)


# =============================================================================
# src/utils/ - 工具类
# =============================================================================

class MathUtils:
    """数学工具"""

    @staticmethod
    def quaternion_to_matrix_np(q: np.ndarray) -> np.ndarray:
        # q = [w,x,y,z]
        w, x, y, z = q
        Nq = w*w + x*x + y*y + z*z
        if Nq < 1e-8:
            return np.eye(3, dtype=np.float32)
        s = 2.0 / Nq
        X, Y, Z = x*s, y*s, z*s
        wX, wY, wZ = w*X, w*Y, w*Z
        xX, xY, xZ = x*X, x*Y, x*Z
        yY, yZ = y*Y, y*Z
        zZ = z*Z
        R = np.array([
            [1.0 - (yY + zZ), xY - wZ, xZ + wY],
            [xY + wZ, 1.0 - (xX + zZ), yZ - wX],
            [xZ - wY, yZ + wX, 1.0 - (xX + yY)]
        ], dtype=np.float32)
        return R

    @staticmethod
    def build_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
        """四元数转旋转矩阵: quaternion [N,4] -> [N,3,3]; q=[w,x,y,z]"""
        q = F.normalize(quaternion, dim=-1)
        w, x, y, z = q.unbind(-1)
        ww, xx, yy, zz = w*w, x*x, y*y, z*z
        wx, wy, wz = w*x, w*y, w*z
        xy, xz, yz = x*y, x*z, y*z
        R = torch.stack([
            1-2*(yy+zz), 2*(xy-wz), 2*(xz+wy),
            2*(xy+wz), 1-2*(xx+zz), 2*(yz-wx),
            2*(xz-wy), 2*(yz+wx), 1-2*(xx+yy)
        ], dim=-1).view(-1,3,3)
        return R

    @staticmethod
    def build_covariance_3d(scaling: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
        R = MathUtils.build_rotation_matrix(rotation)
        D = torch.diag_embed(scaling ** 2)
        return R @ D @ R.transpose(-1, -2)

    @staticmethod
    def project_covariance_2d(cov3d: torch.Tensor, viewmatrix: torch.Tensor, projmatrix: torch.Tensor) -> torch.Tensor:
        """线性化投影近似: J * Cov * J^T (此处仅提供接口，未在简化渲染中使用)"""
        # 这里返回各向同性近似
        var = cov3d.diagonal(dim1=-2, dim2=-1).mean(dim=-1)  # [N]
        return torch.diag_embed(torch.stack([var, var], dim=-1))  # [N,2,2]

    @staticmethod
    def spherical_harmonics_eval(degrees: int, dirs: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
        """球谐函数求值 (极简): 仅 DC（degree=0）返回 coeffs[:,0]
        完整 SH 评估超出教育实现范围，可按需扩展。
        """
        return coeffs[:, 0]  # [N,3]


class IOUtils:
    """IO工具"""

    @staticmethod
    def save_image(image: torch.Tensor, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        img = (image.detach().cpu().clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
        if Image is None:
            return
        Image.fromarray(img).save(path)

    @staticmethod
    def load_image(path: str) -> torch.Tensor:
        if Image is None:
            raise ImportError("Pillow is required to load images")
        img = Image.open(path).convert("RGB")
        arr = torch.from_numpy(np.array(img)).float() / 255.0  # [H,W,3]
        return arr.permute(2, 0, 1).contiguous()

    @staticmethod
    def save_point_cloud(points: np.ndarray, colors: Optional[np.ndarray], path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if path.endswith(".npz"):
            np.savez(path, points=points, colors=colors)
        elif path.endswith(".npy"):
            if colors is not None:
                np.save(path, np.concatenate([points, colors], axis=1))
            else:
                np.save(path, points)
        else:
            # 简单保存为 xyz rgb 文本
            with open(path, "w", encoding="utf-8") as f:
                for i in range(points.shape[0]):
                    if colors is None:
                        f.write(f"{points[i,0]} {points[i,1]} {points[i,2]}\n")
                    else:
                        f.write(f"{points[i,0]} {points[i,1]} {points[i,2]} {colors[i,0]} {colors[i,1]} {colors[i,2]}\n")

    @staticmethod
    def load_point_cloud(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        p = Path(path)
        if p.suffix.lower() in {".npz"}:
            data = np.load(str(p))
            pts = data.get("points", np.zeros((0, 3), dtype=np.float32))
            cols = data.get("colors", None)
            return pts.astype(np.float32), (None if cols is None else cols.astype(np.float32))
        if p.suffix.lower() in {".npy"}:
            arr = np.load(str(p))
            if arr.ndim == 2 and arr.shape[1] >= 6:
                return arr[:, :3].astype(np.float32), arr[:, 3:6].astype(np.float32)
            return arr[:, :3].astype(np.float32), None
        if p.suffix.lower() in {".txt"} and p.name == "points3D.txt":
            # COLMAP points3D.txt
            points = []
            colors = []
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if not line or line.startswith("#"):
                        continue
                    parts = line.strip().split()
                    if len(parts) < 10:
                        continue
                    x, y, z = map(float, parts[1:4])
                    r, g, b = map(float, parts[4:7])
                    points.append([x, y, z])
                    colors.append([r/255.0, g/255.0, b/255.0])
            pts = np.asarray(points, dtype=np.float32)
            cols = np.asarray(colors, dtype=np.float32)
            return pts, cols
        # 其他文本格式：假设 xyz[ rgb]
        try:
            pts = []
            cols = []
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = list(map(float, line.split()))
                    if len(parts) >= 3:
                        pts.append(parts[:3])
                        if len(parts) >= 6:
                            cols.append(parts[3:6])
            pts = np.asarray(pts, dtype=np.float32)
            if cols:
                cols = np.asarray(cols, dtype=np.float32)
            else:
                cols = None
            return pts, cols
        except Exception:
            return np.zeros((0, 3), dtype=np.float32), None


class VisualizationUtils:
    """可视化工具 (占位：函数签名保留)"""

    @staticmethod
    def visualize_cameras(cameras: List[Camera]) -> None:
        pass

    @staticmethod
    def visualize_gaussians(gaussians: GaussianModel) -> None:
        pass

    @staticmethod
    def create_video_from_cameras(gaussians: GaussianModel, cameras: List[Camera], output_path: str) -> None:
        pass


# =============================================================================
# main.py - 主入口
# =============================================================================

def main():
    cfg_path = Path("./config.yaml")
    if cfg_path.exists():
        config = ConfigManager.load_from_yaml(str(cfg_path))
    else:
        config = ConfigManager.get_default_config()
        ConfigManager.save_to_yaml(config, str(cfg_path))

    trainer = GaussianTrainer(config)
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()
