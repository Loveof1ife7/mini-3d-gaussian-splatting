from dataclasses import dataclass
from typing import Dict, Tuple
import torch
import math

from .camera import Camera, CameraUtils
from .gaussian_model import GaussianModel

# =============================================================================
# src/core/renderer.py - 可微分渲染器
# =============================================================================

@dataclass
class RenderSettings:
    '''渲染设置'''
    image_height: int
    image_width: int
    bg_color: torch.Tensor  
    scale_modifier: float = 1.0
    debug: bool = False

class GaussianRenderer:
    """3D高斯可微分渲染器"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.radius_min = 0.01   
        self.radius_max = 30.0   
        self.tile_size = 16
    
    def render(self,
               camera: Camera,
               gaussians: GaussianModel,
               settings: RenderSettings
            ) -> Dict[str, torch.Tensor]:
        pass
    
        """
        主渲染方法
        
        Args:
            camera: 相机对象
            gaussians: 高斯模型
            settings: 渲染设置
            
        Returns:
            Dict包含:
            - "image": 渲染图像 [3, H, W] 
            - "alpha": Alpha通道 [1, H, W]
            - "depth": 深度图 [1, H, W]
            - "viewspace_points": 2D投影点
            - "visibility_filter": 可见性掩码
            - "radii": 2D高斯半径
        """
        # 流水线:
        # 1. 3D高斯投影到2D屏幕空间
        # 2. 视锥剔除
        # 3. 深度排序
        # 4. 瓦片光栅化
        # 5. Alpha混合渲染
        device = gaussians.get_xyz.device
        H, W = settings.image_height, settings.image_width
        bg = settings.bg_color.to(device).view(3, 1, 1)
        
        # 1. 投影3D高斯到2D
        proj = self._project_gaussians_3d_to_2d(camera, gaussians, settings)
        means2D = proj["means2D"]          # [N,2]
        cov2D   = proj["cov2D"]            # [N,2,2]
        depths  = proj["depths"]           # [N]  (相机坐标 Z，>0 表示在前)
        radii   = proj["radii"]            # [N]  (像素半径近似)
        conics  = proj["conics"]           # [N,2,2]  (≈ cov2D^{-1})
        
        # 2. 视锥剔除
        vis_mask = self._frustum_culling(means2D, depths, radii, settings)
        
        if vis_mask.sum() == 0:
            return {
                "image": bg.repeat(1, H, W),
                "alpha": torch.zeros(1, H, W, device),
                "depth": torch.zeros(1, H, W, device),
                "viewspace_points": means2D,
                "visibility_filter": vis_mask,
                "radii": radii,
                "conics": conics
            }
            
        # 3. 深度排序
        sorted_indices = self._sort_gaussians_by_depth(vis_mask, depths)
        
        feats = gaussians.get_features()
        if feats.dim() == 3 and feats.shape[1] >=1 :
            colors = torch.sigmoid(feats[:, 0, :]) # [N, 3]
        else:
            colors = torch.sigmoid(gaussians._features_dc.squeeze(1)) # [N, 3]
        
        opacities = gaussians.get_opacity().squeeze(1)
        
        # 4.5. 瓦片光栅化
        render_results = self._tile_rasterization(
            sorted_indices=sorted_indices,
            gaussians_2d = {"means2D": means2D, "conics": conics, "depths": depths, "radii": radii},
            colors=colors,
            opacities=opacities,
            bg=bg,
            settings=settings
        )

        return {
            "image": render_results["image"],
            "alpha": render_results["alpha"],
            "depth": render_results["depth"],
            "viewspace_points": means2D,
            "visibility_filter": vis_mask,
            "radii": radii,
            "conics": conics,
        }

    
    def _project_gaussians_3d_to_2d(self,
                                   camera: Camera,
                                   gaussians: GaussianModel) -> Tuple[torch.Tensor, ...]:
        
        """
        步骤1: 投影3D高斯到2D
        
        流水线:
        - 变换3D位置到相机坐标系
            - 世界 -> 相机：X_cam = R*X + T
            - 透视投影到像素：x = fx*X/Z + cx, y = fy*Y/Z + cy（y 轴向下）
        - 计算3D协方差矩阵  
        - 投影到屏幕空间
             - 协方差传播：Σ2D ≈ J Σcam J^T，其中 Σcam = Rv Σworld Rv^T
        - 计算2D协方差矩阵
        - 计算2D高斯椭圆参数
        """
        device = gaussians.get_xyz.device
        Xw = gaussians.get_xyz # [N,3]
        N = Xw.shape[0]
        
        # intrinsics 
        W = camera._width
        H = camera._height
        fx = torch.as_tensor(0.5 * W / math.tan(camera._FoVx * 0.5), dtype=X.dtype, device=device)
        fy = torch.as_tensor(0.5 * H / math.tan(camera._FoVy * 0.5), dtype=X.dtype, device=device)
        
        # center of image plane (W/2 ,H/2)
        cx = torch.as_tensor(W * 0.5, dtype=X.dtype, device=device) 
        cy = torch.as_tensor(H * 0.5, dtype=X.dtype, device=device) 
        
        # world -> camera
        WV = camera.world_view_transform().to(device) # [4,4]
        Rv = WV[:3, :3] # [3,3]
        Tv = WV[:3, 3] # [3]

        Xc = Xw @ Rv.T + Tv
        
        # X, Y, Z: 相机坐标系下的点坐标
        #   - 单位：米（或者你的世界单位）
        #   - Z > 0 表示在相机前方，一般 [0.01 ~ +∞)
        X, Y, Z = Xc[:, 0], Xc[:, 1], Xc[:, 2].clamp(min=1e-6)
        
        xpix = fx * X / Z + cx
        ypix = - fy * Y / Z + cy
        means2D = torch.stack([xpix, ypix], dim=-1)  # [N,2]
        
        # 3D Covariance Matrix -> 2D
        cov3d = gaussians.get_covariance() # [N, 3 ,3]
        # cov_cam = Rv Σ Rv^T
        cov_cam = Rv @ cov3d @ Rv.T # pytorch broadcaset Rv to [N, 3, 3]
        
        # Jacobian of projection transformation (X,Y,Z) -> (x,y), [2, 3]
        J = torch.zeros((2, 3), dtype=X.dtype, device=device)
        invZ = 1.0 / Z
        invZ2 = invZ * invZ
        J[0, 0] = fx * invZ
        J[:, 0, 2] = -fx * X * invZ * invZ
        J[:, 1, 1] = -fy * invZ     
        J[:, 1, 2] =  fy * Y * invZ * invZ
        
        cov2d = J @ cov_cam @ J.transpose(-1, -2) # [N, 2, 2]
        
        # 数值稳定
        eps = torch.eye(2, device=X.device, dtype=X.dtype).unsqueeze(0).expand(N, 2, 2) * 1e-6
        cov2d = cov2d + eps
        
        # conic Q = cov2d^{-1}
        conics = torch.linalg.inv(cov2d)
        
        eigvals = torch.linalg.eigvalsh(cov2d) # [N, 2]
        
        r = 3.0 * torch.sqrt(eigvals[:, 1]) # [N]

        r = r.clamp(self.radius_min, self.radius_max)
        
        return {
            "means2D": means2D,
            "cov2D": cov2d,
            "conics": conics,
            "depths": Z,    
            "radii": r,
        }
    def _frustum_culling(self,
                        means2D: torch.Tensor,
                        depths: torch.Tensor,
                        radii: torch.Tensor,
                        settings: RenderSettings) -> torch.Tensor:
        """
        步骤2: 视锥剔除
        
        流水线:
        - 深度测试 (z > near)
        - 屏幕边界测试
        - 半径测试 (radii > 0)
        """
        
        H, W = settings.image_height, settings.image_width
        x, y = means2D[:, 0], means2D[:, 1] # [N, 1]
        
        vis_mask = (depths > 0 ) & (x >= -radii) & (x < W + radii) & (y >= -radii) & (y < H + radii) & (radii > 0) # [N, 1]
        
        return vis_mask 
        
    def _sort_gaussians_by_depth(self,
                                visibility_filter: torch.Tensor,
                                depths: torch.Tensor) -> torch.Tensor:
        """
        步骤3: 按深度排序
        """ 
        visibility_filter = visibility_filter.squeeze(-1) # [N, 1] -> [N]  
        
        depths = depths.squeeze(-1)
        
        #  visibility_filter: [T,F,T,T,F,T],
        #  depths: [2.5,0.3,4.1,1.2,7.7,1.2]
        
        idx = torch.nonzero(visibility_filter, as_tuple=False).flatten() # [M] = [0,2,3,5]
        
        d = depths[idx] # [M] = [2.5, 4.1, 1.2, 1.2]
        
        order = torch.argsort(d)  # [M] = [2, 3, 0, 1]
        
        sorted_idx = idx[order] # [M] = [3, 5, 0, 2]
        
        return sorted_idx
        
    def _tile_rasterization(self,
                            sorted_indices: torch.Tensor,
                            gaussians_2d: Dict[str, torch.Tensor],
                            colors: torch.Tensor,
                            opacities: torch.Tensor,
                            bg: torch.Tensor,
                            settings: RenderSettings
                            ) -> torch.Tensor:
        
        """
        步骤4-5: 瓦片光栅化和Alpha混合
        
        流水线:
        - 计算瓦片边界
        - 高斯-瓦片相交测试
        - 每个像素Alpha混合
        - 早期终止优化
        """
        device = colors.device
        H, W = settings.image_height, settings.image_width
        T = self.tile_size
        
        tiles_x = math.ceil(W / T)
        tiles_y = math.ceil(H / T)
        num_tiles = tiles_x * tiles_y
        tile_lists = [[] for _ in range(num_tiles)]
        
        means2D = gaussians_2d["means2D"]
        conics  = gaussians_2d["conics"]
        depths  = gaussians_2d["depths"]
        radii   = gaussians_2d["radii"]
        
        out_rgb = bg.expand(3, H, W).clone()
        out_a   = torch.zeros(1, H, W, device=device)
        out_d   = torch.zeros(1, H, W, device=device)

        for i in sorted_indices.tolist():
            r = int(radii[i].item())
            if r <= 0:
                continue
            cx, cy = means2D[i]
            
            # gaussian's AABB in the image plane
            x0 = max(int(cx.item()) - r, 0)
            x1 = min(int(cx.item()) + 1 + r, W)
            y0 = max(int(cy.item()) - r, 0)
            y1 = min(int(cy.item()) + 1 + r, H) 
            if x0 >= x1 or y0 >= y1:
                continue
            
            tx0 = x0 // T
            tx1 = (x1 - 1) // T
            ty0 = y0 // T
            ty1 = (y1 - 1) // T
            
            for ty in range(ty0, ty1 + 1):
                for tx in range(tx0, tx1 + 1):
                    tid = ty * tiles_x + tx
                    tile_lists[tid].append(i)
            
            # front-to-back alpha blending
            for ty in range(tiles_y):
                for tx in range(tiles_x):
                    tid = ty * tiles_x + tx
                    lst = tile_lists[tid]
                    
                    # tile's pixel range
                    x0 = tx * T
                    x1 = min(x0 + T, W)
                    y0 = ty * T
                    y1 = min(y0 + T, H)
                    
                    for yy in range(y0, y1):
                        for xx in range(x0, x1):
                            a_acc = out_a[0, yy, xx]
                            if a_acc >= 0.995:
                                continue
                            
                            for i in lst:
                                # 计算二次型 s = (p-u).transpose * Q * (p-u)
                                Q = conics[i] # [2,2]
                                p = torch.tensor([xx, yy], device=device).float()
                                cx, cy = means2D[i]
                                u = torch.tensor([cx, cy], device=device).float()
                                s = (p - u).transpose(-1, -2) @ Q @ (p - u)
                                
                                w = torch.exp(-0.5 * s).clamp(0.0, 1.0)
                                if w < 1e-5:
                                    continue    
                                 
                                a_i = (opacities[i] * w).clamp(0.0, 1.0)
                                if a_i <= 0.0:
                                    continue
                                
                                trans = 1.0 - a_acc
                                contrib = trans * a_i
                                if contrib <= 0.0:
                                    continue
                                
                                out_rgb[:, yy, xx] += contrib * colors[i].view(3)
                                a_acc = a_acc + contrib
                                out_d[0, yy, xx] += contrib * depths[i]
                                
                                if a_acc >= 0.995:
                                    break
                                
                                out_a[0, yy, xx] = a_acc

        return {
            "image": out_rgb.clamp(0, 1),
            "alpha": out_a.clamp(0, 1),
            "depth": out_d
            }
        
    
    