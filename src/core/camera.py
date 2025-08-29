from typing import Optional
import math
import torch
import numpy as np
class Camera:
    """相机类"""
    
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
        """
        初始化相机
        
        Args:
            uid: 相机ID
            R: 旋转矩阵 [3, 3]
            T: 平移向量 [3]
            FoVx: X方向视场角
            FoVy: Y方向视场角
            image: 图像张量 [3, H, W]
            image_name: 图像名称
            width: 图像宽度
            height: 图像高度
        """
        self._uid = uid
        self._R = torch.from_numpy(R).float()
        self._T = torch.from_numpy(T).float()
        self._FoVx = float(FoVx)
        self._FoVy = float(FoVy)
        self._image = image
        self._image_name = image_name
        self._height = height
        self._width = width
        self._proj: Optional[torch.Tensor] = None
        self._wv: Optional[torch.Tensor] = None
    
    # 属性
    @property
    def world_view_transform(self) -> torch.Tensor:
        """世界到视图变换矩阵"""
        if self._wv is None:
            self._wv = CameraUtils.build_world_view_matrix(self._R.cpu().numpy, self._T.cpu().numpy)
        return self._wv
    
    @property
    def projection_matrix(self) -> torch.Tensor:
        """投影矩阵"""
        if self._proj is None:
            self._proj = CameraUtils.build_projection_matrix(znear=0.1, zfar=1000.0,
                                                            fovX=self._FoVx, fovY=self._FoVy)
        return self._proj
    
    @property
    def full_proj_transform(self) -> torch.Tensor:
        """完整投影变换矩阵"""
        return self.projection_matrix @ self.world_view_transform
    
    @property
    def camera_center(self) -> torch.Tensor:
        """相机中心位置"""
        pass

class CameraUtils:
    """相机工具类"""
    
    @staticmethod
    def build_world_view_matrix(R: np.ndarray, T: np.ndarray) -> torch.Tensor:
        """构建世界到视图变换矩阵"""

        assert R.shape == (3, 3), f"R should be [3,3], got {R.shape}"
        assert T.shape == (3, 1), f"T should be [3,1], got {T.shape}"
        
        # 构建4x4视图矩阵
        view_matrix = torch.eye(4, dtype=R.dtype)
        view_matrix[:3, :3] = R.t()  # 旋转部分取转置
        view_matrix[:3, 3] = (-R.t() @ T).flatten()  # 平移部分
        
        return view_matrix
    @staticmethod
    def build_projection_matrix_v1(znear: float, zfar: float, 
                                  fovX: float, fovY: float) -> torch.Tensor:
        """
        版本1: 标准OpenGL形式 (不需要width/height)
        """
        tanHalfFovX = math.tan(fovX * 0.5)
        tanHalfFovY = math.tan(fovY * 0.5)
        
        # 避免除零错误
        if abs(tanHalfFovX) < 1e-6:
            tanHalfFovX = 1e-6
        if abs(tanHalfFovY) < 1e-6:
            tanHalfFovY = 1e-6
            
        A = 1.0 / tanHalfFovX
        B = 1.0 / tanHalfFovY
        C = -(zfar + znear) / (zfar - znear)
        D = -(2 * zfar * znear) / (zfar - znear)
        
        P = torch.tensor([[A, 0,   0, 0],
                          [0, B,   0, 0],
                          [0, 0,   C, D],
                          [0, 0,  -1, 0]], dtype=torch.float32)
        return P
    
    @staticmethod
    def build_projection_matrix_v2(znear: float, zfar: float, 
                                  fovX: float, fovY: float,
                                  width: int, height: int) -> torch.Tensor:
        """
        版本2: 基于焦距推导的形式 (需要width/height)
        """
        # 计算焦距
        focal_x = (width / 2) / math.tan(fovX / 2)
        focal_y = (height / 2) / math.tan(fovY / 2)
        
        # 构建投影矩阵
        proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
        
        proj_matrix[0, 0] = 2 * focal_x / width    # 水平缩放
        proj_matrix[1, 1] = 2 * focal_y / height   # 垂直缩放
        proj_matrix[2, 2] = -(zfar + znear) / (zfar - znear)  # z压缩
        proj_matrix[2, 3] = -2 * zfar * znear / (zfar - znear)  # z平移
        proj_matrix[3, 2] = -1  # 透视除法
        
        return proj_matrix
    def project_points_3d_to_2d(points_3d: torch.Tensor,
                                camera: Camera) -> torch.Tensor:
        """将3D点投影到2D屏幕空间"""
        N = points_3d.shape[0]
        ones = torch.ones((N, 1), device=points_3d.device)    
        homo = torch.cat([points_3d, ones], dim=1) # [N,4]
        M = camera.full_proj_transform.to(points_3d.device) # [4,4]
        clip = (M @ homo.t()).t() # [4, 4] @ [4, N] -> [4, N] -> [N, 4]
        ndc = clip[:, :3] / clip[:, 3:].clamp_min(1e-8) # x ∈ [-1, 1]， y ∈ [-1, 1]
        
        x = (ndc[:, 0] * 0.5 + 0.5) * camera.width # [-1,1] -> [0, 1] -> [0, width]
   
        y = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * camera.height # 在 NDC 中，+1 在上方，-1 在下方（OpenGL 习惯, 在图像像素坐标里，0 在上，往下增大
        
        xy = torch.stack([x, y], dim=1)
        depth = clip[:, 2]  
        return xy, depth
