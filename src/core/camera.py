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
        pass
    
    # 属性
    @property
    def world_view_transform(self) -> torch.Tensor:
        """世界到视图变换矩阵"""
        pass
    
    @property
    def projection_matrix(self) -> torch.Tensor:
        """投影矩阵"""
        pass
    
    @property
    def full_proj_transform(self) -> torch.Tensor:
        """完整投影变换矩阵"""
        pass
    
    @property
    def camera_center(self) -> torch.Tensor:
        """相机中心位置"""
        pass

class CameraUtils:
    """相机工具类"""
    
    @staticmethod
    def build_world_view_matrix(R: np.ndarray, T: np.ndarray) -> torch.Tensor:
        """构建世界到视图变换矩阵"""
        pass    
    
    @staticmethod
    def build_projection_matrix(znear: float, zfar: float, 
                               fovX: float, fovY: float) -> torch.Tensor:
        """构建投影矩阵"""
        pass
    
    @staticmethod
    def project_points_3d_to_2d(points_3d: torch.Tensor,
                                camera: Camera) -> torch.Tensor:
        """将3D点投影到2D屏幕空间"""
        pass