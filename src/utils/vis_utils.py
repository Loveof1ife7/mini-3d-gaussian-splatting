from src.core.camera import Camera
from src.core.gaussian_model import GaussianModel

from typing import List
class VisualizationUtils:
    """可视化工具"""
    
    @staticmethod
    def visualize_cameras(cameras: List[Camera]) -> None:
        """可视化相机位置"""
        pass
    
    @staticmethod
    def visualize_gaussians(gaussians: GaussianModel) -> None:
        """可视化高斯分布"""
        pass
    
    @staticmethod
    def create_video_from_cameras(gaussians: GaussianModel,
                                 cameras: List[Camera],
                                 output_path: str) -> None:
        """从相机序列创建视频"""
        pass