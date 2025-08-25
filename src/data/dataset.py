from typing import List, Dict
from pathlib import Path
from config import TrainingConfig
from core.camera import Camera

class CameraDataset:
    def __init__(self, data_path: str):
        self.data_path = Path(data_path) 
        self.cameras: List[Camera] = []
        self.train_cameras: List[Camera] = []
        self.test_cameras: List[Camera] = []
        pass
    
    def load_cameras(self) -> None :
        pass

    def split_train_test(self, split_ratio: float) -> None:
        pass
    def get_train_cameras(self) -> List[Camera]:
        return self.train_cameras
    
    def get_test_cameras(self) -> List[Camera]:
        return self.test_cameras
    
    def get_scene_info(self) -> Dict[str, any]:
        """获取场景信息 (边界、中心等)"""
        pass
    

class COLMAPDataset(CameraDataset):
    def __init__(self, data_path):
        super().__init__(data_path)
        
    def load_cameras(self):
        """
        加载COLMAP格式数据
        
        流水线:
        1. 读取cameras.txt
        2. 读取images.txt  
        3. 读取points3D.txt
        4. 加载图像文件
        5. 构建Camera对象
        """
        pass
        
    def _read_cameras_txt(self) -> Dict:
        """读取cameras.txt"""
        pass
    
    def _read_images_txt(self) -> Dict:
        """读取images.txt"""
        pass
    
    def _read_points3d_txt(self) -> np.ndarray:
        """读取points3D.txt"""
        pass
    
    def get_point_cloud_path(self) -> str:
        """获取点云文件路径"""
        pass
   
   
    