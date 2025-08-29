import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Warning: PIL/Pillow is not installed. Image saving will be disabled.")

class IOUtils:
    """IO工具"""
  
    @staticmethod
    def save_image(image: torch.Tensor, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        image = (image.detach().cpu().clamp(0, 1) * 255).byte().permute(1,2,0)
        if Image is None:
            return
        Image.fromarray(image.numpy()).save(path)
    
    @staticmethod
    def load_image(path: str) -> torch.Tensor:
        pass
    
    @staticmethod
    def save_point_cloud(points: np.ndarray, colors: np.ndarray, path: str) -> None:
        pass
    
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
        

# pts, cols = IOUtils.load_point_cloud("model.npz")
# # pts.shape == (N, 3), pts.dtype == float32
# # cols is None or shape == (N, 3)
# pts2, cols2 = IOUtils.load_point_cloud("xyzrgb.npy")  # [N,6] 或更多列
# pts3, cols3 = IOUtils.load_point_cloud("points3D.txt")  # COLMAP
# pts4, cols4 = IOUtils.load_point_cloud("foo.txt")  # 每行 xyz 或 xyz rgb
