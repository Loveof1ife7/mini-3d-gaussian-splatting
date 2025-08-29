

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
import os   

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None 

# =============================================================================
# config/config.py - 配置管理
# =============================================================================
@dataclass
class TrainingConfig:
    """训练配置"""
    # 数据路径
    data_path: str = "data/scene"
    output_path: str = "output"
    
    # 训练参数
    iterations: int = 30000
    learning_rate: float = 0.0025
    batch_size: int = 1
    
    # 高斯参数
    position_lr_init: float = 0.00016
    position_lr_final: float = 0.0000016
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 30000
    
    feature_lr: float = 0.0025
    opacity_lr: float = 0.05
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001
    
    # 密度控制参数
    densify_from_iter: int = 500
    densify_until_iter: int = 15000
    densify_grad_threshold: float = 0.0002
    densify_interval: int = 100
    
    # 渲染参数
    image_height: int = 800
    image_width: int = 800
    
    # 设备
    device: str = "cuda"
    
class ConfigManager:
    """配置管理器"""
    
    @staticmethod
    def load_from_yaml(config_path: str) -> TrainingConfig:
        """从YAML文件加载配置"""
        if yaml is None:
            raise ImportError("PyYAML is not installed. pip install pyyaml")
        else:
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

cfg = TrainingConfig(batch_size=64)
print(cfg)