from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

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
        pass
    
    @staticmethod
    def save_to_yaml(config: TrainingConfig, config_path: str) -> None:
        """保存配置到YAML文件"""
        pass
    
    @staticmethod
    def get_default_config() -> TrainingConfig:
        """获取默认配置"""
        pass

cfg = TrainingConfig(batch_size=64)
print(cfg)