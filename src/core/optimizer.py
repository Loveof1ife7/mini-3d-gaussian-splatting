import math
import torch
from typing import Optional
from config.config import TrainingConfig
from src.core.gaussian_model import GaussianModel

class LearningRateScheduler:
    """学习率调度器"""
    def __init__(self, 
                 lr_init: float,
                 lr_final: float,
                 lr_delay_steps: int,
                 lr_delay_mult: float,
                 max_steps: int):
        self.lr_init = lr_init
        self.lr_final = lr_final
        self.lr_delay_steps = lr_delay_steps
        self.lr_delay_mult = lr_delay_mult
        self.max_steps = max_steps
    
    def get_lr(self, step: int) -> float:
        """获取当前步骤的学习率"""
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
    """密度自适应控制器"""
    def __init__(self, config: TrainingConfig):
        self.config = config
        
    def should_densify(self, iteration: int) -> bool:
        return (self.config.densify_from_iter <= iteration <= self.config.densify_until_iter) and \
               (iteration % self.config.densify_interval == 0)
               
    @torch.no_grad()
    def densify_and_prune(self,
                         gaussians: GaussianModel,
                         optimizer: torch.optim.Optimizer,
                         iteration: int,
                         scene_extent: float) -> None:
        """
        密度自适应控制主方法
        
        流水线:
        1. 计算位置梯度
        2. 标记需要分裂的高斯 (大+高梯度)
        3. 标记需要克隆的高斯 (小+高梯度) 
        4. 执行分裂和克隆
        5. 修剪低不透明度高斯
        6. 更新优化器状态
        """
        grad_th = self.config.densify_grad_threshold
        gaussians.density_and_split(grad_th, scene_extent)
        gaussians.density_and_clone(grad_th, scene_extent)
        
        keep = (gaussians.get_opacity().squeeze(1) > 0.01)
        if keep.numel() > 0:
            gaussians.prune_points(keep)
            
        if isinstance(optimizer, torch.optim.Adam):
            optimizer.param_groups.clear()
            optimizer.state.clear
            optimizer.add_param_group({'params': gaussians.get_parameters()})
    
    def _compute_gradient_mask(self,
                              gaussians: GaussianModel,
                              grad_threshold: float) -> torch.Tensor:
        if gaussians._xyz.grad is None:
            return torch.zeros_like(gaussians.get_num_points(), dtype=torch.bool, device=gaussians._xyz.device)
        
        return gaussians._xyz.grad.norm(dim=-1) > grad_threshold
        
    def _get_size_mask(self,
                      gaussians: GaussianModel,
                      percent_dense: float,
                      scene_extent: float) -> torch.Tensor:
        """计算尺寸掩码 (大/小高斯)"""
        size = gaussians.get_scaling().mean(dim = -1)
        th = torch.quantile(size, torch.tensor(percent_dense, device = size.device))
        return size > th
    
class GaussianOptimizer:
    """3DGS优化器包装类"""
    def __init__(self, gaussians: GaussianModel, config: TrainingConfig):
        self.gaussians = gaussians  
        self.config = config  
        self.optimizer: Optional[torch.optim.Adam] = None
        
        self.lr_scheduler = LearningRateScheduler(config.position_lr_init, config.position_lr_final, 0, 1.0, config.iterations)
        self.density_controller = DensityController(self.config)
    
    def setup_optimizer(self) -> None:
        """设置各参数组的优化器"""
        params = [
            {"params": [self.gaussians._xyz], "lr": self.config.position_lr_init},
            {"params": [self.gaussians._features_dc, self.gaussians._features_rest], "lr": self.config.feature_lr},
            {"params": [self.gaussians._opacity], "lr": self.config.opacity_lr},
            {"params": [self.gaussians._scaling], "lr": self.config.scaling_lr},
            {"params": [self.gaussians._rotation], "lr": self.config.rotation_lr},
        ]
        self.optimizer = torch.optim.Adam(params)
    def step(self) -> None:
        """执行一步优化"""
        assert self.optimizer is not None
        self.optimizer.step()
    
    def zero_grad(self) -> None:
        """清零梯度"""
        assert self.optimizer is not None
        self.optimizer.zero_grad(set_to_none=True)
    
    def update_learning_rate(self, iteration: int) -> None:
        """更新学习率"""
        assert self.optimizer is not None
        base_lr = self.lr_scheduler.get_lr(iteration)
        
        self.optimizer.param_groups[0]["lr"] = base_lr
        self.optimizer.param_groups[1]["lr"] = base_lr * (self.config.feature_lr / self.config.position_lr_init)
        self.optimizer.param_groups[2]["lr"] = base_lr * (self.config.opacity_lr / self.config.position_lr_init)
        self.optimizer.param_groups[3]["lr"] = base_lr * (self.config.scaling_lr / self.config.position_lr_init)
        self.optimizer.param_groups[4]["lr"] = base_lr * (self.config.rotation_lr / self.config.position_lr_init)
        
    
    def densify_and_prune(self, iteration: int, scene_extent: float) -> None:
        """密度控制"""
        if self.density_controller.should_densify(iteration):
            assert self.optimizer is not None
            self.density_controller.densify_and_prune(self.gaussians, self.optimizer, iteration, scene_extent)
            self.setup_optimizer()
    
    def reset_opacity(self) -> None:
        """重置不透明度"""
        self.gaussians.reset_opacity()
        
    