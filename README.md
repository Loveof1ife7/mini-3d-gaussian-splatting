# mini-3dgs

## Architecture

```text
+-------------------+      +-----------------+      +------------------+
|  Dataset / Camera | ---> | Gaussian Field  | ---> |  Renderer (2D)   |
| (transforms.json) |      |  (xyz,rgb,σ,α)  |      |  splat + compos. |
+-------------------+      +-----------------+      +------------------+
           ^                          |                        |
           |                          v                        v
           |                     Loss (L1)               Training Loop
           |                          ^                        |
           +--------------------------+------------------------+

mini-3d-gaussian-splatting/
│
├── main.py                          # 主入口文件
├── requirements.txt                 # 依赖包
├── README.md                       # 项目说明
│
├── config/                         # 配置管理
│   ├── __init__.py
│   ├── config.py                   # 配置类和管理器
│   └── default.yaml               # 默认配置文件
│
├── src/                           # 源代码
│   ├── __init__.py
│   │
│   ├── core/                      # 核心算法模块
│   │   ├── __init__.py
│   │   ├── gaussian_model.py      # GaussianModel类
│   │   ├── camera.py              # Camera类 + CameraUtils
│   │   ├── renderer.py            # GaussianRenderer类
│   │   ├── loss.py                # 损失函数 (GaussianLoss, SSIMLoss)
│   │   └── optimizer.py           # 优化器 (GaussianOptimizer, DensityController, LRScheduler)
│   │
│   ├── data/                      # 数据处理模块  
│   │   ├── __init__.py
│   │   ├── dataset.py             # 数据集基类 (CameraDataset, COLMAPDataset)
│   │   └── colmap_utils.py        # COLMAP数据处理工具
│   │
│   ├── training/                  # 训练模块
│   │   ├── __init__.py
│   │   └── trainer.py             # 训练器 (GaussianTrainer)
│   │
│   └── utils/                     # 工具模块
│       ├── __init__.py  
│       ├── math_utils.py          # 数学工具 (MathUtils)
│       ├── io_utils.py            # IO工具 (IOUtils)
│       └── vis_utils.py           # 可视化工具 (VisualizationUtils)
│
├── scripts/                       # 脚本工具
│   ├── preprocess.py              # 数据预处理
│   ├── evaluate.py                # 模型评估
│   └── render_novel_view.py       # 新视角渲染
│
├── tests/                         # 单元测试
│   ├── __init__.py
│   ├── test_gaussian_model.py
│   ├── test_renderer.py
│   └── test_math_utils.py
│
└── examples/                      # 使用示例
    ├── simple_scene/              # 简单场景示例
    └── tutorial.ipynb            # 教程notebook
```

## Classes and Interfaces
```
GaussianModel (nn.Module)
├── 参数管理: _xyz, _features, _scaling, _rotation, _opacity  
├── 属性访问: get_xyz(), get_features(), get_scaling()...
├── 初始化: create_from_pcd(), create_from_random()
├── 操作: densify_and_split(), densify_and_clone(), prune_points()
└── 工具: compute_3d_covariance(), get_num_points()

Camera
├── 基本属性: uid, R, T, FoVx, FoVy, image, width, height
├── 变换矩阵: world_view_transform, projection_matrix, full_proj_transform  
└── 几何属性: camera_center

GaussianRenderer
├── 主方法: render()
├── 流水线步骤:
│   ├── _project_gaussians_3d_to_2d()     # 3D→2D投影
│   ├── _frustum_culling()                # 视锥剔除  
│   ├── _sort_gaussians_by_depth()        # 深度排序
│   └── _tile_rasterization()             # 瓦片光栅化

GaussianOptimizer
├── 组件管理: optimizer, lr_scheduler, density_controller
├── 优化控制: step(), zero_grad(), update_learning_rate()
└── 密度控制: densify_and_prune(), reset_opacity()

GaussianTrainer
├── 组件集成: dataset, gaussians, renderer, optimizer, loss_fn
├── 训练流程: setup(), train(), train_step()
├── 评估保存: validate(), save_checkpoint(), load_checkpoint()
└── 工具方法: get_scene_extent()

数据流向: COLMAPDataset → Camera → GaussianTrainer → GaussianRenderer → GaussianLoss
```

```python

1. 数据接口
# 数据集 → 相机列表
dataset.get_train_cameras() → List[Camera]
dataset.get_test_cameras() → List[Camera]

# 相机 → 渲染参数  
camera.world_view_transform → torch.Tensor [4,4]
camera.projection_matrix → torch.Tensor [4,4]

2. 模型接口
# 高斯模型 → 渲染参数
gaussians.get_xyz() → torch.Tensor [N,3]
gaussians.get_features() → torch.Tensor [N,16,3] 
gaussians.get_scaling() → torch.Tensor [N,3]
gaussians.get_rotation() → torch.Tensor [N,4]
gaussians.get_opacity() → torch.Tensor [N,1]

3. 渲染接口
# 渲染器输入输出
renderer.render(camera, gaussians, settings) → {
    "image": torch.Tensor [3,H,W],
    "alpha": torch.Tensor [1,H,W], 
    "depth": torch.Tensor [1,H,W],
    "viewspace_points": torch.Tensor [N,2],
    "visibility_filter": torch.Tensor [N],
    "radii": torch.Tensor [N]
}

4. 训练接口
# 训练器流程
trainer.setup() → None                    # 初始化所有组件
trainer.train() → None                    # 主训练循环
trainer.train_step(camera) → loss_dict   # 单步训练
trainer.validate() → metrics_dict        # 模型验证

5. 依赖关系
配置层: config/ 
   ↓
核心层: src/core/ (gaussian_model, camera, renderer, loss, optimizer)
   ↓  
数据层: src/data/ (dataset, colmap_utils)
   ↓
训练层: src/training/ (trainer)
   ↓
工具层: src/utils/ (math_utils, io_utils, vis_utils)
```

## Pipeline
```
1.Render

3D高斯参数 → 投影变换 → 视锥剔除 → 深度排序 → 瓦片光栅化 → 像素混合 → 输出图像
     ↑            ↑         ↑        ↑         ↑          ↑         ↑
GaussianModel  MathUtils  Renderer  CUDA     CUDA     CUDA    torch.Tensor

2.Train

COLMAP数据 → 相机采样 → 高斯渲染 → 损失计算 → 梯度反传 → 参数更新 → 密度控制
     ↑          ↑         ↑         ↑         ↑         ↑         ↑
COLMAPDataset Camera  Renderer  GaussianLoss  PyTorch  Optimizer DensityController

3.Density Control

梯度统计 → 标记分裂 → 标记克隆 → 标记修剪 → 执行操作 → 更新优化器
   ↑         ↑         ↑         ↑         ↑          ↑
Trainer  Controller Controller Controller  Model   Optimizer
```

## 