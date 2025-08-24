3d_gaussian_splatting/
├── README.md
├── requirements.txt
├── main.py                     # 主训练脚本
├── config/
│   ├── __init__.py
│   └── default.yaml           # 默认配置文件
├── src/
│   ├── __init__.py
│   ├── gaussian_model.py      # 3D高斯模型核心
│   ├── camera.py              # 相机模型
│   ├── renderer.py            # 可微分渲染器
│   ├── loss.py                # 损失函数
│   ├── optimizer.py           # 优化器和密度控制
│   ├── trainer.py             # 训练器主类
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── io_utils.py        # IO工具
│   │   ├── math_utils.py      # 数学工具
│   │   ├── vis_utils.py       # 可视化工具
│   │   └── colmap_utils.py    # COLMAP数据处理
│   └── data/
│       ├── __init__.py
│       ├── dataset.py         # 数据集类
│       └── transforms.py      # 数据变换
├── scripts/
│   ├── preprocess_colmap.py   # COLMAP数据预处理
│   ├── evaluate.py            # 模型评估脚本
│   └── render_video.py        # 渲染视频脚本
├── tests/
│   ├── __init__.py
│   ├── test_gaussian_model.py
│   ├── test_renderer.py
│   └── test_math_utils.py
└── examples/
    ├── simple_scene/          # 简单场景示例
    └── complex_scene/         # 复杂场景示例