# config.py
# 配置参数模块

import torch

# 模型配置参数
CONFIG = {
    "data_path": "bulk_data.csv",          # 输入数据文件路径
    "model_path": "Recon3D.xml",           # 代谢模型文件路径
    "output_activation": "sigmoid",         # 输出层激活函数类型：sigmoid或linear
    "hidden_layers": [256, 256],           # 隐藏层结构：两个256节点的隐藏层
    "dropout_rate": 0.2,                   # Dropout比率：防止过拟合
    "batch_size": 64,                      # 批处理大小
    "lr": 1e-4,                            # 学习率
    "epochs": 500,                         # 训练轮数
    "early_stop_patience": 1000,           # 早停耐心值：验证损失多少轮未改善就停止（设置很大的值实际上禁用了早停）
    "device": "cuda" if torch.cuda.is_available() else "cpu",  # 训练设备：GPU或CPU
    "flux_constraint_weight": 0.1,         # 通量约束权重：用于正则化
    "temperature": 2.0                     # 知识蒸馏温度参数
} 