# models.py
# 模型架构定义模块

import torch
import torch.nn as nn

class DistilledMLP(nn.Module):
    """
    知识蒸馏多层感知机模型
    实现了一个带有知识蒸馏功能的多层感知机神经网络
    """
    def __init__(self, input_dim, output_dim, config):
        """
        初始化模型
        Args:
            input_dim: 输入特征维度
            output_dim: 输出特征维度
            config: 配置字典，包含模型结构参数
        """
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        # 构建隐藏层
        for i, dim in enumerate(config["hidden_layers"]):
            layers.append(nn.Linear(prev_dim, dim))  # 线性层
            layers.append(nn.ReLU())                 # ReLU激活函数
            if config["dropout_rate"] > 0:
                layers.append(nn.Dropout(config["dropout_rate"]))  # Dropout层防止过拟合
            prev_dim = dim
            
        # 构建输出层
        self.hidden_layers = nn.Sequential(*layers)  # 打包隐藏层
        self.output_layer = nn.Linear(prev_dim, output_dim)  # 输出层
        
        # 选择输出激活函数
        if config["output_activation"] == "sigmoid":
            self.activation = nn.Sigmoid()  # Sigmoid激活函数：输出范围[0,1]
        else:
            self.activation = lambda x: x  # 线性激活：直接输出
            
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入数据张量
        Returns:
            tensor: 模型输出
        """
        x = self.hidden_layers(x)  # 通过隐藏层
        return self.activation(self.output_layer(x))  # 通过输出层和激活函数


class MetabolicLoss(nn.Module):
    """
    代谢网络专用的损失函数
    结合了MSE损失、温度缩放和通量约束
    """
    def __init__(self, config):
        """
        初始化损失函数
        Args:
            config: 配置字典，包含温度和约束权重参数
        """
        super().__init__()
        self.mse_loss = nn.MSELoss()  # 均方误差损失
        self.temperature = config["temperature"]  # 知识蒸馏温度参数
        self.constraint_weight = config["flux_constraint_weight"]  # 通量约束权重
        
    def forward(self, pred, target, model=None):
        """
        计算损失值
        Args:
            pred: 模型预测值
            target: 目标值
            model: 模型实例（用于约束计算）
        Returns:
            tensor: 计算得到的总损失值
        """
        # 计算基础MSE损失
        base_loss = self.mse_loss(pred, target)
        
        # 应用温度缩放（知识蒸馏）
        if self.temperature != 1.0:
            soft_target = target / self.temperature
            soft_pred = pred / self.temperature
            temp_loss = self.mse_loss(soft_pred, soft_target)
            base_loss += temp_loss
            
        # 添加通量约束（确保预测的通量为非负值）
        if self.constraint_weight > 0 and model is not None:
            negative_flux = torch.relu(-model.output_layer.weight)  # 获取负通量
            constraint_loss = torch.mean(negative_flux)  # 计算约束损失
            base_loss += self.constraint_weight * constraint_loss
            
        return base_loss 