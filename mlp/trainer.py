# trainer.py
# 训练引擎模块

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from data_utils import DataPreprocessor
from models import DistilledMLP, MetabolicLoss

class Trainer:
    """
    模型训练器类
    负责模型的训练、验证和测试过程
    """
    def __init__(self, config):
        """
        初始化训练器
        Args:
            config: 配置字典，包含所有训练参数
        """
        self.config = config
        
        # 准备数据集
        self.preprocessor = DataPreprocessor(config)
        self.train_set, self.val_set, self.test_set = self.preprocessor.prepare_data()
        
        # 初始化模型
        input_dim = self.train_set.features.shape[1]  # 输入维度
        output_dim = self.train_set.targets.shape[1]  # 输出维度
        self.model = DistilledMLP(input_dim, output_dim, config).to(config["device"])
        
        # 配置优化器和学习率调度器
        self.optimizer = optim.Adam(self.model.parameters(), lr=config["lr"])  # Adam优化器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', patience=5, factor=0.5)  # 学习率自适应调整
        self.criterion = MetabolicLoss(config)  # 损失函数
        
    def train_epoch(self, dataloader):
        """
        训练一个epoch
        Args:
            dataloader: 训练数据加载器
        Returns:
            float: 平均训练损失
        """
        self.model.train()  # 设置为训练模式
        total_loss = 0
        
        for X_batch, y_batch in dataloader:
            # 将数据移到指定设备
            X_batch = X_batch.to(self.config["device"])
            y_batch = y_batch.to(self.config["device"])
            
            # 前向传播和反向传播
            self.optimizer.zero_grad()  # 清除梯度
            outputs = self.model(X_batch)  # 前向传播
            loss = self.criterion(outputs, y_batch, self.model)  # 计算损失
            loss.backward()  # 反向传播
            self.optimizer.step()  # 更新参数
            
            total_loss += loss.item()
            
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader):
        """
        评估模型
        Args:
            dataloader: 验证或测试数据加载器
        Returns:
            float: 平均评估损失
        """
        self.model.eval()  # 设置为评估模式
        total_loss = 0
        
        with torch.no_grad():  # 不计算梯度
            for X_batch, y_batch in dataloader:
                # 将数据移到指定设备
                X_batch = X_batch.to(self.config["device"])
                y_batch = y_batch.to(self.config["device"])
                
                # 前向传播
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                total_loss += loss.item()
                
        return total_loss / len(dataloader)
    
    def run_training(self, use_early_stopping=True):
        """
        执行完整的训练过程
        包括训练、验证和早停
        
        Args:
            use_early_stopping: 是否使用早停机制，设为False将训练所有epoch
        """
        # 创建数据加载器
        train_loader = DataLoader(self.train_set, 
                                batch_size=self.config["batch_size"], 
                                shuffle=True)
        val_loader = DataLoader(self.val_set, 
                              batch_size=self.config["batch_size"])
        
        best_loss = float('inf')  # 记录最佳验证损失
        patience_counter = 0  # 早停计数器
        
        # 训练循环
        for epoch in range(self.config["epochs"]):
            # 训练和验证
            train_loss = self.train_epoch(train_loader)
            val_loss = self.evaluate(val_loader)
            
            # 学习率调整
            self.scheduler.step(val_loss)
            
            # 早停检查
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), "best_model.pth")  # 保存最佳模型
            else:
                patience_counter += 1
                if use_early_stopping and patience_counter >= self.config["early_stop_patience"]:
                    print(f"早停：在epoch {epoch}处停止训练")
                    break
            
            # 打印训练进度
            print(f"Epoch {epoch+1}/{self.config['epochs']} | "
                  f"训练损失: {train_loss:.4f} | 验证损失: {val_loss:.4f} | "
                  f"学习率: {self.optimizer.param_groups[0]['lr']:.2e}")
                  
    def load_best_model(self):
        """加载训练过程中保存的最佳模型"""
        self.model.load_state_dict(torch.load("best_model.pth"))
        return self.model 