import torch#PyTorch 的主要模块，分别用于张量操作、构建神经网络层和定义优化算法。
import torch.nn as nn#定义神经网络层
import torch.optim as optim#定义优化算法
from torch.utils.data import Dataset, DataLoader #定义数据集和数据加载器
import numpy as np#用于数值计算
from cobra.io import load_model, load_json_model, read_sbml_model#用于加载代谢模型
from cobra.flux_analysis import pfba#用于运行PFBA分析
import pandas as pd#用于数据处理        
from sklearn.preprocessing import StandardScaler#用于数据标准化
from sklearn.model_selection import train_test_split#用于数据集划分

# #####################################
# 配置区（根据实际情况修改）
# #####################################
CONFIG = {
    "data_path": "bulk_data.csv",          # 输入bulk数据路径
    "model_path": "Recon3D.xml",           # 代谢模型文件路径
    "output_activation": "sigmoid",         # 输出层激活函数类型：sigmoid或linear
    "hidden_layers": [256, 256],           # 隐藏层结构：两个256节点的隐藏层
    "dropout_rate": 0.2,                   # Dropout比率：防止过拟合
    "batch_size": 64,                      # 批处理大小
    "lr": 1e-4,                            # 学习率
    "epochs": 500,                         # 训练轮数
    "early_stop_patience": 20,             # 早停耐心值：验证损失多少轮未改善就停止
    "device": "cuda" if torch.cuda.is_available() else "cpu",  # 训练设备：GPU或CPU
    "flux_constraint_weight": 0.1,         # 通量约束权重：用于正则化
    "temperature": 2.0                     # 知识蒸馏温度参数
}

# #####################################
# 数据预处理模块
# #####################################
class MetabolicDataset(Dataset):    
    """自定义数据集类，继承自PyTorch的Dataset类"""
    def __init__(self, features, targets):
        """
        初始化数据集
        Args:
            features: 输入特征数据
            targets: 目标值数据
        """
        self.features = torch.FloatTensor(features)  # 将特征转换为PyTorch浮点张量
        self.targets = torch.FloatTensor(targets)    # 将目标值转换为PyTorch浮点张量
        
    def __len__(self):   # 返回数据集的长度
        return len(self.features)   
    
    def __getitem__(self, idx):
        """
        获取指定索引的数据样本
        Args:
            idx: 样本索引
        Returns:
            tuple: (特征, 目标值)
        """
        return self.features[idx], self.targets[idx]


class DataPreprocessor:
    """数据预处理类，处理代谢数据和PFBA分析"""
    def __init__(self, config):
        """
        初始化预处理器
        Args:
            config: 配置字典，包含数据路径等参数
        """
        self.config = config
        self.scaler = StandardScaler()  # 用于特征标准化
        try:
            # 尝试从本地XML文件加载模型
            self.metabolic_model = read_sbml_model(config["model_path"])     # 读取代谢模型
        except Exception as e:
            raise RuntimeError(f"无法加载代谢模型，请确保模型文件存在或格式正确: {str(e)}")
        
    def _run_pfba(self, input_data):
        """
        对输入数据进行PFBA（Parsimonious Flux Balance Analysis）分析
        Args:
            input_data: 输入数据矩阵
        Returns:
            numpy.ndarray: PFBA分析结果
        """
        flux_data = []
        n_reactions = len(self.metabolic_model.reactions)#recon3d模型中的反应数量
        
        for sample in input_data:
            with self.metabolic_model as model:
                # 检查输入维度
                if len(sample) < n_reactions:
                    print(f"警告：输入数据维度({len(sample)})小于模型反应数量({n_reactions})")
                    # 对于缺失的维度，使用默认边界值
                    for i, rxn in enumerate(model.reactions):
                        if i < len(sample):
                            rxn.bounds = (sample[i], sample[i])
                        else:
                            rxn.bounds = (-1000, 1000)  # 使用默认边界
                else:
                    # 只使用前n_reactions个维度
                    for i, rxn in enumerate(model.reactions):
                        rxn.bounds = (sample[i], sample[i])
                
                try:
                    # 设置特定反应的约束，这部分可以用先验知识设定
                    if 'EX_glc_D_e' in model.reactions:
                        model.reactions.EX_glc_D_e.bounds = (-10, 0)  # 葡萄糖摄入约束
                    if 'EX_o2_e' in model.reactions:
                        model.reactions.EX_o2_e.bounds = (-20, 0)  # 氧气摄入约束
                    
                    # 执行PFBA分析
                    pfba_result = pfba(model)
                    flux_data.append([pfba_result.fluxes[rxn.id] for rxn in model.reactions])
                except Exception as e:
                    print(f"PFBA求解失败: {str(e)}")
                    flux_data.append([0] * n_reactions)  # 失败时使用零通量
                
        return np.array(flux_data)
    
    def prepare_data(self):
        """
        准备训练、验证和测试数据集
        Returns:
            tuple: (训练集, 验证集, 测试集)，每个都是MetabolicDataset实例
        """
        # 加载并标准化数据
        raw_data = pd.read_csv(self.config["data_path"]).values
        scaled_data = self.scaler.fit_transform(raw_data)
        
        # 生成PFBA标签
        flux_labels = self._run_pfba(scaled_data)
        
        # 数据集划分（70% 训练，15% 验证，15% 测试）
        X_train, X_temp, y_train, y_temp = train_test_split(
            scaled_data, flux_labels, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42)
            
        return MetabolicDataset(X_train, y_train), \
               MetabolicDataset(X_val, y_val), \
               MetabolicDataset(X_test, y_test)

# #####################################
# 模型架构
# #####################################
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

# #####################################
# 改进的损失函数
# #####################################
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

# #####################################
# 训练引擎
# #####################################
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
    
    def run_training(self):
        """
        执行完整的训练过程
        包括训练、验证和早停
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
                if patience_counter >= self.config["early_stop_patience"]:
                    print(f"早停：在epoch {epoch}处停止训练")
                    break
            
            # 打印训练进度
            print(f"Epoch {epoch+1}/{self.config['epochs']} | "
                  f"训练损失: {train_loss:.4f} | 验证损失: {val_loss:.4f} | "
                  f"学习率: {self.optimizer.param_groups[0]['lr']:.2e}")

# #####################################
# 主程序
# #####################################
if __name__ == "__main__":
    # 初始化训练器并执行训练
    trainer = Trainer(CONFIG)
    trainer.run_training()
    
    # 在测试集上评估最终性能
    test_loader = DataLoader(trainer.test_set, batch_size=CONFIG["batch_size"])
    final_loss = trainer.evaluate(test_loader)
    print(f"测试集损失: {final_loss:.4f}")