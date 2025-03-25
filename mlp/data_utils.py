# data_utils.py
# 数据处理模块

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from cobra.io import read_sbml_model
from cobra.flux_analysis import pfba

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
        
    def __len__(self):
        """返回数据集中样本的数量"""
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
            # 加载代谢模型
            self.metabolic_model = read_sbml_model(config["model_path"])
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
        n_reactions = len(self.metabolic_model.reactions)
        
        for sample in input_data:
            with self.metabolic_model as model:
                # 检查输入维度
                if len(sample) < n_reactions:
                    print(f"警告：输入数据维度({len(sample)})小于模型反应数量({n_reactions})")
                    # 处理维度不足的情况
                    for i, rxn in enumerate(model.reactions):
                        if i < len(sample):
                            rxn.bounds = (sample[i], sample[i])
                        else:
                            rxn.bounds = (-1000, 1000)  # 使用默认边界
                else:
                    # 设置反应边界
                    for i, rxn in enumerate(model.reactions):
                        rxn.bounds = (sample[i], sample[i])
                
                try:
                    # 设置特定反应的约束
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
            
        return (MetabolicDataset(X_train, y_train),
                MetabolicDataset(X_val, y_val),
                MetabolicDataset(X_test, y_test)) 