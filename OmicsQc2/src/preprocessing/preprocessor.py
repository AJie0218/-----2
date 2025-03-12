"""
数据预处理模块
职责：对收集的原始数据进行预处理和标准化
"""

class Preprocessor:
    def __init__(self):
        """初始化预处理器"""
        pass
        
    def normalize_transcriptomics(self, data):
        """
        标准化转录组数据
        
        参数:
            data: 原始转录组数据
            
        返回:
            标准化后的转录组数据
        """
        # 使用Z-score标准化
        return (data - data.mean()) / data.std()
        
    def normalize_proteomics(self, data):
        """
        标准化蛋白组数据
        
        参数:
            data: 原始蛋白组数据
            
        返回:
            标准化后的蛋白组数据
        """
        # 使用Min-Max标准化
        return (data - data.min()) / (data.max() - data.min())
        
    def normalize_metabolomics(self, data):
        """
        标准化代谢组数据
        
        参数:
            data: 原始代谢组数据
            
        返回:
            标准化后的代谢组数据
        """
        # 使用Z-score标准化
        return (data - data.mean()) / data.std()
        
    def filter_low_quality(self, data, threshold):
        """
        过滤低质量数据
        
        参数:
            data: 原始数据
            threshold: 质量阈值
            
        返回:
            过滤后的数据
        """
        # 过滤低于阈值的行
        return data[data.mean(axis=1) > threshold]
        
    def impute_missing_values(self, data):
        """
        填充缺失值
        
        参数:
            data: 包含缺失值的数据
            
        返回:
            填充后的数据
        """
        # 使用均值填充缺失值
        return data.fillna(data.mean())
        
    def merge_omics_data(self, transcriptomics, proteomics, metabolomics):
        """
        整合多组学数据
        
        参数:
            transcriptomics: 转录组数据
            proteomics: 蛋白组数据
            metabolomics: 代谢组数据
            
        返回:
            整合后的多组学数据
        """
        # 使用基因ID作为索引进行合并
        return transcriptomics.join([proteomics, metabolomics], how='outer') 