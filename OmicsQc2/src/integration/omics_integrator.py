"""
多组学数据整合模块
职责：整合转录组、蛋白组和代谢组数据
"""

import pandas as pd

class OmicsIntegrator:
    def __init__(self):
        """初始化整合器"""
        pass

    def integrate_data(self, transcriptomics, proteomics, metabolomics):
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