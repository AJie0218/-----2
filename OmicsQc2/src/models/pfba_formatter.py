"""
pFBA数据格式化模块
职责：将转录组数据格式化为pFBA模型所需的参数格式
"""

import os
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path

# 设置logger
logger = logging.getLogger(__name__)

class PFBAFormatter:
    def __init__(self, output_dir="models/pfba/data"):
        """
        初始化pFBA数据格式化器
        
        参数:
            output_dir: 格式化后数据的输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def format_transcriptomics(self, transcriptomics_data, threshold=0.5, growth_rate=None):
        """
        将转录组数据格式化为pFBA模型所需的反应约束
        
        参数:
            transcriptomics_data: 转录组数据DataFrame，行为基因，列为样本
            threshold: 基因表达阈值，低于此阈值的基因视为不表达
            growth_rate: 生长速率，如果为None则从数据中估计
            
        返回:
            pFBA参数字典
        """
        logger.info("开始格式化转录组数据为pFBA参数")
        
        # 计算每个基因的平均表达量
        if isinstance(transcriptomics_data, pd.DataFrame):
            gene_expression = transcriptomics_data.mean(axis=1)
        else:
            raise ValueError("转录组数据必须是pandas DataFrame格式")
        
        # 根据阈值确定表达/不表达的基因
        expressed_genes = gene_expression[gene_expression > threshold].index.tolist()
        non_expressed_genes = gene_expression[gene_expression <= threshold].index.tolist()
        
        logger.info(f"表达基因数量: {len(expressed_genes)}")
        logger.info(f"非表达基因数量: {len(non_expressed_genes)}")
        
        # 如果未提供生长速率，使用表达基因的比例估算
        if growth_rate is None:
            growth_rate = len(expressed_genes) / len(gene_expression)
            logger.info(f"估算的生长速率: {growth_rate:.4f}")
        else:
            logger.info(f"使用提供的生长速率: {growth_rate:.4f}")
        
        # 构建pFBA参数字典
        pfba_params = {
            "model_constraints": {
                "expressed_genes": expressed_genes,
                "non_expressed_genes": non_expressed_genes,
                "growth_rate": growth_rate
            },
            "simulation_parameters": {
                "objective_function": "biomass_reaction",
                "minimize_total_flux": True
            }
        }
        
        return pfba_params
    
    def format_media_composition(self, media_composition):
        """
        格式化培养基成分信息为pFBA模型所需的底物摄取约束
        
        参数:
            media_composition: 培养基成分字典，键为代谢物ID，值为浓度
            
        返回:
            底物摄取约束字典
        """
        logger.info("格式化培养基成分信息")
        
        # 构建底物摄取约束
        substrate_constraints = {}
        
        for metabolite, concentration in media_composition.items():
            # 将浓度转换为通量上限
            # 注意：这里使用简化的转换，真实应用中可能需要更复杂的计算
            flux_upper_bound = min(10.0, concentration * 5.0)  # 简化的转换
            
            substrate_constraints[metabolite] = {
                "lower_bound": 0.0,  # 假设只有摄取，无分泌
                "upper_bound": flux_upper_bound
            }
        
        logger.info(f"格式化了 {len(substrate_constraints)} 个底物约束")
        
        return {"substrate_constraints": substrate_constraints}
    
    def format_gene_knockouts(self, knockout_genes):
        """
        格式化基因敲除信息
        
        参数:
            knockout_genes: 要敲除的基因列表
            
        返回:
            基因敲除信息字典
        """
        logger.info(f"格式化基因敲除信息: {len(knockout_genes)} 个基因")
        
        return {"knockout_genes": knockout_genes}
    
    def save_params(self, params, output_name="pfba_params.json"):
        """
        保存pFBA参数到文件
        
        参数:
            params: pFBA参数字典
            output_name: 输出文件名
            
        返回:
            输出文件路径
        """
        output_path = os.path.join(self.output_dir, output_name)
        
        with open(output_path, 'w') as f:
            json.dump(params, f, indent=2)
            
        logger.info(f"pFBA参数已保存至: {output_path}")
        
        return output_path
    
    def format_all(self, transcriptomics_data, media_composition=None, 
                  knockout_genes=None, growth_rate=None, threshold=0.5,
                  output_name="pfba_params.json"):
        """
        一次性格式化所有pFBA所需参数
        
        参数:
            transcriptomics_data: 转录组数据
            media_composition: 培养基成分
            knockout_genes: 敲除基因列表
            growth_rate: 生长速率
            threshold: 基因表达阈值
            output_name: 输出文件名
            
        返回:
            pFBA参数字典和输出文件路径
        """
        # 格式化转录组数据
        params = self.format_transcriptomics(
            transcriptomics_data, 
            threshold=threshold,
            growth_rate=growth_rate
        )
        
        # 格式化培养基成分（如果提供）
        if media_composition:
            media_params = self.format_media_composition(media_composition)
            params.update(media_params)
        
        # 格式化基因敲除信息（如果提供）
        if knockout_genes:
            knockout_params = self.format_gene_knockouts(knockout_genes)
            params.update(knockout_params)
        
        # 保存参数到文件
        output_path = self.save_params(params, output_name)
        
        return params, output_path 