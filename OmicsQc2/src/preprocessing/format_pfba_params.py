"""
转录组数据格式化为pFBA参数
职责：将转录组数据转换为pFBA模型所需的参数格式
"""

import os
import json
import pandas as pd
import numpy as np
import logging
from pathlib import Path

# 设置logger
logger = logging.getLogger(__name__)

def format_pfba_parameters(expression_data: pd.DataFrame,
                         expression_threshold: float = 1.0,
                         output_file: str = None) -> dict:
    """
    将转录组数据格式化为pFBA参数
    
    参数:
        expression_data: 转录组表达数据DataFrame
        expression_threshold: 基因表达阈值
        output_file: 输出文件路径
        
    返回:
        pFBA参数字典
    """
    logger.info("开始格式化pFBA参数")
    
    try:
        # 计算每个基因的平均表达值
        mean_expression = expression_data.mean(axis=1)
        
        # 根据阈值划分基因
        expressed_genes = mean_expression[mean_expression >= expression_threshold].index.tolist()
        non_expressed_genes = mean_expression[mean_expression < expression_threshold].index.tolist()
        
        logger.info(f"表达基因数: {len(expressed_genes)}")
        logger.info(f"未表达基因数: {len(non_expressed_genes)}")
        
        # 计算生长速率（使用表达基因的比例作为简单估计）
        growth_rate = len(expressed_genes) / len(expression_data.index)
        logger.info(f"估计的生长速率: {growth_rate:.4f}")
        
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
            },
            "substrate_constraints": {
                "glucose": {
                    "lower_bound": 0.0,
                    "upper_bound": 10.0
                },
                "glutamine": {
                    "lower_bound": 0.0,
                    "upper_bound": 10.0
                },
                "oxygen": {
                    "lower_bound": 0.0,
                    "upper_bound": 1.05
                }
            }
        }
        
        # 如果指定了输出文件，保存参数
        if output_file:
            output_dir = os.path.dirname(output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                
            with open(output_file, 'w') as f:
                json.dump(pfba_params, f, indent=2)
            logger.info(f"参数已保存到: {output_file}")
        
        return pfba_params
        
    except Exception as e:
        logger.error(f"格式化pFBA参数时出错: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="格式化转录组数据为pFBA参数")
    parser.add_argument("--input", required=True, help="输入转录组数据文件路径")
    parser.add_argument("--output", required=True, help="输出参数文件路径")
    parser.add_argument("--threshold", type=float, default=1.0, help="基因表达阈值")
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 读取数据
    expression_data = pd.read_csv(args.input, sep='\t', index_col=0)
    
    # 格式化参数
    format_pfba_parameters(expression_data, args.threshold, args.output) 