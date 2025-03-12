#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
转录组数据处理与pFBA分析示例脚本
功能：演示如何收集转录组数据，处理为pFBA模型所需的参数，并运行pFBA分析
"""

import os
import sys
import logging
import pandas as pd
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent.resolve()
sys.path.append(str(project_root))

# 导入项目模块
from src.preprocessing.data_collector import DataCollector
from src.preprocessing.preprocessor import Preprocessor
from src.models.pfba_formatter import PFBAFormatter
from models.pfba.pfba_model import PFBAModel
from src.utils.logger import Logger

def setup_logging():
    """设置日志系统"""
    log_dir = os.path.join(project_root, 'logs')
    logger = Logger(log_dir=log_dir)
    return logger.setup('transcriptomics_pfba')

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='转录组数据处理与pFBA分析示例')
    parser.add_argument('--input', type=str, required=True,
                       help='输入数据源（本地文件路径、GEO ID或URL）')
    parser.add_argument('--source-type', type=str, choices=['auto', 'local', 'geo', 'url'],
                       default='auto', help='数据源类型')
    parser.add_argument('--normalize', action='store_true', 
                       help='是否进行数据标准化')
    parser.add_argument('--filter', action='store_true',
                       help='是否过滤低表达基因')
    parser.add_argument('--threshold', type=float, default=5.0,
                       help='基因表达阈值，默认为5.0')
    parser.add_argument('--growth-rate', type=float,
                       help='指定生长速率，若不指定则从数据估计')
    parser.add_argument('--model-path', type=str,
                       help='pFBA模型文件路径（SBML格式），若不指定则使用模拟模式')
    parser.add_argument('--output-prefix', type=str, default='pfba_results',
                       help='输出文件名前缀')
    
    return parser.parse_args()

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置日志
    logger = setup_logging()
    logger.info("开始转录组数据处理与pFBA分析")
    
    try:
        # 1. 收集转录组数据
        logger.info(f"从 {args.input} 收集转录组数据，类型: {args.source_type}")
        collector = DataCollector(data_dir=os.path.join(project_root, 'raw'))
        
        # 收集数据，并保存原始数据到raw目录
        raw_data = collector.collect_transcriptomics(
            source_path=args.input,
            output_file=f"raw_{args.output_prefix}.tsv",
            source_type=args.source_type
        )
        
        logger.info(f"成功收集数据: {raw_data.shape[0]}行 × {raw_data.shape[1]}列")
        
        # 2. 预处理数据
        preprocessor = Preprocessor()
        processed_data = raw_data.copy()
        
        # 标准化处理
        if args.normalize:
            logger.info("进行数据标准化")
            processed_data = preprocessor.normalize_transcriptomics(processed_data)
        
        # 过滤低表达基因
        if args.filter:
            logger.info(f"过滤低表达基因，阈值: {args.threshold}")
            processed_data = preprocessor.filter_low_quality(processed_data, args.threshold)
            logger.info(f"过滤后剩余 {processed_data.shape[0]} 个基因")
        
        # 3. 保存处理后的转录组数据
        processed_dir = os.path.join(project_root, 'processed')
        os.makedirs(processed_dir, exist_ok=True)
        
        processed_file = f"{args.output_prefix}_processed.tsv"
        processed_path = os.path.join(processed_dir, processed_file)
        processed_data.to_csv(processed_path, sep='\t')
        
        logger.info(f"预处理完成，转录组数据保存到: {processed_path}")
        
        # 4. 格式化为pFBA参数
        logger.info("将转录组数据格式化为pFBA参数")
        formatter = PFBAFormatter(output_dir=os.path.join(project_root, 'models', 'pfba', 'data'))
        
        # 模拟培养基成分信息
        media_composition = {
            "glucose": 10.0,     # 葡萄糖浓度(g/L)
            "glutamine": 2.0,    # 谷氨酰胺浓度(g/L)
            "oxygen": 0.21,      # 氧气浓度(比例)
        }
        
        # 模拟基因敲除
        knockout_genes = []  # 这里可以加入要敲除的基因，如GENE006
        
        # 格式化所有参数
        pfba_params, param_path = formatter.format_all(
            transcriptomics_data=processed_data,
            media_composition=media_composition,
            knockout_genes=knockout_genes,
            growth_rate=args.growth_rate,
            threshold=args.threshold,
            output_name=f"{args.output_prefix}_params.json"
        )
        
        logger.info(f"pFBA参数已保存至: {param_path}")
        
        # 5. 运行pFBA模型
        logger.info("运行pFBA模型")
        
        # 初始化模型
        model = PFBAModel(model_path=args.model_path)
        
        # 加载模型
        model.load_model()
        
        # 设置约束条件
        model.set_constraints(constraints_dict=pfba_params)
        
        # 如果有指定的基因敲除，执行敲除
        if knockout_genes:
            model.knockout_genes(knockout_genes)
        
        # 运行模拟
        results = model.run_simulation()
        
        # 导出结果
        outputs_dir = os.path.join(project_root, 'outputs')
        os.makedirs(outputs_dir, exist_ok=True)
        
        output_path = os.path.join(outputs_dir, f"{args.output_prefix}_results.json")
        model.export_results(output_path)
        
        logger.info(f"pFBA模拟完成，结果保存至: {output_path}")
        
        # 输出一些关键通量结果
        if "fluxes" in results:
            fluxes = results["fluxes"]
            if isinstance(fluxes, dict) and len(fluxes) > 0:
                top_fluxes = sorted(fluxes.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
                logger.info("前10个最大通量反应:")
                for rxn, flux in top_fluxes:
                    logger.info(f"  {rxn}: {flux:.4f}")
        
        logger.info(f"目标函数值: {results.get('objective_value', 'N/A')}")
        
    except Exception as e:
        logger.error(f"处理过程中发生错误: {str(e)}", exc_info=True)
        sys.exit(1)
        
    logger.info("转录组数据处理与pFBA分析流程完成")

if __name__ == "__main__":
    main() 