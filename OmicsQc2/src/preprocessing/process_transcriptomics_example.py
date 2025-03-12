#!/usr/bin/env python
# -*- coding: utf-8 -*-
#python OmicsQc2/src/preprocessing/process_transcriptomics_example.py --input test --source-type geo --normalize --impute --output test_output.tsv
"""
转录组数据收集与预处理示例脚本
功能：演示如何收集转录组数据并进行预处理
"""

import os
import sys
import logging
import pandas as pd
from pathlib import Path
import argparse

# 添加项目根目录到Python路径
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent.resolve()
sys.path.append(str(project_root))

# 导入项目模块
from src.preprocessing.data_collector import DataCollector
from src.preprocessing.preprocessor import Preprocessor
from src.utils.logger import Logger

def setup_logging():
    """设置日志系统"""
    log_dir = os.path.join(project_root, 'logs')
    logger = Logger(log_dir=log_dir)
    return logger.setup('transcriptomics_example')

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='转录组数据收集与预处理示例')
    parser.add_argument('--input', type=str, required=True,
                       help='输入数据源（本地文件路径、GEO ID或URL）')
    parser.add_argument('--source-type', type=str, choices=['auto', 'local', 'geo', 'url'],
                       default='auto', help='数据源类型')
    parser.add_argument('--normalize', action='store_true', 
                       help='是否进行数据标准化')
    parser.add_argument('--filter', action='store_true',
                       help='是否过滤低表达基因')
    parser.add_argument('--threshold', type=float, default=10.0,
                       help='过滤阈值，默认为10')
    parser.add_argument('--impute', action='store_true',
                       help='是否填充缺失值')
    parser.add_argument('--output', type=str, default='processed_transcriptomics.tsv',
                       help='输出文件名')
    
    return parser.parse_args()

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置日志
    logger = setup_logging()
    logger.info("开始转录组数据收集与预处理")
    
    try:
        # 1. 收集转录组数据
        logger.info(f"从 {args.input} 收集转录组数据，类型: {args.source_type}")
        collector = DataCollector(data_dir=os.path.join(project_root, 'raw'))
        
        # 收集数据，并保存原始数据到raw目录
        raw_data = collector.collect_transcriptomics(
            source_path=args.input,
            output_file=f"raw_{args.output}",
            source_type=args.source_type
        )
        
        logger.info(f"成功收集数据: {raw_data.shape[0]}行 × {raw_data.shape[1]}列")
        logger.info(f"数据预览:\n{raw_data.head()}")
        
        # 2. 预处理数据
        preprocessor = Preprocessor()
        processed_data = raw_data.copy()
        
        # 填充缺失值
        if args.impute:
            logger.info("填充缺失值")
            processed_data = preprocessor.impute_missing_values(processed_data)
        
        # 标准化处理
        if args.normalize:
            logger.info("进行数据标准化")
            processed_data = preprocessor.normalize_transcriptomics(processed_data)
            logger.info("标准化完成")
        
        # 过滤低表达基因
        if args.filter:
            logger.info(f"过滤低表达基因，阈值: {args.threshold}")
            processed_data = preprocessor.filter_low_quality(processed_data, args.threshold)
            logger.info(f"过滤后剩余 {processed_data.shape[0]} 个基因")
        
        # 3. 保存处理后的数据
        processed_dir = os.path.join(project_root, 'processed')
        os.makedirs(processed_dir, exist_ok=True)
        
        output_path = os.path.join(processed_dir, args.output)
        processed_data.to_csv(output_path, sep='\t')
        
        logger.info(f"预处理完成，结果保存到: {output_path}")
        logger.info(f"处理后数据预览:\n{processed_data.head()}")
        
        # 输出处理统计信息
        logger.info("数据处理统计:")
        logger.info(f"  原始数据大小: {raw_data.shape}")
        logger.info(f"  处理后数据大小: {processed_data.shape}")
        if args.impute:
            logger.info(f"  缺失值填充: 已完成")
        if args.normalize:
            logger.info(f"  数据标准化: 已完成")
        if args.filter:
            logger.info(f"  低表达基因过滤 (阈值={args.threshold}): 已完成")
        
    except Exception as e:
        logger.error(f"数据处理过程中发生错误: {str(e)}", exc_info=True)
        sys.exit(1)
        
    logger.info("转录组数据收集与预处理流程完成")

if __name__ == "__main__":
    main() 