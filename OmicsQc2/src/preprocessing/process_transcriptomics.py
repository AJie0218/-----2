#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
转录组数据处理脚本
职责：收集和处理转录组数据
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(project_root))

from src.preprocessing.data_collector import DataCollector
from src.preprocessing.preprocessor import Preprocessor
from src.utils.logger import Logger

def setup_logging():
    """设置日志系统"""
    log_dir = os.path.join(project_root, 'logs')
    logger = Logger(log_dir=log_dir)
    return logger.setup('process_transcriptomics')

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='转录组数据处理工具')
    parser.add_argument('--input', type=str, required=True,
                        help='输入数据源，可以是本地文件路径、GEO ID或URL')
    parser.add_argument('--output', type=str, default='transcriptomics.tsv',
                        help='输出文件名')
    parser.add_argument('--source-type', type=str, choices=['auto', 'local', 'geo', 'url'],
                        default='auto', help='数据源类型')
    parser.add_argument('--normalize', action='store_true',
                        help='是否进行数据标准化')
    parser.add_argument('--filter', action='store_true',
                        help='是否过滤低表达基因')
    parser.add_argument('--min-count', type=float, default=10.0,
                        help='低表达过滤阈值，默认为10')
    
    return parser.parse_args()

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置日志
    logger = setup_logging()
    logger.info("开始处理转录组数据")
    
    try:
        # 创建数据收集器
        collector = DataCollector(data_dir=os.path.join(project_root, 'raw'))
        
        # 收集数据
        logger.info(f"从{args.input}收集数据，类型: {args.source_type}")
        data = collector.collect_transcriptomics(
            source_path=args.input,
            output_file=args.output,
            source_type=args.source_type
        )
        
        logger.info(f"成功读取数据: {data.shape[0]}行 × {data.shape[1]}列")
        
        # 是否进行标准化处理
        if args.normalize or args.filter:
            preprocessor = Preprocessor()
            
            # 标准化处理
            if args.normalize:
                logger.info("执行数据标准化")
                data = preprocessor.normalize_transcriptomics(data)
            
            # 过滤低表达基因
            if args.filter:
                logger.info(f"过滤低表达基因，阈值: {args.min_count}")
                # 假设preprocessor有一个filter_low_expression方法
                data = preprocessor.filter_low_quality(data, threshold=args.min_count)
        
        # 保存最终处理结果
        processed_path = os.path.join(project_root, 'processed', args.output)
        data.to_csv(processed_path, sep='\t', index=True)
        logger.info(f"数据处理完成，结果保存至: {processed_path}")
        
        # 打印数据预览
        logger.info("\n数据预览:\n" + str(data.head()))
        
    except Exception as e:
        logger.error(f"处理过程中发生错误: {str(e)}", exc_info=True)
        sys.exit(1)
    
    logger.info("转录组数据处理完成")

if __name__ == "__main__":
    main() 