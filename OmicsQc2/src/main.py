#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OmicsQc2主程序入口
职责：提供命令行接口，协调各模块功能执行
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到 Python 路径
script_path = Path(__file__).resolve()
print(f"脚本路径: {script_path}")

# 确保使用正确的项目根目录
if script_path.parent.name == 'src' and script_path.parent.parent.name == 'OmicsQc2':
    project_root = script_path.parent.parent
else:
    # 向上查找直到找到OmicsQc2目录
    current_path = script_path.parent
    while current_path.name != 'OmicsQc2' and current_path != current_path.parent:
        current_path = current_path.parent
    project_root = current_path

print(f"项目根目录: {project_root}")
sys.path.append(str(project_root))

# 导入项目模块
from utils.logger import Logger
from utils.config_manager import ConfigManager

def setup_logging():
    """设置日志系统"""
    log_dir = os.path.join(project_root, 'logs')
    logger = Logger(log_dir=log_dir)
    return logger.setup('omicsqc2')

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='OmicsQc2 - 组学数据质控与分析流程')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='配置文件路径')
    parser.add_argument('--mode', type=str, choices=['preprocess', 'qc', 'model', 'pipeline'],
                        default='pipeline', help='运行模式')
    parser.add_argument('--input', type=str, help='输入文件或目录')
    parser.add_argument('--output', type=str, help='输出文件或目录')
    
    return parser.parse_args()

def run_preprocess(config, input_path, output_path, logger):
    """运行数据预处理"""
    logger.info("开始数据预处理")
    # TODO: 实现预处理逻辑
    logger.info("数据预处理完成")

def run_qc(config, input_path, output_path, logger):
    """运行质量控制"""
    logger.info("开始质量控制")
    # TODO: 实现质量控制逻辑
    logger.info("质量控制完成")

def run_model(config, input_path, output_path, logger):
    """运行模型分析"""
    logger.info("开始模型分析")
    # TODO: 实现模型分析逻辑
    logger.info("模型分析完成")

def run_pipeline(config, input_path, output_path, logger):
    """运行完整流程"""
    logger.info("开始运行完整流程")
    
    # 检查工作目录结构
    for dir_name in ['raw', 'processed', 'outputs', 'logs']:
        os.makedirs(os.path.join(project_root, dir_name), exist_ok=True)
    
    # 依次执行各步骤
    run_preprocess(config, input_path, 'processed', logger)
    run_qc(config, 'processed', 'outputs', logger)
    run_model(config, 'processed', 'outputs', logger)
    
    logger.info("完整流程运行完成")

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置日志
    logger = setup_logging()
    logger.info("OmicsQc2 启动")
    
    # 加载配置
    # 直接使用项目根目录下的config/config.yaml
    config_path = os.path.join(project_root, 'config', 'config.yaml')
    print(f"尝试加载配置文件: {config_path}")
    logger.info(f"尝试加载配置文件: {config_path}")
    
    # 确保配置文件存在
    if not os.path.exists(config_path):
        # 尝试其他可能的路径
        alt_config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml'))
        print(f"配置文件不存在，尝试备选路径: {alt_config_path}")
        logger.warning(f"配置文件 {config_path} 不存在，尝试备选路径: {alt_config_path}")
        
        if os.path.exists(alt_config_path):
            config_path = alt_config_path
        else:
            logger.error(f"配置文件不存在，程序退出")
            sys.exit(1)
    
    logger.info(f"使用配置文件: {config_path}")    
    try:
        # 直接使用绝对路径加载配置文件
        config_manager = ConfigManager()
        config = config_manager.load_yaml(config_path)
        logger.info(f"成功加载配置文件: {config_path}")
    except Exception as e:
        logger.error(f"加载配置文件失败: {str(e)}")
        sys.exit(1)
    
    # 根据模式执行相应功能
    if args.mode == 'preprocess':
        run_preprocess(config, args.input, args.output, logger)
    elif args.mode == 'qc':
        run_qc(config, args.input, args.output, logger)
    elif args.mode == 'model':
        run_model(config, args.input, args.output, logger)
    elif args.mode == 'pipeline':
        run_pipeline(config, args.input, args.output, logger)
    
    logger.info("OmicsQc2 执行完成")

if __name__ == "__main__":
    main() 