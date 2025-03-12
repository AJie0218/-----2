"""
日志工具模块
职责：提供统一的日志记录功能
"""

import logging
import os
from datetime import datetime

class Logger:
    def __init__(self, log_dir='logs', log_level=logging.INFO):
        """
        初始化日志器
        
        参数:
            log_dir: 日志目录
            log_level: 日志级别
        """
        self.log_dir = log_dir
        self.log_level = log_level
        self.logger = None
        
        # 确保日志目录存在
        os.makedirs(log_dir, exist_ok=True)
        
    def setup(self, logger_name):
        """
        设置日志器
        
        参数:
            logger_name: 日志器名称
            
        返回:
            配置好的日志器
        """
        # 创建日志器
        logger = logging.getLogger(logger_name)
        logger.setLevel(self.log_level)
        
        # 清除已有处理器
        if logger.handlers:
            logger.handlers.clear()
            
        # 创建日志文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"{logger_name}_{timestamp}.log")
        
        # 创建文件处理器
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(self.log_level)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.log_level)
        
        # 设置日志格式
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        self.logger = logger
        return logger
        
    def get_logger(self):
        """获取日志器"""
        if not self.logger:
            raise ValueError("日志器尚未设置，请先调用setup方法")
        return self.logger 