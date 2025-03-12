"""
配置管理模块
职责：加载、解析和管理配置文件
"""

import os
import yaml
import json

class ConfigManager:
    def __init__(self, config_dir='config'):
        """
        初始化配置管理器
        
        参数:
            config_dir: 配置文件目录
        """
        self.config_dir = config_dir
        self.config = {}
        
    def load_yaml(self, config_file):
        """
        加载YAML配置文件
        
        参数:
            config_file: 配置文件名
            
        返回:
            配置字典
        """
        # 检查config_file是否为绝对路径
        if os.path.isabs(config_file):
            file_path = config_file
        else:
            file_path = os.path.join(self.config_dir, config_file)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            self.config.update(config)
            return config
        except Exception as e:
            raise ValueError(f"加载配置文件 {file_path} 时出错: {str(e)}")
            
    def load_json(self, config_file):
        """
        加载JSON配置文件
        
        参数:
            config_file: 配置文件名
            
        返回:
            配置字典
        """
        file_path = os.path.join(self.config_dir, config_file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.config.update(config)
            return config
        except Exception as e:
            raise ValueError(f"加载配置文件 {file_path} 时出错: {str(e)}")
            
    def get(self, key, default=None):
        """
        获取配置值
        
        参数:
            key: 配置键
            default: 默认值
            
        返回:
            配置值
        """
        return self.config.get(key, default)
    
    def save_config(self, config, file_path, format='yaml'):
        """
        保存配置
        
        参数:
            config: 配置字典
            file_path: 文件路径
            format: 文件格式，'yaml'或'json'
            
        返回:
            是否保存成功
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            if format.lower() == 'yaml':
                with open(file_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, default_flow_style=False)
            elif format.lower() == 'json':
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"不支持的格式: {format}")
                
            return True
        except Exception as e:
            print(f"保存配置文件失败: {str(e)}")
            return False 