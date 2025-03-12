# src/main.py
import sys
import sqlite3
import pandas as pd
from pathlib import Path
import logging

# 修复路径拼接问题
PROJECT_ROOT = Path(__file__).parent.parent.resolve()  # 使用resolve处理路径规范化
sys.path.append(str(PROJECT_ROOT))

# 自定义模块导入
from src.data_loader import load_transcriptome_data, minmax_normalize
from src.visualization import plot_expression_distribution
from src.database import init_database

# -------------------------
# 修正日志配置（关键修复点）
# -------------------------
# 确保日志目录存在
log_dir = PROJECT_ROOT / "logs"
log_dir.mkdir(parents=True, exist_ok=True)  # 自动创建logs目录

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "main.log", encoding='utf-8'),  # 指定编码
        logging.StreamHandler()
    ]
)


def main():
    """主流程控制器"""
    try:
        # 动态路径配置（处理空格问题）
        raw_path = PROJECT_ROOT / "D:\\学习软件\\pycharm\\pyCharm-数据\\设计成品\\一键化格式2\\OmicsQC\\data\\raw\\transcriptome_sample.csv"
        db_path = PROJECT_ROOT / "D:\\学习软件\\pycharm\\pyCharm-数据\\设计成品\\一键化格式2\\OmicsQC\\data\\omics_qc.db"

        # 初始化数据库
        init_database(db_path)
        logging.info("✅ 数据库初始化完成")

        # 数据加载（处理中文路径）
        raw_data = load_transcriptome_data(raw_path)
        logging.info(f"📥 加载数据: {raw_data.shape}")

        # ...（后续流程保持不变）

    except Exception as e:
        logging.error(f"❌ 致命错误: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
