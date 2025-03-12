# src/database.py
import sqlite3
from pathlib import Path


def init_database(db_path: Path):
    """初始化数据库表结构"""
    conn = sqlite3.connect(db_path)

    # 创建元数据表
    conn.execute("""
    CREATE TABLE IF NOT EXISTS dataset_meta (
        dataset_id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        num_samples INTEGER,
        num_genes INTEGER
    )
    """)

    # 创建标准化数据表（由主函数动态写入）
    conn.commit()
    conn.close()
