import pandas as pd
from pathlib import Path
import sys


def validate_columns(df):
    """动态适配多语言列名（兼容中文列名）"""
    col_mapping = {
        'gene_id': ['GeneID', '基因ID', 'Gene_ID'],
        'sample_A': ['Sample1', '样品A', 'ConditionA'],
        'sample_B': ['Sample2', '样品B', 'ConditionB']
    }

    found_cols = {}
    for target, candidates in col_mapping.items():
        for col in df.columns:
            if any(cand.lower() in col.lower() for cand in candidates):
                found_cols[target] = col
                break
        else:
            raise KeyError(f"未找到匹配列：{target}，候选列名：{candidates}")

    return df.rename(columns=found_cols)[['gene_id', 'sample_A', 'sample_B']]


def preprocess():
    try:
        # 动态路径解析
        base_dir = Path(__file__).parent.parent
        raw_path = base_dir / "data/raw/transcriptome_sample.csv"
        processed_path = base_dir / "data/processed/condition_data.csv"

        # 处理混合编码和分隔符问题
        df = pd.read_csv(raw_path, encoding='gbk', engine='python', sep=r'\s*,|\t', regex=True)

        # 列名标准化
        df_clean = validate_columns(df)

        # 保存为Parquet格式提升IO性能
        df_clean.to_parquet(processed_path)
        print(f"✅ 数据预处理完成 -> {processed_path}")

    except Exception as e:
        print(f"❌ 预处理失败：{str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    preprocess()
