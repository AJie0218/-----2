# src/visualization.py
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path


def plot_expression_distribution(
        df: pd.DataFrame,
        save_path: Path,
        log_scale: bool = True,
        sample_cols: list = None
):
    """绘制表达量分布图"""
    plt.figure(figsize=(12, 6))

    # 自动选择样本列（排除基因ID和名称）
    if not sample_cols:
        sample_cols = df.columns[df.columns.str.startswith('sample')]

    # 长格式转换
    df_melt = df.melt(
        id_vars=df.columns.difference(sample_cols),
        value_vars=sample_cols,
        var_name='sample',
        value_name='expression'
    )

    # 绘制分布图
    sns.violinplot(
        x='sample',
        y='expression',
        data=df_melt,
        inner="quartile",
        palette="Set3"
    )

    if log_scale:
        plt.yscale('log')
        plt.ylabel('Expression (log scale)')
    else:
        plt.ylabel('Expression (linear scale)')

    plt.title('Transcriptome Expression Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
