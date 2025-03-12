# src/data_loader.py
import pandas as pd
from pathlib import Path
from typing import Union, List
import chardet  # 新增编码自动检测


# -------------------------
# 增强版数据加载模块
# 版本: 2.1
# 更新日志:
# 1. 增加自动编码检测功能
# 2. 增强列名清洗规则
# 3. 添加路径类型验证
# 4. 支持更多分隔符类型
# -------------------------

def detect_file_encoding(file_path: Union[str, Path]) -> str:
    """自动检测文件编码"""
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read(10000))  # 读取前10KB进行检测
    return result['encoding']


def load_transcriptome_data(
        file_path: Union[str, Path],
        sep: str = r'\s*,\s*',
        sample_col_pattern: str = r'(sample|condition|s)\d+'
) -> pd.DataFrame:
    """
    加载转录组数据（支持中文路径和复杂分隔符）

    参数:
        file_path: 文件路径（支持带空格和中文）
        sep: 分隔符正则表达式，默认匹配带空格的逗号
        sample_col_pattern: 样本列名的正则匹配模式

    返回:
        标准化后的DataFrame
    """
    try:
        # 统一路径类型并验证存在性
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")

        # 动态检测编码
        encoding = detect_file_encoding(path)

        # 读取数据（支持混合分隔符）
        df = pd.read_csv(
            path,
            encoding=encoding,
            sep=sep,
            engine='python',
            dtype={'gene_id': str}  # 基因ID强制转为字符串
        )

        # 列名标准化处理
        df.columns = (
            df.columns.str.strip()  # 去除前后空格
            .str.lower()  # 全小写
            .str.replace(r'[^\w]+', '_', regex=True)  # 特殊字符转下划线
        )

        # 自动识别样本列（例如 sample1, conditionA 等）
        sample_cols = df.columns[df.columns.str.match(sample_col_pattern, case=False)]
        if len(sample_cols) < 2:
            raise ValueError(f"未检测到足够的样本列，匹配模式: {sample_col_pattern}")

        return df[['gene_id'] + sample_cols.tolist()]

    except pd.errors.ParserError as pe:
        raise ValueError(f"数据解析失败，请检查分隔符设置: {str(pe)}")
    except Exception as e:
        raise RuntimeError(f"加载数据失败: {str(e)}") from e


def minmax_normalize(
        df: pd.DataFrame,
        exclude_cols: List[str] = ['gene_id', 'gene_name'],
        feature_range: tuple = (0, 1)
) -> pd.DataFrame:
    """
    增强版最小-最大标准化

    参数:
        df: 输入数据框
        exclude_cols: 要跳过的列
        feature_range: 标准化范围，默认为(0,1)

    返回:
        标准化后的数据框
    """
    try:
        # 复制数据避免污染原始数据
        df_norm = df.copy()

        # 自动检测数值列
        numeric_cols = (
            df_norm.select_dtypes(include=['number'])
            .columns.difference(exclude_cols)
        )

        # 范围验证
        if not (0 <= feature_range[0] < feature_range[1] <= 1):
            raise ValueError("标准化范围必须在[0,1]区间且min < max")

        # 逐列标准化
        for col in numeric_cols:
            col_min = df_norm[col].min()
            col_max = df_norm[col].max()

            # 处理常数列
            if col_max == col_min:
                df_norm[col] = feature_range[0]
                continue

            # 线性缩放
            df_norm[col] = (
                    (df_norm[col] - col_min) / (col_max - col_min)
                    * (feature_range[1] - feature_range[0])
                    + feature_range[0]
            )

        return df_norm

    except KeyError as ke:
        raise ValueError(f"列不存在: {str(ke)}")
    except Exception as e:
        raise RuntimeError(f"标准化失败: {str(e)}") from e


# 单元测试（可直接运行验证）
if __name__ == "__main__":
    # 测试中文路径和空格处理
    test_path = Path("D:\\学习软件\\pycharm\\pyCharm-数据\\设计成品\\一键化格式2\\OmicsQC\\data\\raw\\transcriptome_sample.csv")

    try:
        df = load_transcriptome_data(test_path)
        print("数据加载成功，维度:", df.shape)

        df_norm = minmax_normalize(df)
        print("标准化完成，统计描述:\n", df_norm.describe())

    except Exception as e:
        print("测试失败:", str(e))
