# 转录组数据收集与预处理指南

本文档介绍如何使用OmicsQc2收集转录组数据并进行预处理。

## 功能概述

OmicsQc2提供了一套完整的转录组数据处理流程，包括：

1. **数据收集**：支持从本地文件、GEO数据库或远程URL收集转录组数据
2. **数据预处理**：
   - 缺失值填充
   - 数据标准化（Z-score标准化）
   - 低表达基因过滤

## 使用方法

### 使用示例脚本

我们提供了一个示例脚本`process_transcriptomics_example.py`，您可以使用它来收集和处理转录组数据：

```bash
# 从本地文件收集数据并进行预处理
python src/preprocessing/process_transcriptomics_example.py --input raw/your_data.csv --normalize --filter --threshold 5

# 从GEO数据库收集数据
python src/preprocessing/process_transcriptomics_example.py --input GSE12345 --source-type geo --normalize --impute

# 从URL收集数据
python src/preprocessing/process_transcriptomics_example.py --input https://example.com/data.csv --source-type url --output my_processed_data.tsv
```

### 命令行参数

- `--input`：输入数据源（本地文件路径、GEO ID或URL）【必需】
- `--source-type`：数据源类型，可选值：`auto`、`local`、`geo`、`url`，默认为`auto`
- `--normalize`：是否进行数据标准化（Z-score标准化）
- `--filter`：是否过滤低表达基因
- `--threshold`：过滤阈值，默认为10
- `--impute`：是否填充缺失值
- `--output`：输出文件名，默认为`processed_transcriptomics.tsv`

## 数据流程

1. **数据收集**：
   - 脚本首先从指定的来源收集转录组数据
   - 原始数据会保存在`raw`目录下

2. **数据预处理**：
   - 根据命令行参数，对数据进行缺失值填充、标准化和低表达基因过滤
   - 处理过程会记录在日志文件中

3. **结果输出**：
   - 预处理后的数据会保存在`processed`目录下
   - 脚本会输出数据处理的统计信息和数据预览

## 示例数据处理流程

以下是一个完整的数据处理流程示例：

```bash
# 从GEO下载GSE63129数据集，进行标准化和过滤
python src/preprocessing/process_transcriptomics_example.py --input GSE63129 --source-type geo --normalize --filter --threshold 5 --output gse63129_processed.tsv
```

此命令会：
1. 从GEO下载GSE63129数据集
2. 对数据进行Z-score标准化
3. 过滤掉平均表达量低于5的基因
4. 将结果保存为`processed/gse63129_processed.tsv`

## 注意事项

- 处理大型数据集可能需要较长时间，尤其是从GEO下载数据时
- Z-score标准化会将数据转换为均值为0、标准差为1的分布
- 低表达基因过滤会移除平均表达量低于阈值的基因 