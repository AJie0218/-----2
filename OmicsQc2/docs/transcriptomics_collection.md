# 转录组数据收集指南

本文档介绍如何使用OmicsQc2的转录组数据收集功能。

## 功能概述

OmicsQc2提供了灵活的转录组数据收集功能，支持从多种来源收集数据：

- **本地文件**：CSV、TSV、TXT、Excel文件
- **GEO数据库**：直接通过GSE ID下载数据
- **远程URL**：支持各种远程数据文件

## 使用方法

### 命令行工具

使用内置的命令行工具进行数据收集和处理：

```bash
# 从本地文件收集数据
python src/preprocessing/process_transcriptomics.py --input path/to/your/data.csv --output transcriptomics.tsv

# 从GEO数据库收集数据
python src/preprocessing/process_transcriptomics.py --input GSE12345 --source-type geo --output gse_data.tsv

# 从URL收集数据
python src/preprocessing/process_transcriptomics.py --input https://example.com/data.csv --source-type url --output url_data.tsv

# 收集数据并进行标准化和过滤
python src/preprocessing/process_transcriptomics.py --input GSE12345 --normalize --filter --min-count 5
```

### Python API

您也可以在自己的Python脚本中使用数据收集功能：

```python
from src.preprocessing.data_collector import DataCollector

# 初始化数据收集器
collector = DataCollector(data_dir="raw", cache_dir=".cache")

# 从本地文件收集数据
df = collector.collect_transcriptomics(
    source_path="path/to/your/data.csv",
    output_file="local_data.tsv"
)

# 从GEO数据库收集数据
geo_df = collector.collect_transcriptomics(
    source_path="GSE12345",
    output_file="geo_data.tsv",
    source_type="geo"
)

# 处理数据...
```

## 支持的数据格式

### 本地文件

- **CSV文件**：逗号分隔的数据文件
- **TSV文件**：制表符分隔的数据文件
- **Excel文件**：.xlsx或.xls格式

文件应包含以下内容：
- 行：基因或转录本
- 列：样本
- 一个或多个列包含基因ID或基因名

### GEO数据

支持从GEO数据库直接下载矩阵数据。您只需提供GSE ID（例如GSE12345）。

### 远程URL

支持从远程URL下载数据文件，包括：
- 直接链接到CSV/TSV文件
- 直接链接到Excel文件
- 包含数据文件的ZIP压缩包

## 数据处理流程

1. **数据收集**：从指定来源获取原始数据
2. **格式检测**：自动检测文件格式、编码和分隔符
3. **数据清洗**：处理缺失值、重复行等
4. **格式转换**：将数据转换为标准格式（基因为行，样本为列）
5. **数据保存**：将收集的数据保存到指定位置

## 输出格式

输出的转录组数据为TSV格式，具有以下特点：
- 以基因/转录本作为行索引
- 样本作为列
- 表达量数据为数值型

## 注意事项

- 从GEO下载大型数据集可能需要较长时间
- 处理大型文件时可能需要较大内存
- 确保网络连接稳定，特别是从远程URL或GEO下载数据时 