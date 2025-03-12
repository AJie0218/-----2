# OmicsQc2

## 项目概述

OmicsQc2是一个用于组学数据（转录组、蛋白组、代谢组）的质量控制、预处理和整合分析平台。该平台实现了从原始数据收集、预处理、质量控制到代谢通量分析的全流程自动化，支持pFBA (parsimonious Flux Balance Analysis) 和FVA (Flux Variability Analysis) 模型分析，同时为机器学习模型提供标准化的数据输入。

## 功能特点

1. **多组学数据处理**
   - 支持转录组、蛋白组、代谢组数据的统一处理
   - 提供灵活的数据格式转换和标准化方法
   - 实现多组学数据的整合分析

2. **全面质量控制**
   - 自动检测异常样本和异常值
   - 提供缺失值分析和处理方法
   - 生成详细的质量控制报告

3. **代谢网络模型分析**
   - 集成pFBA和FVA模型实现代谢通量分析
   - 支持基因敲除模拟
   - 为知识驱动模型提供参数输入

4. **自动化工作流**
   - 提供Snakemake和Nextflow两种工作流实现
   - 实现从原始数据到模型输入的一键式处理
   - 支持大规模数据批处理

## 系统架构

```
OmicsQc2/
├── raw/                   # 原始数据存储
├── processed/             # 预处理后的数据
├── src/                   # 源代码
│   ├── preprocessing/     # 数据预处理模块
│   ├── qc/                # 质量控制模块
│   ├── integration/       # 数据整合模块
│   └── utils/             # 工具函数
├── pipelines/             # 工作流定义
│   ├── snakemake/         # Snakemake工作流
│   └── nextflow/          # Nextflow工作流
├── config/                # 配置文件
├── models/                # 模型相关代码
│   ├── pfba/              # pFBA模型
│   ├── fva/               # FVA模型
│   └── ml/                # 机器学习模型
├── outputs/               # 输出结果
├── docs/                  # 文档
└── tests/                 # 测试代码
```

## 使用指南

### 安装依赖

```bash
# 创建conda环境（推荐）
conda create -n omicsqc2 python=3.8
conda activate omicsqc2

# 安装依赖包
pip install -r requirements.txt
```

### 配置

编辑`config/config.yaml`文件，配置数据路径、预处理参数和模型参数等。

### 运行方式

#### 命令行运行

```bash
# 运行完整流程
python src/main.py --mode pipeline --config config/config.yaml

# 仅运行数据预处理
python src/main.py --mode preprocess --input raw/data.csv --output processed/

# 仅运行质量控制
python src/main.py --mode qc --input processed/ --output outputs/

# 仅运行模型分析
python src/main.py --mode model --input processed/ --output outputs/
```

#### 使用Snakemake工作流

```bash
cd pipelines/snakemake
snakemake --cores 4
```

#### 使用Nextflow工作流

```bash
cd pipelines/nextflow
nextflow run main.nf
```

## 数据示例

`raw/`目录中提供了示例数据文件：
- `transcriptome_sample.csv`: 转录组数据示例
- `proteomics_sample.csv`: 蛋白组数据示例 (待添加)
- `metabolomics_sample.csv`: 代谢组数据示例 (待添加)

## 开发者指南

请参考`docs/`目录下的详细开发文档。 