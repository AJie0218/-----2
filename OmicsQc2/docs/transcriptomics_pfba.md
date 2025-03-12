# 转录组数据处理与pFBA分析指南

本文档介绍如何在OmicsQc2中将转录组数据处理为pFBA（parsimonious Flux Balance Analysis，简约通量平衡分析）模型所需的参数，并进行通量分析。

## 功能概述

OmicsQc2提供了一套完整的流程，可以：

1. **收集转录组数据**：从多种来源收集基因表达数据
2. **预处理数据**：标准化和过滤数据
3. **格式化为pFBA参数**：将处理后的数据转换为pFBA所需的参数格式
4. **运行pFBA模型**：使用格式化的参数运行pFBA分析
5. **输出分析结果**：以易于理解的格式输出通量分析结果

## pFBA模型参数概览

pFBA模型通常需要以下参数：

1. **代谢网络模型**：包含代谢物和反应的SBML格式模型文件
2. **基因表达约束**：基于转录组数据的基因表达/不表达信息
3. **生长速率约束**：细胞的生长速率
4. **培养基成分约束**：培养基中可用代谢物的限制
5. **基因敲除信息**：要敲除的基因列表

## 使用方法

### 使用示例脚本

OmicsQc2提供了一个示例脚本`process_transcriptomics_for_pfba.py`，可用于完整演示如何从转录组数据到pFBA分析的流程：

```bash
# 使用本地转录组数据进行pFBA分析
python src/examples/process_transcriptomics_for_pfba.py --input OmicsQc2/raw/test_transcriptomics.csv --normalize --filter --threshold 10.0

# 使用GEO数据库数据进行pFBA分析
python src/examples/process_transcriptomics_for_pfba.py --input GSE63129 --source-type geo --normalize --growth-rate 0.8

# 指定SBML模型和输出文件前缀
python src/examples/process_transcriptomics_for_pfba.py --input OmicsQc2/raw/test_transcriptomics.csv --model-path models/data/recon3d.xml --output-prefix my_pfba_analysis
```

### 命令行参数

- `--input`：输入转录组数据源（本地文件路径、GEO ID或URL）【必需】
- `--source-type`：数据源类型，可选值：`auto`、`local`、`geo`、`url`，默认为`auto`
- `--normalize`：是否进行数据标准化
- `--filter`：是否过滤低表达基因
- `--threshold`：基因表达阈值，默认为5.0
- `--growth-rate`：指定生长速率，若不指定则从数据估计
- `--model-path`：pFBA模型文件路径（SBML格式），若不指定则使用模拟模式
- `--output-prefix`：输出文件名前缀，默认为`pfba_results`

## 数据流程

1. **数据收集**：
   - 从指定来源收集转录组数据
   - 原始数据保存到`raw`目录

2. **数据预处理**：
   - 对数据进行标准化处理
   - 过滤低表达基因
   - 处理后的数据保存到`processed`目录

3. **pFBA参数格式化**：
   - 将转录组数据转换为基因表达约束
   - 添加培养基成分和基因敲除信息
   - 参数保存到`models/pfba/data`目录

4. **pFBA模型分析**：
   - 加载代谢网络模型
   - 设置生长速率、底物摄取约束
   - 应用基因表达约束
   - 运行pFBA模拟

5. **结果输出**：
   - 模拟结果保存到`outputs`目录
   - 结果包含各反应的通量和目标函数值

## 数据格式说明

### 输入格式

转录组数据应为表格格式（CSV或TSV），其中：
- 行：基因（基因ID或基因名）
- 列：样本
- 值：基因表达量

### 参数格式

格式化后的pFBA参数为JSON格式，包含以下主要内容：

```json
{
  "model_constraints": {
    "expressed_genes": ["GENE001", "GENE002", ...],
    "non_expressed_genes": ["GENE006", ...],
    "growth_rate": 0.8
  },
  "substrate_constraints": {
    "glucose": {"lower_bound": 0.0, "upper_bound": 10.0},
    "glutamine": {"lower_bound": 0.0, "upper_bound": 8.0},
    "oxygen": {"lower_bound": 0.0, "upper_bound": 1.05}
  },
  "knockout_genes": ["GENE006", "GENE011"],
  "simulation_parameters": {
    "objective_function": "biomass_reaction",
    "minimize_total_flux": true
  }
}
```

### 输出格式

pFBA分析结果保存为JSON和CSV格式：

- JSON文件包含完整结果，包括通量、目标函数值和求解状态
- CSV文件列出所有反应的通量值，便于后续分析

## 示例分析流程

以下是一个完整的分析流程示例：

```bash
# 使用测试数据运行pFBA分析
python src/examples/process_transcriptomics_for_pfba.py --input OmicsQc2/raw/test_transcriptomics.csv --normalize --filter --threshold 5.0 --output-prefix example_analysis
```

此命令会：
1. 从测试数据文件收集转录组数据
2. 对数据进行标准化
3. 过滤表达量低于5.0的基因
4. 将数据格式化为pFBA参数
5. 运行pFBA模拟（模拟模式）
6. 输出结果到`outputs/example_analysis_results.json`

## 注意事项

- 真实的pFBA分析需要安装COBRApy库：`pip install cobra`
- 若未指定模型文件，程序将使用模拟模式，生成随机通量作为结果
- 培养基成分和基因敲除可以根据需要在脚本中修改
- 大型代谢网络模型的求解可能需要较长时间 