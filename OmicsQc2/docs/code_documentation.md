# OmicsQc2 代码文档

## 1. 源代码结构

### 1.1 主程序
#### main.py
- **功能**：项目主入口，提供命令行接口
- **主要组件**：
  - 参数解析器
  - 工作流程控制
  - 日志配置
- **使用方式**：
  ```bash
  python main.py --mode [pipeline|preprocess|qc|model] --config config/config.yaml
  ```

### 1.2 预处理模块 (preprocessing/)
#### process_transcriptomics_example.py
- **功能**：转录组数据预处理
- **主要功能**：
  - 数据读取和格式检查
  - 数据标准化
  - 缺失值填充
  - 结果输出
- **使用示例**：
  ```bash
  python process_transcriptomics_example.py --input test --source-type geo --normalize --impute
  ```

#### format_pfba_params.py
- **功能**：生成pFBA模型参数
- **主要功能**：
  - 计算基因表达水平
  - 生成模型约束条件
  - 输出JSON格式参数
- **使用示例**：
  ```bash
  python format_pfba_params.py --input processed_data.tsv --output pfba_params.json --threshold 0.5
  ```

### 1.3 模型模块 (models/)
#### pfba/pfba_model.py
- **功能**：pFBA模型实现
- **主要组件**：
  - PFBAModel类
  - 模型加载和验证
  - 约束条件设置
  - 通量分析计算
- **核心方法**：
  ```python
  def __init__(self, model_path=None)
  def load_model(self)
  def set_constraints(self, constraints_dict)
  def run_simulation(self)
  def export_results(self, output_path)
  ```

### 1.4 质量控制模块 (qc/)
#### qc_metrics.py
- **功能**：数据质量评估
- **主要指标**：
  - 样本相关性分析
  - 表达分布检查
  - 异常值检测
- **输出**：质量控制报告

#### qc_visualization.py
- **功能**：质量控制可视化
- **图表类型**：
  - 相关性热图
  - PCA分析图
  - 箱线图
  - 密度分布图

### 1.5 数据整合模块 (integration/)
#### data_integration.py
- **功能**：多组学数据整合
- **整合方法**：
  - 数据标准化
  - 特征选择
  - 关联分析

#### pathway_analysis.py
- **功能**：通路富集分析
- **分析方法**：
  - GO富集分析
  - KEGG通路分析
  - 网络构建

### 1.6 工具函数 (utils/)
#### data_utils.py
- **功能**：通用数据处理函数
- **主要函数**：
  - 数据读写
  - 格式转换
  - 数据验证

#### visualization_utils.py
- **功能**：通用可视化函数
- **图表类型**：
  - 散点图
  - 热图
  - 条形图
  - 网络图

## 2. 代码示例

### 2.1 数据预处理示例
```python
from preprocessing.process_transcriptomics_example import process_data

# 处理转录组数据
result = process_data(
    input_file="raw/test_data.csv",
    normalize=True,
    impute=True
)
```

### 2.2 pFBA模型使用示例
```python
from models.pfba.pfba_model import PFBAModel

# 初始化模型
model = PFBAModel("models/pfba/data/model.xml")

# 设置约束条件
model.set_constraints({
    "model_constraints": {
        "expressed_genes": ["GENE1", "GENE2"],
        "growth_rate": 0.5
    }
})

# 运行模拟
results = model.run_simulation()
```

### 2.3 质量控制示例
```python
from qc.qc_metrics import QualityControl

# 创建质控对象
qc = QualityControl(data_matrix)

# 运行质控
qc_report = qc.run_qc_analysis()
qc.generate_report("outputs/qc_report.html")
```

## 3. 开发规范

### 3.1 代码风格
- 遵循PEP 8规范
- 使用类型注解
- 编写详细的文档字符串

### 3.2 测试规范
- 单元测试覆盖率 > 80%
- 集成测试确保功能完整性
- 使用pytest进行测试

### 3.3 版本控制
- 使用语义化版本号
- 保持更新日志完整性
- 遵循Git分支管理规范

## 4. 性能优化

### 4.1 数据处理优化
- 使用pandas高效操作
- 实现数据批处理
- 优化内存使用

### 4.2 计算优化
- 使用numpy向量化运算
- 实现并行计算
- 优化算法复杂度

## 5. 错误处理

### 5.1 异常类型
- DataError：数据相关错误
- ModelError：模型相关错误
- ConfigError：配置相关错误

### 5.2 日志记录
- 使用logging模块
- 分级别记录日志
- 保存详细错误信息

## 6. 未来开发计划

### 6.1 功能增强
- 添加更多数据源支持
- 扩展分析模型类型
- 优化可视化效果

### 6.2 性能提升
- 实现分布式计算
- 优化内存管理
- 提高处理速度

### 6.3 用户体验
- 添加Web界面
- 改进错误提示
- 简化配置过程 