import numpy as np
import pandas as pd
from cobra.io import read_sbml_model

def create_test_data():
    """
    创建测试数据集
    生成随机的代谢数据，用于模型训练和测试
    
    Returns:
        None: 数据保存到CSV文件
    """
    # 读取代谢模型
    print("读取代谢模型...")
    model = read_sbml_model("Recon3D.xml")
    n_reactions = len(model.reactions)
    
    # 生成随机数据
    print(f"生成随机数据，反应数量: {n_reactions}")
    n_samples = 100  # 样本数量
    
    # 初始化数据矩阵
    data = np.random.uniform(-1, 1, (n_samples, n_reactions))
    
    # 为特定反应设置特殊范围
    reaction_ids = [rxn.id for rxn in model.reactions]
    
    # 如果存在葡萄糖交换反应，设置特定范围
    if 'EX_glc_D_e' in reaction_ids:
        glc_idx = reaction_ids.index('EX_glc_D_e')
        data[:, glc_idx] = np.random.uniform(-10, 0, n_samples)  # 葡萄糖摄入范围
    
    # 如果存在氧气交换反应，设置特定范围
    if 'EX_o2_e' in reaction_ids:
        o2_idx = reaction_ids.index('EX_o2_e')
        data[:, o2_idx] = np.random.uniform(-20, 0, n_samples)  # 氧气摄入范围
    
    # 创建DataFrame并保存到CSV
    df = pd.DataFrame(data, columns=reaction_ids)
    output_file = "bulk_data.csv"
    df.to_csv(output_file, index=False)
    print(f"测试数据已保存到 {output_file}，共 {n_samples} 个样本，{n_reactions} 个反应")

if __name__ == "__main__":
    create_test_data()