# pfba_data_analyzer.py
# pFBA输入输出数据分析工具

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cobra.io import read_sbml_model
from cobra.flux_analysis import pfba
import torch
from sklearn.preprocessing import StandardScaler
import json
import argparse

# 导入项目配置
sys.path.append('.')
from config import CONFIG

class PFBAAnalyzer:
    """pFBA数据分析器，用于提取和分析pFBA的输入和输出数据"""
    
    def __init__(self, config=None):
        """
        初始化pFBA分析器
        Args:
            config: 配置字典，若不提供则使用默认CONFIG
        """
        self.config = config or CONFIG
        self.scaler = StandardScaler()
        
        # 输出目录
        self.output_dir = 'pfba_analysis_results'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 加载代谢模型
        try:
            print(f"正在加载代谢模型: {self.config['model_path']}")
            self.model = read_sbml_model(self.config['model_path'])
            print(f"成功加载模型，包含 {len(self.model.reactions)} 个反应和 {len(self.model.metabolites)} 个代谢物")
        except Exception as e:
            print(f"无法加载代谢模型: {str(e)}")
            self.model = None
            
    def load_expression_data(self, data_path=None):
        """
        加载基因表达数据
        Args:
            data_path: 数据文件路径，若不提供则使用配置中的路径
        Returns:
            pandas.DataFrame: 加载的数据
        """
        data_path = data_path or self.config['data_path']
        print(f"正在加载表达数据: {data_path}")
        
        try:
            data = pd.read_csv(data_path)
            print(f"成功加载数据，形状: {data.shape}")
            return data
        except Exception as e:
            print(f"加载数据时出错: {str(e)}")
            return None
            
    def analyze_sample(self, sample_idx=0, n_samples=1, save_results=True):
        """
        分析指定样本的pFBA输入输出
        Args:
            sample_idx: 样本索引
            n_samples: 要分析的样本数量
            save_results: 是否保存结果到文件
        """
        if self.model is None:
            print("代谢模型未加载，无法执行分析")
            return
            
        # 加载表达数据
        data = self.load_expression_data()
        if data is None:
            return
            
        # 标准化数据
        scaled_data = self.scaler.fit_transform(data.values)
        
        # 分析指定数量的样本
        end_idx = min(sample_idx + n_samples, len(data))
        
        for idx in range(sample_idx, end_idx):
            print(f"\n正在分析样本 {idx}...")
            self._analyze_single_sample(scaled_data[idx], idx, save_results)
            
    def _analyze_single_sample(self, sample_data, sample_idx, save_results):
        """
        分析单个样本的pFBA输入输出
        Args:
            sample_data: 样本数据
            sample_idx: 样本索引
            save_results: 是否保存结果到文件
        """
        # 准备数据结构记录分析结果
        pfba_input = {}
        
        # 记录pFBA输入 - 反应约束
        with self.model as model:
            # 设置反应边界
            print("设置反应边界条件...")
            n_reactions = len(model.reactions)
            
            if len(sample_data) < n_reactions:
                print(f"警告：输入数据维度({len(sample_data)})小于模型反应数量({n_reactions})")
                for i, rxn in enumerate(model.reactions):
                    if i < len(sample_data):
                        bound_value = sample_data[i]
                        if bound_value < 0:
                            rxn.bounds = (bound_value * 1.1, bound_value * 0.9)
                        else:
                            rxn.bounds = (bound_value * 0.9, bound_value * 1.1)
                        pfba_input[rxn.id] = {'bound': bound_value, 'original_bounds': rxn.bounds}
                    else:
                        rxn.bounds = (-1000, 1000)
                        pfba_input[rxn.id] = {'bound': None, 'original_bounds': (-1000, 1000)}
            else:
                for i, rxn in enumerate(model.reactions):
                    bound_value = sample_data[i]
                    if bound_value < 0:
                        rxn.bounds = (bound_value * 1.1, bound_value * 0.9)
                    else:
                        rxn.bounds = (bound_value * 0.9, bound_value * 1.1)
                    pfba_input[rxn.id] = {'bound': bound_value, 'original_bounds': rxn.bounds}
            
            # 设置特定反应的约束
            print("设置特定代谢物约束...")
            if 'EX_glc_D_e' in model.reactions:
                model.reactions.EX_glc_D_e.bounds = (-10, 0)
                pfba_input['EX_glc_D_e'] = {'special_constraint': True, 'bounds': (-10, 0)}
                
            if 'EX_o2_e' in model.reactions:
                model.reactions.EX_o2_e.bounds = (-20, 0)
                pfba_input['EX_o2_e'] = {'special_constraint': True, 'bounds': (-20, 0)}
            
            # 执行PFBA并记录结果
            print("执行pFBA分析...")
            try:
                pfba_result = pfba(model)
                print(f"pFBA求解成功，目标函数值: {pfba_result.objective_value}")
                
                # 获取通量结果
                fluxes = {rxn.id: pfba_result.fluxes[rxn.id] for rxn in model.reactions}
                
                # 获取输入约束和输出通量的对比
                input_vs_output = {}
                for rxn_id, flux in fluxes.items():
                    rxn_input = pfba_input.get(rxn_id, {})
                    input_vs_output[rxn_id] = {
                        'input_bound': rxn_input.get('bound', None),
                        'output_flux': flux
                    }
                
                # 保存结果
                if save_results:
                    self._save_results(sample_idx, pfba_input, fluxes, input_vs_output, pfba_result.objective_value)
                
                # 可视化部分结果
                self._visualize_results(sample_idx, input_vs_output, 20)  # 只显示前20个反应
                
            except Exception as e:
                print(f"pFBA求解失败: {str(e)}")
    
    def _save_results(self, sample_idx, pfba_input, pfba_output, input_vs_output, objective_value):
        """保存分析结果到文件"""
        # 创建结果目录
        sample_dir = os.path.join(self.output_dir, f'sample_{sample_idx}')
        os.makedirs(sample_dir, exist_ok=True)
        
        # 保存pFBA输入
        with open(os.path.join(sample_dir, 'pfba_input.json'), 'w') as f:
            json.dump(pfba_input, f, indent=2)
            
        # 保存pFBA输出
        with open(os.path.join(sample_dir, 'pfba_output.json'), 'w') as f:
            output_data = {
                'fluxes': pfba_output,
                'objective_value': objective_value
            }
            json.dump(output_data, f, indent=2)
            
        # 保存输入输出对比
        pd.DataFrame(input_vs_output).T.to_csv(os.path.join(sample_dir, 'input_vs_output.csv'))
        
        print(f"分析结果已保存到目录: {sample_dir}")
    
    def _visualize_results(self, sample_idx, input_vs_output, max_reactions=20):
        """可视化pFBA输入和输出的对比"""
        # 提取数据
        df = pd.DataFrame(input_vs_output).T
        df = df.head(max_reactions)  # 只显示前n个反应
        
        # 创建图表
        plt.figure(figsize=(12, 8))
        
        # 绘制输入边界和输出通量
        x = np.arange(len(df))
        width = 0.35
        
        plt.bar(x - width/2, df['input_bound'], width, label='输入边界值')
        plt.bar(x + width/2, df['output_flux'], width, label='输出通量值')
        
        plt.xlabel('反应ID')
        plt.ylabel('值')
        plt.title(f'样本 {sample_idx} 的pFBA输入与输出对比')
        plt.xticks(x, df.index, rotation=90)
        plt.legend()
        plt.tight_layout()
        
        # 保存图表
        output_file = os.path.join(self.output_dir, f'sample_{sample_idx}', 'visualization.png')
        plt.savefig(output_file)
        print(f"可视化结果已保存到: {output_file}")
        
        plt.close()
        
    def analyze_model(self):
        """分析代谢模型的基本特性"""
        if self.model is None:
            print("代谢模型未加载，无法执行分析")
            return
            
        # 提取模型信息
        model_info = {
            'reactions_count': len(self.model.reactions),
            'metabolites_count': len(self.model.metabolites),
            'genes_count': len(self.model.genes),
            'compartments': list(self.model.compartments),
            'objective_reaction': str(self.model.objective.expression),
            'boundary_reactions': [r.id for r in self.model.reactions if r.boundary],
            'exchange_reactions': [r.id for r in self.model.exchanges]
        }
        
        # 保存模型信息
        with open(os.path.join(self.output_dir, 'model_info.json'), 'w') as f:
            json.dump(model_info, f, indent=2)
            
        print(f"模型信息已保存到: {os.path.join(self.output_dir, 'model_info.json')}")
        
        # 输出摘要
        print("\n代谢模型摘要:")
        print(f"反应数量: {model_info['reactions_count']}")
        print(f"代谢物数量: {model_info['metabolites_count']}")
        print(f"基因数量: {model_info['genes_count']}")
        print(f"室室: {', '.join(model_info['compartments'])}")
        print(f"目标函数: {model_info['objective_reaction']}")
        print(f"交换反应数量: {len(model_info['exchange_reactions'])}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='pFBA输入输出数据分析工具')
    parser.add_argument('--sample', type=int, default=0, help='要分析的样本索引')
    parser.add_argument('--n_samples', type=int, default=1, help='要分析的样本数量')
    parser.add_argument('--model', action='store_true', help='是否分析模型基本特性')
    args = parser.parse_args()
    
    analyzer = PFBAAnalyzer()
    
    if args.model:
        analyzer.analyze_model()
        
    analyzer.analyze_sample(args.sample, args.n_samples)
    
    print("\n分析完成!")

if __name__ == "__main__":
    main()