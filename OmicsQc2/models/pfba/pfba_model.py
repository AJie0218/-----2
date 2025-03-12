"""
pFBA模型模块
职责：封装pFBA (parsimonious Flux Balance Analysis) 相关功能
"""

import os
import json
import pandas as pd
import numpy as np
import logging
from pathlib import Path

# 如果安装了COBRApy，可以使用它来进行真实的pFBA分析
try:
    import cobra
    from cobra.flux_analysis import pfba
    HAS_COBRA = True
except ImportError:
    HAS_COBRA = False
    
# 设置logger
logger = logging.getLogger(__name__)

class PFBAModel:
    def __init__(self, model_path=None):
        """
        初始化pFBA模型
        
        参数:
            model_path: 模型文件路径（SBML格式）
        """
        self.model_path = model_path
        self.model = None
        self.simulation_results = None
        self.constraints = {}
        
    def load_model(self):
        """
        加载模型
        
        返回:
            加载状态
        """
        if not HAS_COBRA:
            logger.warning("未安装COBRApy库，将使用模拟模式")
            self.model = "SIMULATION_MODE"
            return True
            
        try:
            if self.model_path is None:
                logger.error("未指定模型文件路径")
                return False
                
            if not os.path.exists(self.model_path):
                logger.error(f"模型文件不存在: {self.model_path}")
                return False
                
            # 使用COBRApy加载SBML模型
            self.model = cobra.io.read_sbml_model(self.model_path)
            logger.info(f"成功加载模型: {self.model.id}, 包含 {len(self.model.reactions)} 个反应和 {len(self.model.metabolites)} 个代谢物")
            return True
            
        except Exception as e:
            logger.error(f"加载模型时出错: {str(e)}")
            return False
        
    def set_constraints(self, constraints_dict=None, growth_rate=None, media_composition=None):
        """
        设置模型约束条件
        
        参数:
            constraints_dict: 约束条件字典
            growth_rate: 生长速率
            media_composition: 培养基成分
            
        返回:
            是否成功设置约束
        """
        # 保存约束条件
        self.constraints = constraints_dict or {}
        
        # 如果提供了单独的生长速率和培养基成分，添加到约束条件中
        if growth_rate is not None:
            self.constraints.setdefault("model_constraints", {})["growth_rate"] = growth_rate
            
        if media_composition is not None:
            # 将培养基成分转换为底物约束
            substrate_constraints = {}
            for metabolite, concentration in media_composition.items():
                flux_upper_bound = min(10.0, concentration * 5.0)
                substrate_constraints[metabolite] = {
                    "lower_bound": 0.0,
                    "upper_bound": flux_upper_bound
                }
            self.constraints["substrate_constraints"] = substrate_constraints
            
        # 如果使用COBRApy，应用约束条件到模型
        if HAS_COBRA and isinstance(self.model, cobra.Model):
            try:
                # 应用生长速率约束（如果有）
                if "model_constraints" in self.constraints and "growth_rate" in self.constraints["model_constraints"]:
                    growth_rate = self.constraints["model_constraints"]["growth_rate"]
                    # 通常需要找到生物量反应来设置约束
                    biomass_rxn = self.model.reactions.get_by_id("biomass_reaction")  # 实际应用中应使用正确的ID
                    biomass_rxn.lower_bound = growth_rate * 0.9  # 允许10%的误差
                    biomass_rxn.upper_bound = growth_rate * 1.1
                    logger.info(f"已设置生长速率约束: {growth_rate}")
                
                # 应用底物约束（如果有）
                if "substrate_constraints" in self.constraints:
                    substrate_constraints = self.constraints["substrate_constraints"]
                    for metabolite_id, bounds in substrate_constraints.items():
                        # 尝试找到对应的交换反应
                        exchange_rxn = self.model.reactions.get_by_id(f"EX_{metabolite_id}")  # 命名可能不同
                        exchange_rxn.lower_bound = bounds["lower_bound"]
                        exchange_rxn.upper_bound = bounds["upper_bound"]
                    logger.info(f"已设置 {len(substrate_constraints)} 个底物约束")
                
                # 应用基因表达约束（如果有）
                if "model_constraints" in self.constraints and "expressed_genes" in self.constraints["model_constraints"]:
                    # 这里只是简单演示，实际实现可能更复杂
                    expressed_genes = self.constraints["model_constraints"]["expressed_genes"]
                    non_expressed_genes = self.constraints["model_constraints"]["non_expressed_genes"]
                    
                    # 根据未表达的基因禁用相关反应
                    for gene_id in non_expressed_genes:
                        if gene_id in self.model.genes:
                            gene = self.model.genes.get_by_id(gene_id)
                            for reaction in gene.reactions:
                                # 如果反应只依赖于这个基因
                                if len(reaction.genes) == 1:
                                    reaction.lower_bound = 0
                                    reaction.upper_bound = 0
                    
                    logger.info(f"已根据基因表达约束 {len(non_expressed_genes)} 个反应")
                
                return True
                
            except Exception as e:
                logger.error(f"设置约束条件时出错: {str(e)}")
                return False
        else:
            # 模拟模式
            logger.info(f"模拟模式: 已设置约束条件")
            return True
        
    def knockout_genes(self, gene_list):
        """
        模拟基因敲除
        
        参数:
            gene_list: 要敲除的基因列表
            
        返回:
            敲除操作状态
        """
        if not gene_list:
            logger.info("未指定要敲除的基因")
            return True
            
        # 保存敲除基因列表
        self.constraints["knockout_genes"] = gene_list
        
        # 如果使用COBRApy，应用基因敲除
        if HAS_COBRA and isinstance(self.model, cobra.Model):
            try:
                # 对每个基因执行敲除
                for gene_id in gene_list:
                    if gene_id in self.model.genes:
                        gene = self.model.genes.get_by_id(gene_id)
                        gene.knock_out()
                        logger.info(f"已敲除基因: {gene_id}")
                    else:
                        logger.warning(f"找不到基因: {gene_id}")
                
                return True
                
            except Exception as e:
                logger.error(f"基因敲除时出错: {str(e)}")
                return False
        else:
            # 模拟模式
            logger.info(f"模拟模式: 已敲除 {len(gene_list)} 个基因")
            return True
        
    def run_simulation(self):
        """
        运行pFBA模拟
        
        返回:
            模拟结果
        """
        # 如果使用COBRApy，执行真实的pFBA
        if HAS_COBRA and isinstance(self.model, cobra.Model):
            try:
                # 运行pFBA
                pfba_solution = pfba(self.model)
                
                # 提取结果
                fluxes = pd.Series(pfba_solution.fluxes)
                objective_value = pfba_solution.objective_value
                
                # 保存结果
                self.simulation_results = {
                    "fluxes": fluxes.to_dict(),
                    "objective_value": objective_value,
                    "status": "optimal"
                }
                
                logger.info(f"pFBA模拟完成，目标函数值: {objective_value:.4f}")
                return self.simulation_results
                
            except Exception as e:
                logger.error(f"运行pFBA模拟时出错: {str(e)}")
                self.simulation_results = {"status": "error", "message": str(e)}
                return self.simulation_results
        else:
            # 模拟模式：生成随机通量作为模拟结果
            logger.info("使用模拟模式生成随机通量")
            
            # 生成一些随机反应和通量
            num_reactions = 100
            reaction_ids = [f"R{i}" for i in range(num_reactions)]
            fluxes = {}
            
            # 使用正态分布生成随机通量 - 兼容旧版numpy
            np.random.seed(42)  # 使用固定种子确保可复现
            random_fluxes = np.random.normal(loc=0, scale=1.0, size=num_reactions)
            
            # 应用基因敲除的影响（简单模拟）
            if "knockout_genes" in self.constraints:
                knockout_genes = self.constraints["knockout_genes"]
                # 为简单起见，假设基因ID形如"G{i}"，对应反应ID为"R{i}"
                for gene_id in knockout_genes:
                    if gene_id.startswith("G"):
                        rxn_id = "R" + gene_id[1:]
                        if int(gene_id[1:]) < num_reactions:
                            random_fluxes[int(gene_id[1:])] = 0.0
            
            # 应用生长速率约束（简单模拟）
            if "model_constraints" in self.constraints and "growth_rate" in self.constraints["model_constraints"]:
                growth_rate = self.constraints["model_constraints"]["growth_rate"]
                # 假设R0是生物量反应
                random_fluxes[0] = growth_rate
            
            # 创建通量字典
            for i, flux in enumerate(random_fluxes):
                fluxes[reaction_ids[i]] = flux
            
            # 保存结果
            self.simulation_results = {
                "fluxes": fluxes,
                "objective_value": random_fluxes[0],  # 假设R0是目标函数
                "status": "optimal (simulated)"
            }
            
            logger.info(f"模拟完成，生成了 {len(fluxes)} 个随机通量")
            return self.simulation_results
        
    def export_results(self, output_path):
        """
        导出结果
        
        参数:
            output_path: 输出路径
            
        返回:
            导出状态
        """
        if self.simulation_results is None:
            logger.error("没有可导出的模拟结果")
            return False
        
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 导出结果
            with open(output_path, 'w') as f:
                json.dump(self.simulation_results, f, indent=2)
                
            logger.info(f"结果已导出至: {output_path}")
            
            # 如果结果包含通量，也导出为CSV格式
            if "fluxes" in self.simulation_results:
                fluxes = self.simulation_results["fluxes"]
                fluxes_df = pd.DataFrame.from_dict(fluxes, orient='index', columns=['flux'])
                fluxes_df.index.name = 'reaction'
                
                csv_path = output_path.replace('.json', '.csv')
                fluxes_df.to_csv(csv_path)
                logger.info(f"通量已导出至CSV: {csv_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"导出结果时出错: {str(e)}")
            return False
        
    def generate_virtual_data(self, num_samples=10):
        """
        生成虚拟数据
        
        参数:
            num_samples: 样本数量
            
        返回:
            生成的虚拟数据
        """
        logger.info(f"生成 {num_samples} 个虚拟样本")
        
        # 生成虚拟基因
        num_genes = 50
        gene_ids = [f"G{i}" for i in range(num_genes)]
        
        # 生成虚拟样本
        rng = np.random.default_rng(42)  # 使用固定种子确保可复现
        expression_data = rng.normal(loc=5.0, scale=2.0, size=(num_genes, num_samples))
        
        # 确保有一些低表达基因
        expression_data[expression_data < 0] = 0
        
        # 创建DataFrame
        samples = [f"Sample{i+1}" for i in range(num_samples)]
        df = pd.DataFrame(expression_data, index=gene_ids, columns=samples)
        
        logger.info(f"已生成虚拟数据: {df.shape[0]} 个基因, {df.shape[1]} 个样本")
        
        return df 