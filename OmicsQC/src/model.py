# model.py
from cobra.io import load_model
import pandas as pd


class FluxModel:
    def __init__(self, model_path="D:\\学习软件\\pycharm\\pyCharm-数据\\设计成品\\一键化格式2\\Recon3D.xml"):
        self.model = load_model(model_path)

    def run_pFBA(self, condition_data):
        """通量平衡分析封装"""
        with self.model as model:
            # 设置培养基成分约束
            for met, value in condition_data["media"].items():
                model.reactions.get_by_id(met).bounds = (value, value)

            # 执行pFBA
            solution = model.optimize()
            fluxes = solution.fluxes.to_dict()

            # 生成虚拟数据
            virtual_data = {
                "growth_rate": solution.objective_value,
                "fluxes": fluxes,
                "condition_id": condition_data["id"]
            }
        return virtual_data
