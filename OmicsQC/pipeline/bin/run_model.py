import cobra
import pandas as pd
from pathlib import Path
import os
import logging


def run_flux_analysis():
    # 动态路径配置
    base_dir = Path(__file__).parent.parent
    model_path = base_dir / "Recon3D.xml"
    data_path = base_dir / "data/processed/condition_data.parquet"
    output_path = base_dir / "results/flux_results.xlsx"

    # 日志配置
    logging.basicConfig(filename=base_dir / 'logs/model.log', level=logging.INFO)

    try:
        # 加载预处理数据
        condition_data = pd.read_parquet(data_path)

        # 初始化代谢模型
        model = cobra.io.read_sbml_model(str(model_path))

        # 执行通量平衡分析
        with model:
            solution = model.optimize()
            flux_df = solution.fluxes.to_frame().T

        # 结果保存（兼容Excel格式）
        flux_df.to_excel(output_path, sheet_name='pFBA_Results')
        logging.info(f"通量分析成功完成 -> {output_path}")

    except Exception as e:
        logging.error(f"模型运行失败：{str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    run_flux_analysis()
