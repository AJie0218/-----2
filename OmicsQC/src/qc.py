import sqlite3
import pandas as pd


class QualityController:
    def check_sample(self, required_fields, sample_id):  # 修正参数顺序
        """检查样本时需传递 required_fields 和 sample_id"""
        conn = sqlite3.connect("omics_qc.db")
        try:
            # 动态构建查询语句
            query = f"SELECT {','.join(required_fields)} FROM omics_raw WHERE sample_id = ?"
            df = pd.read_sql_query(query, conn, params=(sample_id,))

            # 返回检查结果
            return {
                "status": "pass" if not df.empty else "fail",
                "data": df.to_dict(),
                "missing_fields": [col for col in required_fields if col not in df.columns]
            }
        except sqlite3.OperationalError as e:
            return {"error": f"数据库查询失败: {str(e)}"}
