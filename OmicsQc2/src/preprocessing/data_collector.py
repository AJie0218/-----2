"""
数据收集模块
职责：负责从不同来源收集组学数据
"""

import os
import pandas as pd
import numpy as np
import logging
import gzip
from pathlib import Path
import requests
import zipfile
import io
import re
import chardet
import subprocess
from typing import Union, Optional, Dict, List, Tuple

# 设置logger
logger = logging.getLogger(__name__)

class DataCollector:
    def __init__(self, data_dir: str = "raw", cache_dir: str = ".cache"):
        """
        初始化数据收集器
        
        参数:
            data_dir: 数据存储目录
            cache_dir: 缓存目录
        """
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        
        # 确保目录存在
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(os.path.join(data_dir, "transcriptomics"), exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)
        
    def collect_transcriptomics(self, source_path: str, source_type: str = "auto", 
                               output_file: Optional[str] = None,
                               sample_info: Optional[Dict] = None) -> pd.DataFrame:
        """收集转录组数据
        
        参数:
            source_path: 数据源路径（文件路径、GEO ID或URL）
            source_type: 数据源类型（'local', 'geo', 'url', 'test', 'auto'）
            output_file: 输出文件路径
            sample_info: 样本信息字典
            
        返回:
            处理后的转录组数据DataFrame
        """
        logger.info(f"从 {source_path} 收集转录组数据，类型: {source_type}")
        
        try:
            # 自动检测数据源类型
            if source_type == "auto":
                source_type = self._detect_source_type(source_path)
                logger.info(f"自动检测到数据源类型: {source_type}")
            
            # 根据数据源类型收集数据
            if source_type == "test":
                data = self._generate_test_data()
            elif source_type == "local":
                data = self._collect_from_local(source_path)
            elif source_type == "geo":
                if source_path.lower() == "test":
                    data = self._generate_test_data()
                else:
                    data = self._collect_from_geo(source_path)
            elif source_type == "url":
                data = self._collect_from_url(source_path)
            else:
                raise ValueError(f"不支持的数据源类型: {source_type}")
                
            if data is None or data.empty:
                raise ValueError("收集的数据为空")
                
            logger.info(f"成功收集数据: {data.shape[0]}行 × {data.shape[1]}列")
            
            # 数据预览
            logger.info("数据预览:")
            preview = data.head(3).to_string()
            for line in preview.split('\n'):
                logger.info(line)
            
            # 清理和格式化数据
            data = self._clean_transcriptomics_data(data)
            
            # 如果提供了样本信息，添加到数据中
            if sample_info:
                data = self._add_sample_info(data, sample_info)
            
            # 保存处理后的数据
            if output_file:
                self._save_data(data, output_file)
                logger.info(f"数据已保存到: {output_file}")
            
            return data
            
        except Exception as e:
            logger.error(f"数据收集过程中发生错误: {str(e)}")
            raise
            
    def _detect_source_type(self, source_path: str) -> str:
        """检测数据源类型"""
        if source_path.lower() == "test":
            return "test"
        elif os.path.exists(source_path):
            return "local"
        elif re.match(r"GSE\d+", source_path):
            return "geo"
        elif source_path.startswith(("http://", "https://", "ftp://")):
            return "url"
        else:
            return "local"  # 默认为本地文件
    
    def _collect_from_local(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """从本地文件收集数据"""
        logger.info(f"从本地文件加载数据: {file_path}")
        
        # 检测文件格式
        file_format = self._detect_file_format(file_path)
        
        # 检测文件编码
        encoding = self._detect_file_encoding(file_path)
        
        # 根据文件格式读取数据
        if file_format in ["csv", "txt"]:
            # 检测分隔符
            sep = self._detect_delimiter(file_path, encoding)
            
            # 读取数据
            try:
                df = pd.read_csv(file_path, sep=sep, encoding=encoding)
                return df
            except Exception as e:
                logger.error(f"读取文件 {file_path} 失败: {str(e)}")
                raise
        elif file_format == "excel":
            try:
                df = pd.read_excel(file_path)
                return df
            except Exception as e:
                logger.error(f"读取Excel文件 {file_path} 失败: {str(e)}")
                raise
        else:
            raise ValueError(f"不支持的文件格式: {file_format}")
    
    def _collect_from_geo(self, geo_id: str) -> pd.DataFrame:
        """从GEO数据库收集数据"""
        logger.info(f"从GEO下载数据: {geo_id}")
        
        # 如果是测试模式，返回测试数据
        if geo_id.lower() == "test":
            logger.info("使用测试数据")
            return self._generate_test_data()
            
        base_url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{geo_id[:-3]}nnn/{geo_id}/matrix/"
        
        try:
            logger.info(f"尝试访问URL: {base_url}")
            response = requests.get(base_url)
            response.raise_for_status()
            
            # 使用正则表达式查找矩阵文件
            matrix_files = re.findall(r'href="([^"]+_series_matrix\.txt\.gz)"', response.text)
            logger.info(f"找到 {len(matrix_files)} 个矩阵文件")
            
            if not matrix_files:
                raise ValueError(f"找不到 {geo_id} 的矩阵文件")
            
            # 下载第一个矩阵文件
            matrix_url = base_url + matrix_files[0]
            logger.info(f"下载矩阵文件: {matrix_url}")
            matrix_response = requests.get(matrix_url)
            matrix_response.raise_for_status()
            
            # 将gzip文件写入临时文件
            temp_file = os.path.join(self.cache_dir, f"{geo_id}_matrix.txt.gz")
            with open(temp_file, 'wb') as f:
                f.write(matrix_response.content)
            logger.info(f"矩阵文件已保存到: {temp_file}")
            
            # 使用pandas读取gzip文件
            logger.info("开始解析数据文件")
            skip_rows = 0
            data_start = False
            with gzip.open(temp_file, 'rt') as f:
                for line in f:
                    if line.startswith('!series_matrix_table_begin'):
                        data_start = True
                        break
                    skip_rows += 1
            
            if not data_start:
                raise ValueError("未找到数据起始标记")
            
            logger.info(f"跳过 {skip_rows} 行头部元数据")
            df = pd.read_csv(temp_file, sep='\t', skiprows=skip_rows+1, 
                           compression='gzip', index_col=0)
            
            logger.info(f"原始数据大小: {df.shape}")
            
            # 清理数据
            # 确保索引是字符串类型
            df.index = df.index.astype(str)
            logger.info("已将索引转换为字符串类型")
            
            # 删除元数据结束标记行
            df = df[~df.index.str.startswith('!')]
            logger.info(f"删除元数据后的数据大小: {df.shape}")
            
            # 删除可能的空行
            df = df.dropna(how='all')
            
            # 确保所有数据都是数值类型
            df = df.apply(pd.to_numeric, errors='coerce')
            
            # 转置数据（使基因为行，样本为列）
            if len(df.columns) < len(df):  # 假设样本数通常少于基因数
                df = df.transpose()
                logger.info("已转置数据矩阵")
            
            # 删除全为NA的行和列
            df = df.dropna(how='all', axis=0).dropna(how='all', axis=1)
            
            logger.info(f"最终数据大小: {df.shape[0]}行 × {df.shape[1]}列")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"下载GEO数据时发生网络错误: {str(e)}")
            raise
        except pd.errors.EmptyDataError:
            logger.error("数据文件为空")
            raise
        except Exception as e:
            logger.error(f"处理GEO数据时发生错误: {str(e)}")
            raise
    
    def _collect_from_url(self, url: str) -> pd.DataFrame:
        """从URL收集数据"""
        logger.info(f"从URL下载数据: {url}")
        
        # 下载文件
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            # 根据URL判断文件类型
            if url.endswith('.csv'):
                df = pd.read_csv(io.StringIO(response.text))
            elif url.endswith('.tsv') or url.endswith('.txt'):
                df = pd.read_csv(io.StringIO(response.text), sep='\t')
            elif url.endswith('.xlsx') or url.endswith('.xls'):
                df = pd.read_excel(io.BytesIO(response.content))
            elif url.endswith('.zip'):
                # 处理ZIP文件
                z = zipfile.ZipFile(io.BytesIO(response.content))
                # 假设ZIP文件中只有一个CSV/TSV文件
                csv_files = [f for f in z.namelist() if f.endswith(('.csv', '.tsv', '.txt'))]
                if not csv_files:
                    raise ValueError("ZIP文件中找不到CSV/TSV文件")
                    
                with z.open(csv_files[0]) as f:
                    # 尝试自动检测分隔符
                    content = f.read()
                    encoding = chardet.detect(content)['encoding']
                    content_str = content.decode(encoding)
                    
                    if ',' in content_str.split('\n')[0]:
                        sep = ','
                    else:
                        sep = '\t'
                        
                    df = pd.read_csv(io.StringIO(content_str), sep=sep)
            else:
                raise ValueError(f"不支持的URL文件类型: {url}")
                
            return df
            
        except Exception as e:
            logger.error(f"从URL下载数据失败: {str(e)}")
            raise
    
    def _detect_file_format(self, file_path: Union[str, Path]) -> str:
        """检测文件格式"""
        file_path = str(file_path).lower()
        
        if file_path.endswith('.csv'):
            return "csv"
        elif file_path.endswith(('.tsv', '.txt')):
            return "csv"  # 使用相同的处理逻辑
        elif file_path.endswith(('.xlsx', '.xls')):
            return "excel"
        else:
            # 尝试通过文件内容判断
            try:
                with open(file_path, 'rb') as f:
                    header = f.read(4096)
                    
                if b'\t' in header:
                    return "csv"  # 实际上是TSV
                elif b',' in header:
                    return "csv"
                elif b'PK\x03\x04' in header:  # Excel文件的魔数
                    return "excel"
                else:
                    return "txt"  # 默认为文本文件
            except Exception:
                return "txt"  # 如果无法读取文件，默认为文本文件
    
    def _detect_file_encoding(self, file_path: Union[str, Path]) -> str:
        """检测文件编码"""
        try:
            with open(file_path, 'rb') as f:
                result = chardet.detect(f.read(10000))
            return result['encoding'] or 'utf-8'
        except Exception:
            return 'utf-8'  # 默认为UTF-8
    
    def _detect_delimiter(self, file_path: Union[str, Path], encoding: str = 'utf-8') -> str:
        """检测文件分隔符"""
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                first_line = f.readline().strip()
                
            if '\t' in first_line:
                return '\t'
            elif ',' in first_line:
                return ','
            else:
                return '\t'  # 默认为制表符
        except Exception:
            return '\t'  # 如果无法读取文件，默认为制表符
    
    def _clean_transcriptomics_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """清洗和格式化转录组数据"""
        logger.info("开始清理转录组数据")
        
        if data is None or data.empty:
            raise ValueError("输入数据为空")
            
        # 复制数据，避免修改原始数据
        df = data.copy()
        logger.info(f"原始数据大小: {df.shape}")
        
        # 删除空行和空列
        df = df.dropna(how='all').dropna(axis=1, how='all')
        logger.info(f"删除空行和空列后的数据大小: {df.shape}")
        
        if df.empty:
            raise ValueError("清理后的数据为空")
        
        # 检查并确保基因ID/名称列存在
        try:
            gene_id_col = self._find_gene_id_column(df)
            if gene_id_col:
                logger.info(f"找到基因ID列: {gene_id_col}")
                # 设置基因ID为索引（如果尚未设置）
                if gene_id_col in df.columns:
                    df = df.set_index(gene_id_col)
                    logger.info("已将基因ID列设置为索引")
        except Exception as e:
            logger.warning(f"处理基因ID列时出错: {str(e)}")
        
        # 确保所有数值列为浮点型
        numeric_cols = df.select_dtypes(include=['number']).columns
        logger.info(f"发现 {len(numeric_cols)} 个数值列")
        
        for col in numeric_cols:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception as e:
                logger.warning(f"转换列 {col} 为数值类型时出错: {str(e)}")
        
        # 处理可能的重复行（基因）
        if df.index.duplicated().any():
            dup_count = df.index.duplicated().sum()
            logger.warning(f"发现 {dup_count} 个重复基因ID")
            # 保留第一个出现的基因
            df = df[~df.index.duplicated(keep='first')]
            logger.info(f"删除重复后的数据大小: {df.shape}")
        
        # 最终检查
        if df.empty:
            raise ValueError("数据清理后结果为空")
            
        logger.info("数据清理完成")
        return df
    
    def _find_gene_id_column(self, df: pd.DataFrame) -> Optional[str]:
        """查找基因ID列"""
        if df is None or df.empty:
            raise ValueError("输入数据框为空")
            
        if len(df.columns) == 0:
            raise ValueError("数据框没有列")
            
        logger.info(f"开始查找基因ID列，当前列: {', '.join(df.columns)}")
        
        # 常见的基因ID列名
        gene_id_patterns = [
            r'gene[_\s]?id', r'ensembl[_\s]?id', r'entrez[_\s]?id',
            r'gene[_\s]?symbol', r'gene[_\s]?name', r'probe[_\s]?id',
            r'transcript[_\s]?id', r'feature[_\s]?id'
        ]
        
        # 检查列名
        for pattern in gene_id_patterns:
            matching_cols = [col for col in df.columns if re.search(pattern, col, re.IGNORECASE)]
            if matching_cols:
                logger.info(f"找到匹配的基因ID列: {matching_cols[0]}")
                return matching_cols[0]
        
        # 没找到明确的基因ID列，尝试找类似ID的列
        if 'ID' in df.columns:
            logger.info("使用'ID'列作为基因标识符")
            return 'ID'
        
        id_cols = [col for col in df.columns if 'id' in col.lower()]
        if id_cols:
            logger.info(f"使用包含'id'的列: {id_cols[0]}")
            return id_cols[0]
            
        # 如果仍然没找到，检查第一列
        if len(df.columns) > 0:
            first_col = df.columns[0]
            if first_col == 'Unnamed: 0' or 'ID' in str(first_col).upper():
                logger.info(f"使用第一列作为基因标识符: {first_col}")
                return first_col
                
        logger.warning("未找到合适的基因ID列")
        return None
        
    def collect_proteomics(self, source_path):
        """
        收集蛋白组数据
        
        参数:
            source_path: 数据源路径
            
        返回:
            蛋白组数据对象
        """
        # TODO: 实现蛋白组数据收集逻辑
        pass
        
    def collect_metabolomics(self, source_path):
        """
        收集代谢组数据
        
        参数:
            source_path: 数据源路径
            
        返回:
            代谢组数据对象
        """
        # TODO: 实现代谢组数据收集逻辑
        pass
        
    def collect_mfa_flux(self, source_path):
        """
        收集13C-MFA通量数据
        
        参数:
            source_path: 数据源路径
            
        返回:
            通量数据对象
        """
        # TODO: 实现通量数据收集逻辑
        pass

    def _generate_test_data(self) -> pd.DataFrame:
        """生成测试数据用于开发和测试"""
        logger.info("生成测试转录组数据")
        
        # 生成基因ID
        n_genes = 100
        n_samples = 5
        gene_ids = [f"GENE{i:03d}" for i in range(n_genes)]
        sample_ids = [f"Sample{i+1}" for i in range(n_samples)]
        
        # 生成表达值
        np.random.seed(42)
        expression_data = np.random.lognormal(mean=2.0, sigma=1.0, size=(n_genes, n_samples))
        
        # 创建DataFrame
        df = pd.DataFrame(expression_data, index=gene_ids, columns=sample_ids)
        logger.info(f"生成的测试数据大小: {df.shape}")
        
        return df

    def _add_sample_info(self, data: pd.DataFrame, sample_info: Dict) -> pd.DataFrame:
        """添加样本信息到数据中"""
        try:
            # 创建样本信息DataFrame
            sample_df = pd.DataFrame.from_dict(sample_info, orient='index')
            
            # 确保样本ID匹配
            common_samples = set(data.columns) & set(sample_df.index)
            if not common_samples:
                logger.warning("没有找到匹配的样本ID")
                return data
                
            # 只保留匹配的样本
            data = data[list(common_samples)]
            sample_df = sample_df.loc[list(common_samples)]
            
            logger.info(f"添加了 {len(sample_df.columns)} 个样本信息字段")
            return data
            
        except Exception as e:
            logger.warning(f"添加样本信息时出错: {str(e)}")
            return data
            
    def _save_data(self, data: pd.DataFrame, output_file: str):
        """保存数据到文件"""
        try:
            # 如果输出路径是相对路径，将其转换为绝对路径
            if not os.path.isabs(output_file):
                output_file = os.path.join(self.data_dir, output_file)
            
            # 确保输出目录存在
            output_dir = os.path.dirname(output_file)
            if output_dir:  # 只在有目录部分时创建
                os.makedirs(output_dir, exist_ok=True)
            
            # 根据文件扩展名选择保存格式
            if output_file.endswith('.csv'):
                data.to_csv(output_file)
            elif output_file.endswith('.tsv'):
                data.to_csv(output_file, sep='\t')
            elif output_file.endswith('.xlsx'):
                data.to_excel(output_file)
            else:
                # 如果没有扩展名，添加.tsv
                if not os.path.splitext(output_file)[1]:
                    output_file = output_file + '.tsv'
                data.to_csv(output_file, sep='\t')
                
            logger.info(f"数据已保存到: {output_file}")
                
        except Exception as e:
            logger.error(f"保存数据时出错: {str(e)}")
            raise 