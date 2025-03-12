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
        
    def collect_transcriptomics(self, source_path: Union[str, Path], 
                               output_file: Optional[str] = None,
                               source_type: str = "auto",
                               sample_info: Optional[Dict] = None) -> pd.DataFrame:
        """
        收集转录组数据
        
        参数:
            source_path: 数据源路径，可以是本地文件路径、GEO ID或其他远程URL
            output_file: 输出文件路径，如果为None则不保存到文件
            source_type: 数据源类型，可选值：'auto', 'local', 'geo', 'url'
            sample_info: 样本信息字典，用于标注样本属性
            
        返回:
            转录组数据DataFrame，行为基因，列为样本
        """
        # 自动检测数据源类型
        if source_type == "auto":
            source_type = self._detect_source_type(source_path)
            
        logger.info(f"开始收集转录组数据，来源: {source_path}, 类型: {source_type}")
        
        # 根据数据源类型调用相应的数据收集方法
        if source_type == "local":
            data = self._collect_from_local(source_path)
        elif source_type == "geo":
            data = self._collect_from_geo(source_path)
        elif source_type == "url":
            data = self._collect_from_url(source_path)
        else:
            raise ValueError(f"不支持的数据源类型: {source_type}")
        
        # 数据初步清洗和格式化
        data = self._clean_transcriptomics_data(data)
        
        # 保存数据到文件（如果指定了输出路径）
        if output_file:
            output_path = os.path.join(self.data_dir, "transcriptomics", output_file)
            data.to_csv(output_path, sep='\t', index=True)
            logger.info(f"转录组数据已保存至: {output_path}")
        
        return data
    
    def _detect_source_type(self, source_path: str) -> str:
        """检测数据源类型"""
        # 检查是否为本地文件
        if os.path.exists(source_path):
            return "local"
        
        # 检查是否为GEO ID (格式如GSE12345)
        if re.match(r"^GSE\d+$", source_path):
            return "geo"
        
        # 检查是否为URL
        if source_path.startswith(("http://", "https://", "ftp://")):
            return "url"
        
        # 默认为本地文件（可能不存在）
        return "local"
    
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
        logger.info(f"从GEO数据库下载数据: {geo_id}")
        
        # 构建GEO FTP URL
        base_url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{geo_id[:-3]}nnn/{geo_id}/matrix/"
        
        # 获取文件列表
        try:
            response = requests.get(base_url)
            response.raise_for_status()
            
            # 使用正则表达式查找矩阵文件
            matrix_files = re.findall(r'href="([^"]+_series_matrix\.txt\.gz)"', response.text)
            
            if not matrix_files:
                raise ValueError(f"找不到 {geo_id} 的矩阵文件")
            
            # 下载第一个矩阵文件
            matrix_url = base_url + matrix_files[0]
            matrix_response = requests.get(matrix_url)
            matrix_response.raise_for_status()
            
            # 将gzip文件写入临时文件
            temp_file = os.path.join(self.cache_dir, f"{geo_id}_matrix.txt.gz")
            with open(temp_file, 'wb') as f:
                f.write(matrix_response.content)
            
            # 使用pandas读取gzip文件
            # 注意：GEO文件格式较为特殊，需要跳过头部元数据
            # 寻找表达数据开始的位置
            skip_rows = 0
            with gzip.open(temp_file, 'rt') as f:
                for line in f:
                    skip_rows += 1
                    if line.startswith('!series_matrix_table_begin'):
                        break
                        
            # 读取数据
            df = pd.read_csv(temp_file, sep='\t', skiprows=skip_rows+1, 
                             compression='gzip', index_col=0)
            
            # 清理数据
            # 删除元数据结束标记行
            df = df[~df.index.str.startswith('!')]
            
            # 转置数据（使基因为行，样本为列）
            if len(df.columns) < len(df):  # 假设样本数通常少于基因数
                df = df.transpose()
                
            return df
            
        except Exception as e:
            logger.error(f"从GEO下载数据失败: {str(e)}")
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
        # 复制数据，避免修改原始数据
        df = data.copy()
        
        # 检查并确保基因ID/名称列存在
        gene_id_col = self._find_gene_id_column(df)
        if gene_id_col:
            # 设置基因ID为索引（如果尚未设置）
            if gene_id_col in df.columns:
                df = df.set_index(gene_id_col)
        
        # 删除空行和空列
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # 确保所有数值列为浮点型
        for col in df.select_dtypes(include=['number']).columns:
            df[col] = df[col].astype(float)
        
        # 处理可能的重复行（基因）
        if df.index.duplicated().any():
            logger.warning(f"发现{df.index.duplicated().sum()}个重复基因ID")
            # 保留第一个出现的基因
            df = df[~df.index.duplicated(keep='first')]
        
        return df
    
    def _find_gene_id_column(self, df: pd.DataFrame) -> Optional[str]:
        """查找基因ID列"""
        # 常见的基因ID列名
        gene_id_patterns = [
            r'gene[_\s]?id', r'ensembl[_\s]?id', r'entrez[_\s]?id',
            r'gene[_\s]?symbol', r'gene[_\s]?name'
        ]
        
        # 检查列名
        for pattern in gene_id_patterns:
            matching_cols = [col for col in df.columns if re.search(pattern, col, re.IGNORECASE)]
            if matching_cols:
                return matching_cols[0]
        
        # 没找到明确的基因ID列，尝试找类似ID的列
        if 'ID' in df.columns:
            return 'ID'
        
        id_cols = [col for col in df.columns if 'id' in col.lower()]
        if id_cols:
            return id_cols[0]
            
        # 如果仍然没找到，可能第一列是未命名的基因ID列
        if df.columns[0] == 'Unnamed: 0':
            return df.columns[0]
            
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