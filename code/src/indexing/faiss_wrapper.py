"""
FAISS库的二次封装

本模块提供FAISS库的统一封装接口，简化索引的创建、训练和搜索操作。

论文章节：第4章 4.4节 - FAISS索引封装
"""

import numpy as np
from typing import Tuple, Optional, List, Union
from pathlib import Path
from abc import ABC, abstractmethod

class FaissIndex(ABC):
    """
    FAISS索引基类
    
    定义所有FAISS索引的通用接口。
    
    Args:
        dimension (int): 向量维度
        metric (str): 距离度量类型 ("l2" 或 "ip")
    """
    
    def __init__(self, dimension: int, metric: str = "l2"):
        self.dimension = dimension
        self.metric = metric
        self.index = None
        self.is_trained = False
        self.num_vectors = 0
        
    @abstractmethod
    def build(self, vectors: np.ndarray) -> None:
        """
        构建索引
        
        Args:
            vectors: 向量数组 [num_vectors, dimension]
        """
        pass
    
    @abstractmethod
    def search(
        self,
        queries: np.ndarray,
        k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        搜索最近邻
        
        Args:
            queries: 查询向量 [num_queries, dimension]
            k: 返回的最近邻数量
            
        Returns:
            distances: 距离数组 [num_queries, k]
            indices: 索引数组 [num_queries, k]
        """
        pass
    
    def add(self, vectors: np.ndarray) -> None:
        """
        添加向量到索引
        
        Args:
            vectors: 向量数组 [num_vectors, dimension]
        """
        if self.index is None:
            raise RuntimeError("Index not initialized. Call build() first.")
            
        # TODO: 实现向量添加
        self.num_vectors += len(vectors)
    
    def train(self, vectors: np.ndarray) -> None:
        """
        训练索引（如需要）
        
        Args:
            vectors: 训练向量
        """
        # 基类默认不需要训练
        self.is_trained = True
    
    def save(self, path: str) -> None:
        """
        保存索引到文件
        
        Args:
            path: 保存路径
        """
        import faiss
        
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(save_path))
        
    def load(self, path: str) -> None:
        """
        从文件加载索引
        
        Args:
            path: 索引文件路径
        """
        import faiss
        
        self.index = faiss.read_index(str(path))
        self.num_vectors = self.index.ntotal
        self.is_trained = True
        
    def get_num_vectors(self) -> int:
        """
        获取索引中的向量数量
        """
        return self.num_vectors
    
    def reset(self) -> None:
        """
        重置索引
        """
        if self.index is not None:
            self.index.reset()
        self.num_vectors = 0
        
    @staticmethod
    def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
        """
        L2归一化向量
        
        Args:
            vectors: 输入向量
            
        Returns:
            归一化后的向量
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        return vectors / norms