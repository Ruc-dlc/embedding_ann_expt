"""
IVF索引封装（对比用）

本模块实现IVF（Inverted File Index）索引的封装，作为与HNSW对比的基准方法。

论文章节：第5章 5.2节 - 基准方法对比
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from .faiss_wrapper import FaissIndex

class IVFIndex(FaissIndex):
    """
    IVF索引
    
    基于倒排文件的近似最近邻索引，将向量空间划分为多个聚类。
    
    核心参数：
    - nlist: 聚类中心数量
    - nprobe: 搜索时探测的聚类数量
    
    Args:
        dimension (int): 向量维度
        nlist (int): 聚类数量，默认100
        nprobe (int): 搜索时探测数量，默认10
        metric (str): 距离度量 ("l2" 或 "ip")
        use_pq (bool): 是否使用PQ量化
        pq_m (int): PQ子向量数量
    """
    
    def __init__(
        self,
        dimension: int,
        nlist: int = 100,
        nprobe: int = 10,
        metric: str = "l2",
        use_pq: bool = False,
        pq_m: int = 8
    ):
        super().__init__(dimension, metric)
        
        self.nlist = nlist
        self.nprobe = nprobe
        self.use_pq = use_pq
        self.pq_m = pq_m
        
        self._create_index()
        
    def _create_index(self) -> None:
        """
        创建IVF索引
        """
        import faiss
        
        # 创建量化器
        if self.metric == "ip":
            quantizer = faiss.IndexFlatIP(self.dimension)
        else:
            quantizer = faiss.IndexFlatL2(self.dimension)
            
        # 创建IVF索引
        if self.use_pq:
            if self.metric == "ip":
                self.index = faiss.IndexIVFPQ(
                    quantizer, self.dimension, self.nlist, 
                    self.pq_m, 8, faiss.METRIC_INNER_PRODUCT
                )
            else:
                self.index = faiss.IndexIVFPQ(
                    quantizer, self.dimension, self.nlist, self.pq_m, 8
                )
        else:
            if self.metric == "ip":
                self.index = faiss.IndexIVFFlat(
                    quantizer, self.dimension, self.nlist, faiss.METRIC_INNER_PRODUCT
                )
            else:
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist)
                
        self.is_trained = False
        
    def train(self, vectors: np.ndarray) -> None:
        """
        训练IVF索引
        
        IVF索引需要先训练聚类中心。
        
        Args:
            vectors: 训练向量 [num_vectors, dimension]
        """
        vectors = np.ascontiguousarray(vectors.astype(np.float32))
        
        if self.metric == "ip":
            vectors = self.normalize_vectors(vectors)
            
        self.index.train(vectors)
        self.is_trained = True
        
    def build(self, vectors: np.ndarray) -> None:
        """
        构建索引
        
        Args:
            vectors: 向量数组 [num_vectors, dimension]
        """
        vectors = np.ascontiguousarray(vectors.astype(np.float32))
        
        if self.metric == "ip":
            vectors = self.normalize_vectors(vectors)
            
        # 如果未训练，先训练
        if not self.is_trained:
            self.train(vectors)
            
        self.index.add(vectors)
        self.num_vectors = len(vectors)
        
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
        queries = np.ascontiguousarray(queries.astype(np.float32))
        
        if self.metric == "ip":
            queries = self.normalize_vectors(queries)
            
        # 设置nprobe
        self.index.nprobe = self.nprobe
        
        distances, indices = self.index.search(queries, k)
        
        return distances, indices
    
    def set_nprobe(self, nprobe: int) -> None:
        """
        设置nprobe参数
        
        nprobe越大，搜索精度越高但速度越慢。
        
        Args:
            nprobe: 新的nprobe值
        """
        self.nprobe = nprobe
        
    def get_cluster_stats(self) -> Dict[str, Any]:
        """
        获取聚类统计信息
        
        Returns:
            统计信息字典
        """
        # TODO: 实现聚类分析
        return {
            'num_vectors': self.num_vectors,
            'nlist': self.nlist,
            'nprobe': self.nprobe,
            'dimension': self.dimension,
            'metric': self.metric,
            'use_pq': self.use_pq
        }
    
