"""
HNSW索引封装

本模块实现HNSW（Hierarchical Navigable Small World）图索引的封装。
HNSW是本论文的核心ANN索引结构，具有优秀的搜索效率和可扩展性。

论文章节：第3章 3.3节 - HNSW算法原理
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from .faiss_wrapper import FaissIndex

class HNSWIndex(FaissIndex):
    """
    HNSW索引
    
    基于图的近似最近邻索引，使用多层跳表结构加速搜索。
    
    核心参数：
    - M: 每个节点的最大连接数，影响图的稀疏程度
    - ef_construction: 构建时的搜索宽度，影响构建质量
    - ef_search: 搜索时的候选队列大小，影响搜索精度
    
    Args:
        dimension (int): 向量维度
        M (int): 每层最大连接数，默认32
        ef_construction (int): 构建时ef值，默认200
        ef_search (int): 搜索时ef值，默认128
        metric (str): 距离度量 ("l2" 或 "ip")
    """
    
    def __init__(
        self,
        dimension: int,
        M: int = 32,
        ef_construction: int = 200,
        ef_search: int = 128,
        metric: str = "l2"
    ):
        super().__init__(dimension, metric)
        
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        
        self._create_index()
        
    def _create_index(self) -> None:
        """
        创建HNSW索引
        """
        import faiss
        
        if self.metric == "ip":
            self.index = faiss.IndexHNSWFlat(self.dimension, self.M, faiss.METRIC_INNER_PRODUCT)
        else:
            self.index = faiss.IndexHNSWFlat(self.dimension, self.M)
            
        # 设置构建参数
        self.index.hnsw.efConstruction = self.ef_construction
        self.is_trained = True  # HNSW不需要训练
        
    def build(self, vectors: np.ndarray) -> None:
        """
        构建索引
        
        Args:
            vectors: 向量数组 [num_vectors, dimension]
        """
        vectors = np.ascontiguousarray(vectors.astype(np.float32))
        
        # 对于内积度量，需要归一化
        if self.metric == "ip":
            vectors = self.normalize_vectors(vectors)
            
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
            
        # 设置搜索参数
        self.index.hnsw.efSearch = self.ef_search
        
        distances, indices = self.index.search(queries, k)
        
        return distances, indices
    
    def set_ef_search(self, ef_search: int) -> None:
        """
        设置搜索时的ef参数
        
        ef_search越大，搜索精度越高但速度越慢。
        
        Args:
            ef_search: 新的ef值
        """
        self.ef_search = ef_search
        if self.index is not None:
            self.index.hnsw.efSearch = ef_search
            
    def get_graph_stats(self) -> Dict[str, Any]:
        """
        获取图结构统计信息
        
        用于分析HNSW图的结构特性。
        
        Returns:
            统计信息字典
        """
        # TODO: 实现图结构分析
        return {
            'num_vectors': self.num_vectors,
            'M': self.M,
            'ef_construction': self.ef_construction,
            'ef_search': self.ef_search,
            'dimension': self.dimension,
            'metric': self.metric
        }
    
    def get_neighbors(self, node_id: int, level: int = 0) -> np.ndarray:
        """
        获取节点的邻居
        
        用于分析图拓扑结构。
        
        Args:
            node_id: 节点ID
            level: HNSW层级
            
        Returns:
            邻居节点ID数组
        """
        # TODO: 实现邻居获取
        pass
    
    def trace_search_path(
        self,
        query: np.ndarray,
        k: int
    ) -> Dict[str, Any]:
        """
        追踪搜索路径
        
        记录搜索过程中访问的节点，用于分析搜索行为。
        
        Args:
            query: 查询向量 [dimension]
            k: 返回的最近邻数量
            
        Returns:
            包含搜索路径信息的字典
        """
        # TODO: 实现搜索路径追踪
        pass