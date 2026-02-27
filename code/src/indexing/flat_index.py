"""
Flat精确索引（基准）

本模块实现精确最近邻搜索的Flat索引，作为计算召回率的ground truth基准。

论文章节：第5章 5.2节 - 基准方法
"""

import numpy as np
from typing import Tuple, Optional
from .faiss_wrapper import FaissIndex

class FlatIndex(FaissIndex):
    """
    Flat精确索引
    
    暴力搜索所有向量，保证100%召回率，用作ground truth。
    
    注意：对于大规模数据集，搜索效率较低。
    
    Args:
        dimension (int): 向量维度
        metric (str): 距离度量 ("l2" 或 "ip")
    """
    
    def __init__(self, dimension: int, metric: str = "l2"):
        super().__init__(dimension, metric)
        self._create_index()
        
    def _create_index(self) -> None:
        """
        创建Flat索引
        """
        import faiss
        
        if self.metric == "ip":
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
            
        self.is_trained = True  # Flat索引不需要训练
        
    def build(self, vectors: np.ndarray) -> None:
        """
        构建索引
        
        Args:
            vectors: 向量数组 [num_vectors, dimension]
        """
        vectors = np.ascontiguousarray(vectors.astype(np.float32))
        
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
        精确搜索最近邻
        
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
            
        distances, indices = self.index.search(queries, k)
        
        return distances, indices
    
    def compute_ground_truth(
        self,
        queries: np.ndarray,
        k: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算ground truth
        
        用于评估近似索引的召回率。
        
        Args:
            queries: 查询向量
            k: 返回数量
            
        Returns:
            gt_distances: 真实距离
            gt_indices: 真实索引
        """
        return self.search(queries, k)

def compute_recall(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    k: int
) -> float:
    """
    计算Recall@k
    
    Args:
        predictions: 预测的最近邻索引 [num_queries, k]
        ground_truth: 真实的最近邻索引 [num_queries, k_gt]
        k: 评估的k值
        
    Returns:
        Recall@k值
    """
    num_queries = predictions.shape[0]
    
    recall_sum = 0.0
    for i in range(num_queries):
        pred_set = set(predictions[i, :k].tolist())
        gt_set = set(ground_truth[i, :k].tolist())
        recall_sum += len(pred_set & gt_set) / k
        
    return recall_sum / num_queries