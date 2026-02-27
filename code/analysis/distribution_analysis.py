"""
向量分布分析（紧致度、均匀性）

本模块实现向量空间分布特性的分析，包括：
- 紧致度 (Compactness): 正样本对的余弦距离分布
- 均匀性 (Uniformity): 向量在空间中的分布均匀程度
- 对齐度 (Alignment): 正样本对的对齐程度

论文章节：第5章 5.3节 - 向量分布分析
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats


def _cosine_similarity_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    批量计算余弦相似度

    Args:
        a: 向量数组 [N, dim]
        b: 向量数组 [N, dim]

    Returns:
        余弦相似度数组 [N]
    """
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return np.sum(a_norm * b_norm, axis=1)


def _cosine_distance_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    批量计算余弦距离 = 1 - cos_sim

    Args:
        a: 向量数组 [N, dim]
        b: 向量数组 [N, dim]

    Returns:
        余弦距离数组 [N]，范围 [0, 2]
    """
    return 1.0 - _cosine_similarity_batch(a, b)


class DistributionAnalyzer:
    """
    向量分布分析器

    分析向量空间的分布特性，用于验证距离感知训练的效果。
    所有距离度量均使用余弦距离（1 - cos_sim），与论文损失函数一致。

    Args:
        embeddings: 向量数组 [num_vectors, embedding_dim]
    """

    def __init__(self, embeddings: Optional[np.ndarray] = None):
        self.embeddings = embeddings

    def set_embeddings(self, embeddings: np.ndarray) -> None:
        """设置要分析的向量"""
        self.embeddings = embeddings

    def compute_compactness(
        self,
        query_embeddings: np.ndarray,
        pos_doc_embeddings: np.ndarray
    ) -> Dict[str, float]:
        """
        计算紧致度

        分析正样本对之间的余弦距离分布。
        余弦距离越小，说明正样本对越紧致。

        Args:
            query_embeddings: Query向量 [num_queries, dim]
            pos_doc_embeddings: 正样本文档向量 [num_queries, dim]

        Returns:
            紧致度统计信息
        """
        cos_distances = _cosine_distance_batch(query_embeddings, pos_doc_embeddings)

        return {
            'mean_cosine_distance': float(np.mean(cos_distances)),
            'std_cosine_distance': float(np.std(cos_distances)),
            'min_cosine_distance': float(np.min(cos_distances)),
            'max_cosine_distance': float(np.max(cos_distances)),
            'median_cosine_distance': float(np.median(cos_distances)),
            'q25_cosine_distance': float(np.percentile(cos_distances, 25)),
            'q75_cosine_distance': float(np.percentile(cos_distances, 75))
        }

    def compute_uniformity(
        self,
        embeddings: Optional[np.ndarray] = None,
        sample_size: int = 10000,
        t: float = 2.0
    ) -> float:
        """
        计算均匀性

        使用Wang & Isola (2020)提出的均匀性度量。
        L_uniform = log E[exp(-t * ||f(x) - f(y)||^2)]

        Args:
            embeddings: 向量数组（如果为None则使用self.embeddings）
            sample_size: 采样数量（用于大规模数据）
            t: 温度参数

        Returns:
            均匀性分数（越小越均匀，理论最小值为-dim*log(2)/t）
        """
        if embeddings is None:
            embeddings = self.embeddings

        if embeddings is None:
            raise ValueError("未提供向量数据")

        n = len(embeddings)

        # 对大规模数据进行采样
        if n > sample_size:
            indices = np.random.choice(n, sample_size, replace=False)
            embeddings = embeddings[indices]
            n = sample_size

        # L2归一化
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)

        # 计算所有pair的L2距离平方（归一化后 ||a-b||^2 = 2*(1-cos_sim)）
        sq_distances = np.sum(
            (embeddings[:, np.newaxis, :] - embeddings[np.newaxis, :, :]) ** 2,
            axis=2
        )

        # 排除对角线
        mask = ~np.eye(n, dtype=bool)
        sq_distances = sq_distances[mask]

        # 计算均匀性
        uniformity = np.log(np.mean(np.exp(-t * sq_distances)))

        return float(uniformity)

    def compute_alignment(
        self,
        query_embeddings: np.ndarray,
        pos_doc_embeddings: np.ndarray,
        alpha: float = 2.0
    ) -> float:
        """
        计算对齐度

        度量正样本对之间的对齐程度，使用余弦距离。
        L_align = E[(1 - cos_sim(q, d+))^alpha]

        Args:
            query_embeddings: Query向量
            pos_doc_embeddings: 正样本向量
            alpha: 指数参数

        Returns:
            对齐度分数（越小越好）
        """
        cos_distances = _cosine_distance_batch(query_embeddings, pos_doc_embeddings)
        alignment = np.mean(cos_distances ** alpha)

        return float(alignment)

    def compute_distance_distribution(
        self,
        query_embeddings: np.ndarray,
        pos_doc_embeddings: np.ndarray,
        neg_doc_embeddings: np.ndarray
    ) -> Dict[str, Any]:
        """
        计算余弦距离分布

        分析正负样本的余弦距离分布特性。
        正样本距离应小，负样本距离应大，两者重叠越少越好。

        Args:
            query_embeddings: Query向量 [num_queries, dim]
            pos_doc_embeddings: 正样本向量 [num_queries, dim]
            neg_doc_embeddings: 负样本向量 [num_queries, num_negatives, dim]

        Returns:
            距离分布统计
        """
        # 正样本余弦距离
        pos_cos_distances = _cosine_distance_batch(query_embeddings, pos_doc_embeddings)

        # 负样本余弦距离
        num_queries, num_negatives, dim = neg_doc_embeddings.shape
        neg_cos_distances_list = []
        for j in range(num_negatives):
            neg_cos_dist = _cosine_distance_batch(
                query_embeddings, neg_doc_embeddings[:, j, :]
            )
            neg_cos_distances_list.append(neg_cos_dist)
        neg_cos_distances_flat = np.concatenate(neg_cos_distances_list)

        # 使用KDE估计分布重叠
        pos_kde = stats.gaussian_kde(pos_cos_distances)
        neg_kde = stats.gaussian_kde(neg_cos_distances_flat)

        x_range = np.linspace(
            0,
            max(pos_cos_distances.max(), neg_cos_distances_flat.max()),
            1000
        )
        pos_density = pos_kde(x_range)
        neg_density = neg_kde(x_range)

        # 计算重叠面积（越小越好）
        overlap = np.trapz(np.minimum(pos_density, neg_density), x_range)

        return {
            'positive': {
                'mean': float(np.mean(pos_cos_distances)),
                'std': float(np.std(pos_cos_distances)),
                'min': float(np.min(pos_cos_distances)),
                'max': float(np.max(pos_cos_distances))
            },
            'negative': {
                'mean': float(np.mean(neg_cos_distances_flat)),
                'std': float(np.std(neg_cos_distances_flat)),
                'min': float(np.min(neg_cos_distances_flat)),
                'max': float(np.max(neg_cos_distances_flat))
            },
            'separation': float(np.mean(neg_cos_distances_flat) - np.mean(pos_cos_distances)),
            'overlap': float(overlap)
        }

    def full_analysis(
        self,
        query_embeddings: np.ndarray,
        pos_doc_embeddings: np.ndarray,
        neg_doc_embeddings: Optional[np.ndarray] = None,
        doc_embeddings: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        完整分析

        执行所有分布分析并汇总结果。

        Args:
            query_embeddings: Query向量
            pos_doc_embeddings: 正样本向量
            neg_doc_embeddings: 负样本向量（可选）
            doc_embeddings: 全部文档向量（可选，用于均匀性分析）

        Returns:
            完整分析结果
        """
        results = {}

        # 紧致度
        results['compactness'] = self.compute_compactness(query_embeddings, pos_doc_embeddings)

        # 对齐度
        results['alignment'] = self.compute_alignment(query_embeddings, pos_doc_embeddings)

        # 均匀性（如果提供了文档向量）
        if doc_embeddings is not None:
            results['uniformity'] = self.compute_uniformity(doc_embeddings)

        # 距离分布（如果提供了负样本）
        if neg_doc_embeddings is not None:
            results['distance_distribution'] = self.compute_distance_distribution(
                query_embeddings, pos_doc_embeddings, neg_doc_embeddings
            )

        return results