"""
评估指标计算

本模块实现信息检索常用的评估指标。

论文章节：第5章 5.2节 - 评估指标
"""

import numpy as np
from typing import List, Dict, Optional, Union

def compute_recall(
    predictions: Union[List[int], np.ndarray],
    ground_truth: Union[List[int], np.ndarray],
    k: int
) -> float:
    """
    计算Recall@k
    
    Recall@k = |{预测的前k个} ∩ {真实相关}| / min(k, |真实相关|)
    
    Args:
        predictions: 预测的文档ID列表
        ground_truth: 真实相关的文档ID列表
        k: 评估的k值
        
    Returns:
        Recall@k值
    """
    if len(ground_truth) == 0:
        return 0.0
        
    pred_set = set(list(predictions)[:k])
    gt_set = set(ground_truth)
    
    return len(pred_set & gt_set) / min(k, len(gt_set))

def compute_recall_batch(
    predictions: List[List[int]],
    ground_truths: List[List[int]],
    k: int
) -> float:
    """
    批量计算Recall@k
    
    Args:
        predictions: 预测列表的列表
        ground_truths: 真实值列表的列表
        k: 评估的k值
        
    Returns:
        平均Recall@k
    """
    recalls = []
    for pred, gt in zip(predictions, ground_truths):
        recalls.append(compute_recall(pred, gt, k))
    return np.mean(recalls)

def compute_mrr(
    predictions: Union[List[int], np.ndarray],
    ground_truth: Union[List[int], np.ndarray],
    k: Optional[int] = None
) -> float:
    """
    计算MRR (Mean Reciprocal Rank)
    
    MRR = 1 / rank(第一个相关文档)
    
    Args:
        predictions: 预测的文档ID列表
        ground_truth: 真实相关的文档ID列表
        k: 只考虑前k个预测（可选）
        
    Returns:
        MRR值
    """
    gt_set = set(ground_truth)
    
    if k is not None:
        predictions = list(predictions)[:k]
        
    for rank, pred_id in enumerate(predictions, 1):
        if pred_id in gt_set:
            return 1.0 / rank
            
    return 0.0

def compute_mrr_batch(
    predictions: List[List[int]],
    ground_truths: List[List[int]],
    k: Optional[int] = None
) -> float:
    """
    批量计算MRR
    
    Args:
        predictions: 预测列表的列表
        ground_truths: 真实值列表的列表
        k: 只考虑前k个预测
        
    Returns:
        平均MRR
    """
    mrrs = []
    for pred, gt in zip(predictions, ground_truths):
        mrrs.append(compute_mrr(pred, gt, k))
    return np.mean(mrrs)

def compute_ndcg(
    predictions: Union[List[int], np.ndarray],
    relevance_scores: Dict[int, float],
    k: int
) -> float:
    """
    计算NDCG@k (Normalized Discounted Cumulative Gain)
    
    Args:
        predictions: 预测的文档ID列表
        relevance_scores: 文档ID到相关性分数的映射
        k: 评估的k值
        
    Returns:
        NDCG@k值
    """
    # 计算DCG
    dcg = 0.0
    for i, pred_id in enumerate(list(predictions)[:k]):
        rel = relevance_scores.get(pred_id, 0.0)
        dcg += rel / np.log2(i + 2)  # log2(rank + 1), rank从1开始
        
    # 计算理想DCG (IDCG)
    ideal_rels = sorted(relevance_scores.values(), reverse=True)[:k]
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_rels))
    
    if idcg == 0:
        return 0.0
        
    return dcg / idcg

def compute_precision(
    predictions: Union[List[int], np.ndarray],
    ground_truth: Union[List[int], np.ndarray],
    k: int
) -> float:
    """
    计算Precision@k
    
    Args:
        predictions: 预测的文档ID列表
        ground_truth: 真实相关的文档ID列表
        k: 评估的k值
        
    Returns:
        Precision@k值
    """
    if k == 0:
        return 0.0
        
    pred_set = set(list(predictions)[:k])
    gt_set = set(ground_truth)
    
    return len(pred_set & gt_set) / k

def compute_map(
    predictions: List[List[int]],
    ground_truths: List[List[int]],
    k: Optional[int] = None
) -> float:
    """
    计算MAP (Mean Average Precision)
    
    Args:
        predictions: 预测列表的列表
        ground_truths: 真实值列表的列表
        k: 只考虑前k个预测
        
    Returns:
        MAP值
    """
    aps = []
    
    for preds, gts in zip(predictions, ground_truths):
        gt_set = set(gts)
        
        if k is not None:
            preds = list(preds)[:k]
            
        if len(gt_set) == 0:
            aps.append(0.0)
            continue
            
        hits = 0
        precision_sum = 0.0
        
        for rank, pred_id in enumerate(preds, 1):
            if pred_id in gt_set:
                hits += 1
                precision_sum += hits / rank
                
        ap = precision_sum / min(len(gt_set), len(preds)) if hits > 0 else 0.0
        aps.append(ap)
        
    return np.mean(aps)

class MetricsComputer:
    """
    指标计算器
    
    统一的指标计算接口。
    
    Args:
        metrics (List[str]): 要计算的指标列表
    """
    
    SUPPORTED_METRICS = ['recall', 'mrr', 'ndcg', 'precision', 'map']
    
    def __init__(self, metrics: List[str] = None):
        if metrics is None:
            metrics = ['recall@10', 'recall@100', 'mrr@10']
        self.metrics = metrics
        
    def compute(
        self,
        predictions: List[List[int]],
        ground_truths: List[List[int]],
        relevance_scores: Optional[List[Dict[int, float]]] = None
    ) -> Dict[str, float]:
        """
        计算所有指标
        
        Args:
            predictions: 预测结果
            ground_truths: 真实标签
            relevance_scores: 相关性分数（用于NDCG）
            
        Returns:
            指标字典
        """
        results = {}
        
        for metric in self.metrics:
            if '@' in metric:
                metric_name, k = metric.split('@')
                k = int(k)
            else:
                metric_name = metric
                k = 10
                
            if metric_name == 'recall':
                results[metric] = compute_recall_batch(predictions, ground_truths, k)
            elif metric_name == 'mrr':
                results[metric] = compute_mrr_batch(predictions, ground_truths, k)
            elif metric_name == 'map':
                results[metric] = compute_map(predictions, ground_truths, k)
            # TODO: 添加其他指标
            
        return results