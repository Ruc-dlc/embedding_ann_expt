"""
语义评估（Recall、MRR）

本模块实现检索结果的语义质量评估。

论文章节：第5章 5.2节 - 语义评估指标
"""

import numpy as np
from typing import Dict, List, Optional, Any, Union


class SemanticEvaluator:
    """
    语义评估器
    
    评估检索系统的语义匹配质量。
    
    Args:
        k_values: 要评估的k值列表
    """
    
    def __init__(self, k_values: List[int] = None):
        if k_values is None:
            k_values = [1, 5, 10, 20, 50, 100]
        self.k_values = k_values
        
    def evaluate(
        self,
        predictions: List[List[int]],
        ground_truths: List[List[int]],
        relevance_scores: Optional[List[Dict[int, float]]] = None
    ) -> Dict[str, float]:
        """
        执行完整评估
        Perform complete evaluation
        
        Args:
            predictions: 每个查询的预测结果ID列表
            ground_truths: 每个查询的真实相关ID列表
            relevance_scores: 相关性分数（用于NDCG）
            
        Returns:
            评估指标字典
        """
        results = {}
        
        # Recall@k
        for k in self.k_values:
            results[f'recall@{k}'] = self.compute_recall(predictions, ground_truths, k)
            
        # MRR@k
        for k in [10, 100]:
            if k in self.k_values or k <= max(self.k_values):
                results[f'mrr@{k}'] = self.compute_mrr(predictions, ground_truths, k)
                
        # MAP
        results['map'] = self.compute_map(predictions, ground_truths)
        
        # NDCG (if relevance scores provided)
        if relevance_scores is not None:
            for k in [10, 100]:
                results[f'ndcg@{k}'] = self.compute_ndcg(
                    predictions, relevance_scores, k
                )
                
        return results
    
    def compute_recall(
        self,
        predictions: List[List[int]],
        ground_truths: List[List[int]],
        k: int
    ) -> float:
        """
        计算Recall@k
        Compute Recall@k
        """
        recalls = []
        
        for pred, gt in zip(predictions, ground_truths):
            if len(gt) == 0:
                continue
                
            pred_set = set(pred[:k])
            gt_set = set(gt)
            
            recall = len(pred_set & gt_set) / min(k, len(gt_set))
            recalls.append(recall)
            
        return float(np.mean(recalls)) if recalls else 0.0
    
    def compute_mrr(
        self,
        predictions: List[List[int]],
        ground_truths: List[List[int]],
        k: Optional[int] = None
    ) -> float:
        """
        计算MRR (Mean Reciprocal Rank)
        Compute MRR
        """
        mrrs = []
        
        for pred, gt in zip(predictions, ground_truths):
            gt_set = set(gt)
            pred_list = pred[:k] if k else pred
            
            rr = 0.0
            for rank, doc_id in enumerate(pred_list, 1):
                if doc_id in gt_set:
                    rr = 1.0 / rank
                    break
                    
            mrrs.append(rr)
            
        return float(np.mean(mrrs)) if mrrs else 0.0
    
    def compute_map(
        self,
        predictions: List[List[int]],
        ground_truths: List[List[int]]
    ) -> float:
        """
        计算MAP (Mean Average Precision)
        Compute MAP
        """
        aps = []
        
        for pred, gt in zip(predictions, ground_truths):
            gt_set = set(gt)
            
            if len(gt_set) == 0:
                continue
                
            hits = 0
            precision_sum = 0.0
            
            for rank, doc_id in enumerate(pred, 1):
                if doc_id in gt_set:
                    hits += 1
                    precision_sum += hits / rank
                    
            ap = precision_sum / len(gt_set) if hits > 0 else 0.0
            aps.append(ap)
            
        return float(np.mean(aps)) if aps else 0.0
    
    def compute_ndcg(
        self,
        predictions: List[List[int]],
        relevance_scores: List[Dict[int, float]],
        k: int
    ) -> float:
        """
        计算NDCG@k
        Compute NDCG@k
        """
        ndcgs = []
        
        for pred, rel_scores in zip(predictions, relevance_scores):
            # DCG
            dcg = 0.0
            for i, doc_id in enumerate(pred[:k]):
                rel = rel_scores.get(doc_id, 0.0)
                dcg += rel / np.log2(i + 2)
                
            # IDCG
            ideal_rels = sorted(rel_scores.values(), reverse=True)[:k]
            idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_rels))
            
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcgs.append(ndcg)
            
        return float(np.mean(ndcgs)) if ndcgs else 0.0
    
    def compute_success_rate(
        self,
        predictions: List[List[int]],
        ground_truths: List[List[int]],
        k: int
    ) -> float:
        """
        计算Success@k（至少召回一个相关文档的比例）
        Compute Success@k (proportion of queries with at least one relevant doc in top-k)
        """
        successes = 0
        
        for pred, gt in zip(predictions, ground_truths):
            pred_set = set(pred[:k])
            gt_set = set(gt)
            
            if pred_set & gt_set:
                successes += 1
                
        return successes / len(predictions) if predictions else 0.0