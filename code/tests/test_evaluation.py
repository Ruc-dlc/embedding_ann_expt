#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评估模块单元测试

本模块测试评估指标计算的正确性。
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSemanticMetrics:
    """语义指标测试"""
    
    def test_recall_at_k(self):
        """测试Recall@k计算"""
        from src.utils.metrics import compute_recall
        
        predictions = [1, 2, 3, 4, 5]
        ground_truth = [1, 3, 7]
        
        recall = compute_recall(predictions, ground_truth, k=5)
        
        # 预测的前5个中有2个在ground_truth中 (1和3)
        # Recall@5 = 2 / min(5, 3) = 2/3
        assert abs(recall - 2/3) < 0.01
        
    def test_recall_perfect(self):
        """测试完美召回"""
        from src.utils.metrics import compute_recall
        
        predictions = [1, 2, 3, 4, 5]
        ground_truth = [1, 2, 3]
        
        recall = compute_recall(predictions, ground_truth, k=3)
        
        assert recall == 1.0
        
    def test_recall_zero(self):
        """测试零召回"""
        from src.utils.metrics import compute_recall
        
        predictions = [1, 2, 3]
        ground_truth = [4, 5, 6]
        
        recall = compute_recall(predictions, ground_truth, k=3)
        
        assert recall == 0.0
        
    def test_mrr(self):
        """测试MRR计算"""
        from src.utils.metrics import compute_mrr
        
        # 第一个相关文档在位置2
        predictions = [1, 3, 5]
        ground_truth = [3, 7]
        
        mrr = compute_mrr(predictions, ground_truth)
        
        # MRR = 1/2 = 0.5
        assert abs(mrr - 0.5) < 0.01
        
    def test_mrr_first_position(self):
        """测试第一位置的MRR"""
        from src.utils.metrics import compute_mrr
        
        predictions = [3, 1, 5]
        ground_truth = [3, 7]
        
        mrr = compute_mrr(predictions, ground_truth)
        
        assert mrr == 1.0
        
    def test_mrr_not_found(self):
        """测试未找到相关文档的MRR"""
        from src.utils.metrics import compute_mrr
        
        predictions = [1, 2, 3]
        ground_truth = [4, 5]
        
        mrr = compute_mrr(predictions, ground_truth, k=3)
        
        assert mrr == 0.0


class TestBatchMetrics:
    """批量指标测试"""
    
    def test_recall_batch(self):
        """测试批量Recall计算"""
        from src.utils.metrics import compute_recall_batch
        
        predictions = [
            [1, 2, 3],
            [4, 5, 6],
            [1, 4, 7]
        ]
        ground_truths = [
            [1, 2],
            [7, 8],
            [1, 7]
        ]
        
        recall = compute_recall_batch(predictions, ground_truths, k=3)
        
        # Query 1: 2/2 = 1.0
        # Query 2: 0/2 = 0.0
        # Query 3: 2/2 = 1.0
        # Mean: (1.0 + 0.0 + 1.0) / 3 = 0.667
        assert abs(recall - 2/3) < 0.01
        
    def test_mrr_batch(self):
        """测试批量MRR计算"""
        from src.utils.metrics import compute_mrr_batch
        
        predictions = [
            [1, 2, 3],  # 相关在位置1
            [4, 5, 6],  # 无相关
            [7, 1, 4]   # 相关在位置2
        ]
        ground_truths = [
            [1],
            [9],
            [1]
        ]
        
        mrr = compute_mrr_batch(predictions, ground_truths)
        
        # Query 1: 1/1 = 1.0
        # Query 2: 0
        # Query 3: 1/2 = 0.5
        # Mean: (1.0 + 0 + 0.5) / 3 = 0.5
        assert abs(mrr - 0.5) < 0.01


class TestMetricsComputer:
    """指标计算器测试"""
    
    def test_compute_all_metrics(self):
        """测试计算所有指标"""
        from src.utils.metrics import MetricsComputer
        
        computer = MetricsComputer(metrics=['recall@10', 'mrr@10'])
        
        predictions = [
            list(range(100)),
            list(range(100))
        ]
        ground_truths = [
            [0, 5, 50],
            [1, 2, 3]
        ]
        
        results = computer.compute(predictions, ground_truths)
        
        assert 'recall@10' in results
        assert 'mrr@10' in results


class TestSemanticEvaluator:
    """语义评估器测试"""
    
    def test_evaluator_basic(self):
        """测试评估器基本功能"""
        from evaluation.semantic_eval import SemanticEvaluator
        
        evaluator = SemanticEvaluator(k_values=[1, 5, 10])
        
        predictions = [list(range(100)) for _ in range(10)]
        ground_truths = [[0, 1, 2] for _ in range(10)]
        
        results = evaluator.evaluate(predictions, ground_truths)
        
        assert 'recall@1' in results
        assert 'recall@5' in results
        assert 'recall@10' in results
        assert 'map' in results


class TestEfficiencyEvaluator:
    """效率评估器测试"""
    
    def test_latency_measurement(self):
        """测试延迟测量"""
        from evaluation.efficiency_eval import EfficiencyEvaluator
        import time
        
        evaluator = EfficiencyEvaluator(warmup_runs=1, num_runs=3)
        
        def mock_search(queries, k):
            time.sleep(0.001)  # 模拟1ms延迟
            return np.zeros((len(queries), k)), np.zeros((len(queries), k), dtype=int)
            
        queries = np.random.randn(10, 128).astype('float32')
        
        latency_stats = evaluator.measure_single_query_latency(mock_search, queries, k=10, num_samples=5)
        
        assert latency_stats.mean_ms > 0
        assert latency_stats.p50_ms > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])