"""
效率评估（延迟、QPS、Visited Nodes）

本模块实现检索系统的效率评估。

论文章节：第5章 5.4节 - ANN搜索效率分析
"""

import time
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass


@dataclass
class LatencyStats:
    """延迟统计"""
    mean_ms: float
    std_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'mean_ms': self.mean_ms,
            'std_ms': self.std_ms,
            'p50_ms': self.p50_ms,
            'p95_ms': self.p95_ms,
            'p99_ms': self.p99_ms,
            'min_ms': self.min_ms,
            'max_ms': self.max_ms
        }


class EfficiencyEvaluator:
    """
    效率评估器
    
    评估检索系统的延迟和吞吐量性能。
    
    Args:
        warmup_runs (int): 预热运行次数
        num_runs (int): 正式测试运行次数
    """
    
    def __init__(self, warmup_runs: int = 3, num_runs: int = 10):
        self.warmup_runs = warmup_runs
        self.num_runs = num_runs
        
    def evaluate(
        self,
        search_fn: Callable,
        queries: np.ndarray,
        k: int = 10,
        batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        执行效率评估
        
        Args:
            search_fn: 搜索函数 (queries, k) -> (distances, indices)
            queries: 查询向量
            k: 返回数量
            batch_size: 批次大小（如果为None则一次处理所有查询）
            
        Returns:
            效率指标字典
        """
        results = {}
        
        # 单查询延迟
        single_latency = self.measure_single_query_latency(search_fn, queries, k)
        results['single_query_latency'] = single_latency.to_dict()
        
        # 批量QPS
        if batch_size is None:
            batch_size = len(queries)
        qps = self.measure_qps(search_fn, queries, k, batch_size)
        results['qps'] = qps
        
        # 吞吐量
        throughput = self.measure_throughput(search_fn, queries, k)
        results['throughput'] = throughput
        
        return results
    
    def measure_single_query_latency(
        self,
        search_fn: Callable,
        queries: np.ndarray,
        k: int = 10,
        num_samples: int = 100
    ) -> LatencyStats:
        """
        测量单查询延迟
        """
        # 采样查询
        if len(queries) > num_samples:
            indices = np.random.choice(len(queries), num_samples, replace=False)
            sample_queries = queries[indices]
        else:
            sample_queries = queries
            
        # 预热
        for _ in range(self.warmup_runs):
            search_fn(sample_queries[:1], k)
            
        # 测量每个查询的延迟
        latencies = []
        for query in sample_queries:
            query = query.reshape(1, -1)
            
            start_time = time.perf_counter()
            search_fn(query, k)
            elapsed = time.perf_counter() - start_time
            
            latencies.append(elapsed * 1000)  # 转换为毫秒
            
        latencies = np.array(latencies)
        
        return LatencyStats(
            mean_ms=float(np.mean(latencies)),
            std_ms=float(np.std(latencies)),
            p50_ms=float(np.percentile(latencies, 50)),
            p95_ms=float(np.percentile(latencies, 95)),
            p99_ms=float(np.percentile(latencies, 99)),
            min_ms=float(np.min(latencies)),
            max_ms=float(np.max(latencies))
        )
    
    def measure_qps(
        self,
        search_fn: Callable,
        queries: np.ndarray,
        k: int = 10,
        batch_size: int = 100
    ) -> Dict[str, float]:
        """
        测量QPS (Queries Per Second)
        """
        # 预热
        for _ in range(self.warmup_runs):
            search_fn(queries[:batch_size], k)
            
        # 测量
        times = []
        for _ in range(self.num_runs):
            start_time = time.perf_counter()
            search_fn(queries[:batch_size], k)
            elapsed = time.perf_counter() - start_time
            times.append(elapsed)
            
        avg_time = np.mean(times)
        qps = batch_size / avg_time
        
        return {
            'qps': float(qps),
            'batch_size': batch_size,
            'avg_batch_time_ms': float(avg_time * 1000),
            'std_batch_time_ms': float(np.std(times) * 1000)
        }
    
    def measure_throughput(
        self,
        search_fn: Callable,
        queries: np.ndarray,
        k: int = 10,
        duration_seconds: float = 5.0
    ) -> Dict[str, float]:
        """
        测量持续吞吐量
        """
        # 预热
        for _ in range(self.warmup_runs):
            search_fn(queries, k)
            
        # 持续测量
        total_queries = 0
        start_time = time.perf_counter()
        
        while time.perf_counter() - start_time < duration_seconds:
            search_fn(queries, k)
            total_queries += len(queries)
            
        elapsed = time.perf_counter() - start_time
        
        return {
            'total_queries': total_queries,
            'total_time_s': float(elapsed),
            'throughput_qps': float(total_queries / elapsed)
        }
    
    def compare_methods(
        self,
        methods: Dict[str, Callable],
        queries: np.ndarray,
        k: int = 10
    ) -> Dict[str, Dict[str, Any]]:
        """
        比较多个方法的效率
        
        Args:
            methods: {方法名: 搜索函数}
            queries: 查询向量
            k: 返回数量
            
        Returns:
            每个方法的效率指标
        """
        results = {}
        
        for name, search_fn in methods.items():
            results[name] = self.evaluate(search_fn, queries, k)
            
        return results