"""
帕累托前沿分析

本模块实现召回率-效率帕累托前沿的分析和可视化。

论文章节：第5章 5.4节 - 帕累托分析
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any


class ParetoAnalyzer:
    """
    帕累托前沿分析器

    分析不同方法在召回率-延迟权衡上的帕累托最优性。
    """

    def __init__(self):
        self.results: Dict[str, Dict[str, float]] = {}

    def add_result(
        self,
        name: str,
        recall: float,
        latency: float,
        **kwargs
    ) -> None:
        """
        添加实验结果

        Args:
            name: 方法/配置名称
            recall: 召回率
            latency: 延迟（ms）
            **kwargs: 其他指标
        """
        self.results[name] = {
            'recall': recall,
            'latency': latency,
            **kwargs
        }

    def compute_pareto_frontier(
        self,
        maximize_recall: bool = True,
        minimize_latency: bool = True
    ) -> List[str]:
        """
        计算帕累托前沿

        找出所有帕累托最优的配置点。

        Args:
            maximize_recall: 是否最大化召回率
            minimize_latency: 是否最小化延迟

        Returns:
            帕累托最优配置名称列表
        """
        if not self.results:
            return []

        names = list(self.results.keys())
        points = np.array([
            [self.results[n]['recall'], self.results[n]['latency']]
            for n in names
        ])

        # 标准化方向（都变成越大越好）
        if minimize_latency:
            points[:, 1] = -points[:, 1]
        if not maximize_recall:
            points[:, 0] = -points[:, 0]

        # 找帕累托前沿
        pareto_mask = self._compute_pareto_mask(points)

        return [names[i] for i in range(len(names)) if pareto_mask[i]]

    def _compute_pareto_mask(self, points: np.ndarray) -> np.ndarray:
        """
        计算帕累托掩码

        Args:
            points: 点集 [n_points, n_objectives]

        Returns:
            布尔掩码，True表示帕累托最优
        """
        n_points = len(points)
        is_pareto = np.ones(n_points, dtype=bool)

        for i in range(n_points):
            if not is_pareto[i]:
                continue

            # 检查是否被其他点支配
            for j in range(n_points):
                if i == j or not is_pareto[j]:
                    continue

                # 如果j在所有维度都>=i，且至少一个维度>i，则i被支配
                if np.all(points[j] >= points[i]) and np.any(points[j] > points[i]):
                    is_pareto[i] = False
                    break

        return is_pareto

    def compute_pareto_improvement(
        self,
        baseline_name: str,
        target_name: str
    ) -> Dict[str, float]:
        """
        计算帕累托改进

        计算从baseline到target的帕累托改进量。

        Args:
            baseline_name: 基准方法名
            target_name: 目标方法名

        Returns:
            改进量字典
        """
        if baseline_name not in self.results or target_name not in self.results:
            return {}

        baseline = self.results[baseline_name]
        target = self.results[target_name]

        return {
            'recall_improvement': target['recall'] - baseline['recall'],
            'recall_improvement_pct': (target['recall'] - baseline['recall']) / baseline['recall'] * 100,
            'latency_improvement': baseline['latency'] - target['latency'],
            'latency_improvement_pct': (baseline['latency'] - target['latency']) / baseline['latency'] * 100
        }

    def find_best_at_latency(
        self,
        max_latency: float
    ) -> Optional[str]:
        """
        在延迟约束下找最佳方法

        Args:
            max_latency: 最大允许延迟

        Returns:
            最佳方法名（如果有）
        """
        best_name = None
        best_recall = -1

        for name, metrics in self.results.items():
            if metrics['latency'] <= max_latency and metrics['recall'] > best_recall:
                best_recall = metrics['recall']
                best_name = name

        return best_name

    def find_best_at_recall(
        self,
        min_recall: float
    ) -> Optional[str]:
        """
        在召回率约束下找最佳方法

        Args:
            min_recall: 最小要求召回率

        Returns:
            最佳方法名（如果有）
        """
        best_name = None
        best_latency = float('inf')

        for name, metrics in self.results.items():
            if metrics['recall'] >= min_recall and metrics['latency'] < best_latency:
                best_latency = metrics['latency']
                best_name = name

        return best_name

    def get_summary(self) -> Dict[str, Any]:
        """
        获取分析摘要

        Returns:
            摘要字典
        """
        pareto_frontier = self.compute_pareto_frontier()

        # 找极值点
        if self.results:
            best_recall_name = max(self.results.keys(),
                                   key=lambda n: self.results[n]['recall'])
            best_latency_name = min(self.results.keys(),
                                    key=lambda n: self.results[n]['latency'])
        else:
            best_recall_name = None
            best_latency_name = None

        return {
            'num_methods': len(self.results),
            'pareto_frontier': pareto_frontier,
            'num_pareto_optimal': len(pareto_frontier),
            'best_recall': {
                'name': best_recall_name,
                'metrics': self.results.get(best_recall_name, {})
            },
            'best_latency': {
                'name': best_latency_name,
                'metrics': self.results.get(best_latency_name, {})
            }
        }