"""
综合评估流水线

本模块整合所有评估组件，提供完整的评估流水线。

论文章节：第5章 - 综合评估
"""

import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import time

from .semantic_eval import SemanticEvaluator
from .efficiency_eval import EfficiencyEvaluator


class ComprehensiveEvaluator:
    """
    综合评估器

    整合语义评估和效率评估的完整评估流水线。

    Args:
        index: 向量索引
        ground_truth_index: 精确索引（用于计算ground truth）
        output_dir: 结果输出目录
    """

    def __init__(
        self,
        index: Any,
        ground_truth_index: Optional[Any] = None,
        output_dir: Optional[str] = None
    ):
        self.index = index
        self.ground_truth_index = ground_truth_index
        self.output_dir = Path(output_dir) if output_dir else None

        # 初始化子评估器
        self.semantic_evaluator = SemanticEvaluator()
        self.efficiency_evaluator = EfficiencyEvaluator()

    def evaluate(
        self,
        queries: np.ndarray,
        ground_truths: Optional[List[List[int]]] = None,
        k_values: List[int] = None,
        run_efficiency: bool = True,
        experiment_name: str = "experiment"
    ) -> Dict[str, Any]:
        """
        执行综合评估

        Args:
            queries: 查询向量
            ground_truths: 真实相关文档ID列表
            k_values: 评估的k值列表
            run_efficiency: 是否运行效率评估
            experiment_name: 实验名称

        Returns:
            综合评估结果
        """
        if k_values is None:
            k_values = [1, 5, 10, 20, 50, 100]

        results = {
            'experiment_name': experiment_name,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'num_queries': len(queries),
            'k_values': k_values
        }

        # 计算ground truth（如果需要）
        if ground_truths is None and self.ground_truth_index is not None:
            max_k = max(k_values)
            _, gt_indices = self.ground_truth_index.search(queries, max_k)
            ground_truths = [list(gt) for gt in gt_indices]

        # 获取预测结果
        max_k = max(k_values) if k_values else 100
        distances, indices = self.index.search(queries, max_k)
        predictions = [list(pred) for pred in indices]

        # 语义评估
        if ground_truths is not None:
            self.semantic_evaluator.k_values = k_values
            results['semantic'] = self.semantic_evaluator.evaluate(
                predictions, ground_truths
            )

        # 效率评估
        if run_efficiency:
            search_fn = lambda q, k: self.index.search(q, k)
            results['efficiency'] = self.efficiency_evaluator.evaluate(
                search_fn, queries, k=10
            )

        # 保存结果
        if self.output_dir:
            self._save_results(results, experiment_name)

        return results

    def compare_methods(
        self,
        methods: Dict[str, Any],
        queries: np.ndarray,
        ground_truths: Optional[List[List[int]]] = None,
        k: int = 10
    ) -> Dict[str, Dict[str, Any]]:
        """
        比较多个方法

        Args:
            methods: {方法名: 索引实例}
            queries: 查询向量
            ground_truths: 真实相关文档
            k: 评估的k值

        Returns:
            每个方法的评估结果
        """
        results = {}

        original_index = self.index

        for name, index in methods.items():
            self.index = index

            results[name] = self.evaluate(
                queries,
                ground_truths,
                k_values=[k],
                experiment_name=name
            )

        self.index = original_index

        return results

    def run_ablation(
        self,
        queries: np.ndarray,
        ground_truths: List[List[int]],
        configs: Dict[str, Dict[str, Any]],
        k: int = 10
    ) -> Dict[str, Dict[str, Any]]:
        """
        运行消融实验

        Args:
            queries: 查询向量
            ground_truths: 真实相关文档
            configs: {配置名: 配置参数}
            k: 评估的k值

        Returns:
            消融实验结果
        """
        results = {}

        for config_name, config in configs.items():
            # 应用配置（例如调整索引参数）
            self._apply_config(config)

            # 评估
            results[config_name] = self.evaluate(
                queries,
                ground_truths,
                k_values=[k],
                experiment_name=f"ablation_{config_name}"
            )

        return results

    def _apply_config(self, config: Dict[str, Any]) -> None:
        """应用配置"""
        if 'ef_search' in config and hasattr(self.index, 'set_ef_search'):
            self.index.set_ef_search(config['ef_search'])
        if 'nprobe' in config and hasattr(self.index, 'set_nprobe'):
            self.index.set_nprobe(config['nprobe'])

    def _save_results(self, results: Dict[str, Any], name: str) -> None:
        """保存结果"""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 转换numpy类型
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        results = convert(results)

        output_path = self.output_dir / f"{name}_results.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

    def generate_report(self, results: Dict[str, Any]) -> str:
        """
        生成评估报告

        Args:
            results: 评估结果

        Returns:
            Markdown格式的报告
        """
        report = []
        report.append(f"# 评估报告: {results.get('experiment_name', 'Unknown')}")
        report.append(f"\n**日期:** {results.get('timestamp', 'N/A')}")
        report.append(f"\n**查询数量:** {results.get('num_queries', 0)}")

        # 语义指标
        if 'semantic' in results:
            report.append("\n## 语义检索指标")
            for metric, value in results['semantic'].items():
                report.append(f"- {metric}: {value:.4f}")

        # 效率指标
        if 'efficiency' in results:
            report.append("\n## 效率指标")
            eff = results['efficiency']
            if 'single_query_latency' in eff:
                lat = eff['single_query_latency']
                report.append(f"- 平均延迟: {lat.get('mean_ms', 0):.2f} ms")
                report.append(f"- P95延迟: {lat.get('p95_ms', 0):.2f} ms")
            if 'qps' in eff:
                report.append(f"- QPS: {eff['qps'].get('qps', 0):.1f}")

        return "\n".join(report)