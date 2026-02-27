"""
HNSW 搜索效率分析器

主要功能：
1. 加载或构建 HNSW 索引
2. 执行搜索并收集 Visited Nodes、延迟等统计信息
3. 分析 ef_search 参数对搜索质量和效率的影响
4. 生成 Recall vs Latency 和 Visited Nodes vs ef 对比图

论文章节：第5章 §5.4 - ANN搜索效率分析
"""

import numpy as np
import time
from typing import List, Tuple, Dict, Optional, Set, Any


class HNSWSimulator:
    """
    HNSW 搜索效率分析器

    基于FAISS构建HNSW索引，通过搜索统计信息分析搜索效率。
    核心指标：Visited Nodes（访问节点数）、Latency（延迟）、Recall。

    Args:
        dimension: 向量维度
        M: HNSW每层最大邻居数
        ef_construction: 构建时的搜索宽度
    """

    def __init__(self, dimension: int, M: int = 32, ef_construction: int = 200):
        self.dimension = dimension
        self.M = M
        self.ef_construction = ef_construction
        self.vectors = None
        self.faiss_index = None
        self.num_vectors = 0

    def build_index(self, data: np.ndarray) -> None:
        """
        构建HNSW索引

        Args:
            data: 向量数据 [num_vectors, dimension]
        """
        import faiss

        self.vectors = data.astype(np.float32)
        self.num_vectors = len(data)
        self.dimension = data.shape[1]

        self.faiss_index = faiss.IndexHNSWFlat(self.dimension, self.M)
        self.faiss_index.hnsw.efConstruction = self.ef_construction
        self.faiss_index.add(self.vectors)

    def load_index(self, index_path: str, vectors: Optional[np.ndarray] = None) -> None:
        """
        从文件加载FAISS索引

        Args:
            index_path: 索引文件路径
            vectors: 原始向量（用于距离分析，可选）
        """
        import faiss

        self.faiss_index = faiss.read_index(index_path)
        self.num_vectors = self.faiss_index.ntotal
        self.dimension = self.faiss_index.d

        if vectors is not None:
            self.vectors = vectors.astype(np.float32)
        elif hasattr(self.faiss_index, 'reconstruct_n'):
            self.vectors = self.faiss_index.reconstruct_n(0, self.num_vectors)

    def search_with_stats(
        self,
        queries: np.ndarray,
        k: int = 10,
        ef_search: int = 128
    ) -> Dict[str, Any]:
        """
        执行搜索并收集统计信息

        利用FAISS的搜索统计接口获取visited节点数等信息。

        Args:
            queries: 查询向量 [num_queries, dimension]
            k: 返回最近邻数量
            ef_search: 搜索时的ef参数

        Returns:
            搜索统计信息字典，包含 avg_visited_nodes、avg_latency_ms 等
        """
        import faiss

        queries = queries.astype(np.float32)
        num_queries = len(queries)

        # 设置ef_search参数
        self.faiss_index.hnsw.efSearch = ef_search

        # 重置统计计数器
        if hasattr(faiss, 'cvar') and hasattr(faiss.cvar, 'hnsw_stats'):
            faiss.cvar.hnsw_stats.reset()

        # 执行搜索并计时
        start_time = time.time()
        distances, indices = self.faiss_index.search(queries, k)
        search_time = time.time() - start_time

        # 收集HNSW搜索统计（FAISS的hnsw_stats）
        total_distance_computations = 0
        if hasattr(faiss, 'cvar') and hasattr(faiss.cvar, 'hnsw_stats'):
            total_distance_computations = faiss.cvar.hnsw_stats.ndis

        avg_visited_nodes = float(total_distance_computations / num_queries) if num_queries > 0 else 0

        stats = {
            'num_queries': num_queries,
            'k': k,
            'ef_search': ef_search,
            'total_time_ms': float(search_time * 1000),
            'avg_latency_ms': float(search_time * 1000 / num_queries),
            'qps': float(num_queries / search_time) if search_time > 0 else 0,
            'total_distance_computations': int(total_distance_computations),
            'avg_visited_nodes': avg_visited_nodes,
            'distances': distances,
            'indices': indices
        }

        return stats

    def analyze_ef_sensitivity(
        self,
        queries: np.ndarray,
        ground_truth: np.ndarray,
        ef_values: Optional[List[int]] = None,
        k: int = 10
    ) -> Dict[int, Dict[str, float]]:
        """
        分析ef参数对搜索质量和效率的影响

        在不同ef值下测试搜索性能，生成recall-latency-visited nodes数据。

        Args:
            queries: 查询向量 [num_queries, dimension]
            ground_truth: 精确最近邻索引 [num_queries, k]
            ef_values: 要测试的ef值列表
            k: 返回最近邻数量

        Returns:
            每个ef值对应的性能指标
        """
        if ef_values is None:
            ef_values = [16, 32, 64, 128, 256, 512]

        results = {}
        for ef in ef_values:
            stats = self.search_with_stats(queries, k=k, ef_search=ef)

            # 计算召回率
            recall = 0.0
            indices = stats['indices']
            for i in range(len(queries)):
                pred_set = set(indices[i].tolist())
                gt_set = set(ground_truth[i, :k].tolist())
                recall += len(pred_set & gt_set) / k
            recall /= len(queries)

            results[ef] = {
                'recall@k': float(recall),
                'avg_latency_ms': stats['avg_latency_ms'],
                'qps': stats['qps'],
                'avg_visited_nodes': stats['avg_visited_nodes']
            }

        return results

    def compare_search_behavior(
        self,
        queries: np.ndarray,
        ground_truth: np.ndarray,
        ef_search: int = 128,
        k: int = 10
    ) -> Dict[str, Any]:
        """
        详细分析搜索行为

        对每个查询分析搜索结果的距离分布。

        Args:
            queries: 查询向量
            ground_truth: 精确最近邻
            ef_search: 搜索ef值
            k: 返回数量

        Returns:
            搜索行为分析结果
        """
        stats = self.search_with_stats(queries, k=k, ef_search=ef_search)
        indices = stats['indices']
        distances = stats['distances']

        # 逐查询分析
        per_query_stats = []
        for i in range(len(queries)):
            pred_set = set(indices[i].tolist())
            gt_set = set(ground_truth[i, :k].tolist())
            hit_count = len(pred_set & gt_set)

            # 结果中的距离统计
            result_distances = distances[i]
            valid_mask = result_distances < 1e30
            valid_distances = result_distances[valid_mask]

            query_stat = {
                'recall': float(hit_count / k),
                'hit_count': hit_count,
                'mean_result_distance': float(np.mean(valid_distances)) if len(valid_distances) > 0 else 0,
                'min_result_distance': float(np.min(valid_distances)) if len(valid_distances) > 0 else 0,
                'max_result_distance': float(np.max(valid_distances)) if len(valid_distances) > 0 else 0,
            }
            per_query_stats.append(query_stat)

        recalls = [s['recall'] for s in per_query_stats]

        return {
            'overall': {
                'mean_recall': float(np.mean(recalls)),
                'std_recall': float(np.std(recalls)),
                'min_recall': float(np.min(recalls)),
                'max_recall': float(np.max(recalls)),
            },
            'per_query': per_query_stats,
            'search_stats': {
                'avg_latency_ms': stats['avg_latency_ms'],
                'qps': stats['qps'],
                'avg_visited_nodes': stats['avg_visited_nodes']
            }
        }


def plot_search_paths(
    baseline_stats: Dict[str, Any],
    dacl_stats: Dict[str, Any],
    save_path: str = "search_path_comparison.pdf"
) -> None:
    """
    绘制搜索效率对比图

    左图: Recall vs Latency
    右图: Visited Nodes vs ef_search

    Args:
        baseline_stats: Baseline的ef敏感度结果
        dacl_stats: DACL-DR的ef敏感度结果
        save_path: 图片保存路径
    """
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 左图: Recall vs Latency
    ef_vals_b = sorted(baseline_stats.keys())
    ef_vals_d = sorted(dacl_stats.keys())

    latency_b = [baseline_stats[ef]['avg_latency_ms'] for ef in ef_vals_b]
    recall_b = [baseline_stats[ef]['recall@k'] for ef in ef_vals_b]
    latency_d = [dacl_stats[ef]['avg_latency_ms'] for ef in ef_vals_d]
    recall_d = [dacl_stats[ef]['recall@k'] for ef in ef_vals_d]

    ax1.plot(latency_b, recall_b, 'o--', label='Baseline (InfoNCE)', color='blue', linewidth=2)
    ax1.plot(latency_d, recall_d, '^-', label='DACL-DR (Ours)', color='red', linewidth=2)
    ax1.set_xlabel('Latency (ms)', fontsize=12)
    ax1.set_ylabel('Recall@10', fontsize=12)
    ax1.set_title('Recall vs Latency', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # 右图: Visited Nodes vs ef_search
    visited_b = [baseline_stats[ef]['avg_visited_nodes'] for ef in ef_vals_b]
    visited_d = [dacl_stats[ef]['avg_visited_nodes'] for ef in ef_vals_d]

    ax2.plot(ef_vals_b, visited_b, 'o--', label='Baseline (InfoNCE)', color='blue', linewidth=2)
    ax2.plot(ef_vals_d, visited_d, '^-', label='DACL-DR (Ours)', color='red', linewidth=2)
    ax2.set_xlabel('ef_search', fontsize=12)
    ax2.set_ylabel('Avg Visited Nodes', fontsize=12)
    ax2.set_title('Visited Nodes vs ef_search', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"图片已保存至 {save_path}")


if __name__ == "__main__":
    print("HNSW搜索效率分析器 - 需要先训练模型并构建索引后使用")
    print("用法示例:")
    print("  simulator = HNSWSimulator(dimension=768, M=32)")
    print("  simulator.build_index(embeddings)")
    print("  stats = simulator.search_with_stats(queries, k=10)")
    print("  ef_results = simulator.analyze_ef_sensitivity(queries, ground_truth)")