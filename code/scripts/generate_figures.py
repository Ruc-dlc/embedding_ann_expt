#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成论文图表脚本

严格按照论文第五章的12张数据驱动图生成：

§5.2 混合权重实验:
  图1: Recall/MRR/NDCG vs w（fig:w_retrieval_curves）

§5.3 表示空间分析:
  图2: Pos Mean / Pos Var vs w（fig:pos_mean_var_curves）
  图3: Alignment / Uniformity vs w（fig:alignment_uniformity_curves）
  图4: Alignment-Uniformity 二维散点（fig:alignment_uniformity_scatter）
  图5: t-SNE 对比 w=0 vs w=0.6（fig:tsne_visualization）

§5.4 ANN搜索效率:
  图6: Recall@100 vs ef_search（fig:recall_vs_ef）
  图7: Latency vs ef_search（fig:latency_vs_ef）
  图8: Recall@100 vs Visited Nodes（fig:recall_vs_visited）

§5.5 消融实验:
  图9: 消融 Pos Mean 柱状图（fig:ablation_pos_mean）
  图10: 消融 Alignment-Uniformity 对比（fig:ablation_align_uni）
  图11: 消融 Recall vs ef_search 曲线（fig:ablation_recall_ef）
  图12: 消融 Visited Nodes 柱状图（fig:ablation_visited）

用法:
    python scripts/generate_figures.py --results_dir results/ --output_dir figures/
"""

import argparse
import logging
import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="生成论文图表")

    parser.add_argument("--results_dir", type=str, default="results",
                        help="实验结果目录")
    parser.add_argument("--output_dir", type=str, default="figures",
                        help="图表保存目录")
    parser.add_argument("--format", type=str, default="pdf",
                        choices=["pdf", "png", "svg"],
                        help="图表输出格式")
    parser.add_argument("--dpi", type=int, default=300,
                        help="图片分辨率（仅对png格式生效）")
    parser.add_argument("--figures", type=str, nargs='*', default=None,
                        help="指定生成的图表编号（如 1 3 7），不指定则生成全部")

    return parser.parse_args()


def load_results(results_dir: str) -> dict:
    """
    加载实验结果

    扫描results目录下所有JSON文件并汇总。
    """
    results = {}
    results_path = Path(results_dir)

    if not results_path.exists():
        logger.warning(f"结果目录不存在: {results_dir}")
        return results

    for result_file in results_path.glob("**/*.json"):
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                key = str(result_file.relative_to(results_path)).replace('.json', '').replace('/', '_')
                results[key] = json.load(f)
        except Exception as e:
            logger.warning(f"加载失败 {result_file}: {e}")

    return results


def aggregate_results(raw: dict) -> dict:
    """
    将分散的实验结果JSON聚合为图表函数期望的结构。

    评估脚本各自输出独立JSON到不同子目录，load_results()将路径转为平面key。
    本函数将这些平面结果重组为各图表函数期望的嵌套结构。

    Key 映射关系:
      ablation_w{w}/w{w}_nq_results.json       → w_retrieval
      ablation_w{w}/w{w}_nq_repr_representation → representation_distribution
      ef_sweep/w{w}_nq_ef_sweep.json            → ef_sensitivity
      tsne/w{w}_tsne_representation.json        → embeddings_baseline / embeddings_dacl
      ablation/ablation_NQ_results.json         → ablation_representation / ablation_ef_sensitivity / ablation_efficiency
    """
    aggregated = dict(raw)

    W_VALUES = ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0']

    # ===== 图1: w_retrieval (Recall/MRR/NDCG vs w) =====
    w_retrieval = {}
    for w in W_VALUES:
        key = f"ablation_w{w}_w{w}_nq_results"
        if key in raw:
            sem = raw[key].get("semantic", {})
            w_retrieval[w] = {
                "recall@10": sem.get("Recall@10", 0),
                "mrr@10": sem.get("MRR@10", 0),
                "ndcg@10": sem.get("NDCG@10", 0),
                "recall@100": sem.get("Recall@100", 0),
            }
    if w_retrieval:
        aggregated["w_retrieval"] = w_retrieval

    # ===== 图2-4: representation_distribution =====
    repr_dist = {}
    for w in W_VALUES:
        key = f"ablation_w{w}_w{w}_nq_repr_representation"
        if key in raw:
            repr_dist[w] = raw[key]
    if repr_dist:
        aggregated["representation_distribution"] = repr_dist

    # ===== 图5: t-SNE embeddings =====
    tsne_map = {
        "tsne_w0.0_tsne_representation": "embeddings_baseline",
        "tsne_w0.6_tsne_representation": "embeddings_dacl",
    }
    for src_key, dst_key in tsne_map.items():
        if src_key in raw and "embeddings" in raw[src_key]:
            aggregated[dst_key] = raw[src_key]

    # ===== 图6-8: ef_sensitivity =====
    ef_sens = {}
    for w in W_VALUES:
        key = f"ef_sweep_w{w}_nq_ef_sweep"
        if key in raw:
            sweep = raw[key].get("results", {})
            renamed = {}
            for ef_val, m in sweep.items():
                renamed[ef_val] = {
                    "recall@100": m.get("recall@100", 0),
                    "latency_ms": m.get("avg_latency_ms", 0),
                    "visited_nodes": m.get("avg_visited_nodes", 0),
                }
            ef_sens[f"w={w}"] = renamed
    if ef_sens:
        aggregated["ef_sensitivity"] = ef_sens

    # ===== 图9-12: ablation results =====
    abl_key = "ablation_ablation_NQ_results"
    if abl_key in raw:
        abl = raw[abl_key]
        models = ['A', 'B', 'C', 'D']

        # 图9-10: ablation_representation
        abl_repr = {}
        for mid in models:
            if mid in abl and "representation" in abl[mid]:
                abl_repr[mid] = abl[mid]["representation"]
        if abl_repr:
            aggregated["ablation_representation"] = abl_repr

        # 图11: ablation_ef_sensitivity
        abl_ef = {}
        for mid in models:
            if mid in abl and "ef_sweep" in abl[mid]:
                abl_ef[mid] = abl[mid]["ef_sweep"]
        if abl_ef:
            aggregated["ablation_ef_sensitivity"] = abl_ef

        # 图12: ablation_efficiency (重命名字段以匹配图表函数)
        abl_eff = {}
        for mid in models:
            if mid in abl and "efficiency" in abl[mid]:
                eff = abl[mid]["efficiency"]
                abl_eff[mid] = {
                    "visited_nodes": eff.get("avg_visited_nodes", 0),
                    "avg_latency_ms": eff.get("avg_latency_ms", 0),
                    "qps": eff.get("qps", 0),
                }
        if abl_eff:
            aggregated["ablation_efficiency"] = abl_eff

    return aggregated


# ============================================================
# 图1: Recall/MRR/NDCG vs w（§5.2 fig:w_retrieval_curves）
# ============================================================
def generate_fig1_w_retrieval_curves(results: dict, output_path: str, fmt: str = "pdf"):
    """
    生成检索效果随混合权重w变化的曲线

    三个子图：Recall@10、MRR@10、NDCG@10 vs w ∈ {0, 0.2, 0.4, 0.6, 0.8, 1.0}
    """
    import matplotlib.pyplot as plt

    logger.info(f"[图1] 生成 Recall/MRR/NDCG vs w -> {output_path}")

    if 'w_retrieval' not in results:
        logger.warning("[图1] 未找到权重消融检索结果，跳过")
        return

    data = results['w_retrieval']
    w_values = sorted([float(k) for k in data.keys()])

    recall_vals = [data[str(w)].get('recall@10', 0) for w in w_values]
    mrr_vals = [data[str(w)].get('mrr@10', 0) for w in w_values]
    ndcg_vals = [data[str(w)].get('ndcg@10', 0) for w in w_values]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    for ax, vals, metric_name, color in [
        (ax1, recall_vals, 'Recall@10', 'tab:blue'),
        (ax2, mrr_vals, 'MRR@10', 'tab:orange'),
        (ax3, ndcg_vals, 'NDCG@10', 'tab:green')
    ]:
        ax.plot(w_values, vals, 'o-', color=color, linewidth=2, markersize=8)
        ax.set_xlabel("Distance Weight (w)", fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(f"{metric_name} vs w", fontsize=14)
        ax.set_xticks(w_values)
        ax.grid(True, alpha=0.3)

        # 标注最优点
        best_idx = int(np.argmax(vals))
        ax.annotate(f'{vals[best_idx]:.3f}',
                    (w_values[best_idx], vals[best_idx]),
                    textcoords="offset points", xytext=(0, 10),
                    ha='center', fontsize=10, fontweight='bold', color=color)

    plt.tight_layout()
    fig.savefig(output_path, format=fmt, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# 图2: Pos Mean / Pos Var vs w（§5.3 fig:pos_mean_var_curves）
# ============================================================
def generate_fig2_pos_mean_var(results: dict, output_path: str, fmt: str = "pdf"):
    """
    生成正例相似度均值和方差随w变化的曲线

    双Y轴：左轴 Pos Mean（↑越好），右轴 Pos Var（↓越好）
    """
    import matplotlib.pyplot as plt

    logger.info(f"[图2] 生成 Pos Mean/Var vs w -> {output_path}")

    if 'representation_distribution' not in results:
        logger.warning("[图2] 未找到表示分布数据，跳过")
        return

    data = results['representation_distribution']
    w_values = sorted([float(k) for k in data.keys()])

    pos_means = [data[str(w)].get('pos_mean', 0) for w in w_values]
    pos_vars = [data[str(w)].get('pos_var', 0) for w in w_values]

    fig, ax1 = plt.subplots(figsize=(8, 6))

    color_mean = 'tab:blue'
    ax1.plot(w_values, pos_means, 'o-', color=color_mean, label='Pos Mean', linewidth=2, markersize=8)
    ax1.set_xlabel("Distance Weight (w)", fontsize=12)
    ax1.set_ylabel("Pos Mean (↑)", color=color_mean, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=color_mean)
    ax1.set_xticks(w_values)

    ax2 = ax1.twinx()
    color_var = 'tab:red'
    ax2.plot(w_values, pos_vars, 's--', color=color_var, label='Pos Var', linewidth=2, markersize=8)
    ax2.set_ylabel("Pos Var (↓)", color=color_var, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=color_var)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=11)

    ax1.set_title("Positive Similarity Statistics vs w", fontsize=14)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, format=fmt, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# 图3: Alignment / Uniformity vs w（§5.3 fig:alignment_uniformity_curves）
# ============================================================
def generate_fig3_alignment_uniformity_curves(results: dict, output_path: str, fmt: str = "pdf"):
    """
    生成Alignment和Uniformity随w变化的曲线

    双Y轴：左轴 Alignment（↓越好），右轴 Uniformity（↓越好）
    """
    import matplotlib.pyplot as plt

    logger.info(f"[图3] 生成 Alignment/Uniformity vs w -> {output_path}")

    if 'representation_distribution' not in results:
        logger.warning("[图3] 未找到表示分布数据，跳过")
        return

    data = results['representation_distribution']
    w_values = sorted([float(k) for k in data.keys()])

    alignment_vals = [data[str(w)].get('alignment', 0) for w in w_values]
    uniformity_vals = [data[str(w)].get('uniformity', 0) for w in w_values]

    fig, ax1 = plt.subplots(figsize=(8, 6))

    color_align = 'tab:blue'
    ax1.plot(w_values, alignment_vals, 'o-', color=color_align, label='Alignment', linewidth=2, markersize=8)
    ax1.set_xlabel("Distance Weight (w)", fontsize=12)
    ax1.set_ylabel("Alignment (↓)", color=color_align, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=color_align)
    ax1.set_xticks(w_values)

    ax2 = ax1.twinx()
    color_uni = 'tab:red'
    ax2.plot(w_values, uniformity_vals, 's--', color=color_uni, label='Uniformity', linewidth=2, markersize=8)
    ax2.set_ylabel("Uniformity (↓)", color=color_uni, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=color_uni)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=11)

    ax1.set_title("Alignment-Uniformity vs w", fontsize=14)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, format=fmt, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# 图4: Alignment-Uniformity 散点（§5.3 fig:alignment_uniformity_scatter）
# ============================================================
def generate_fig4_alignment_uniformity_scatter(results: dict, output_path: str, fmt: str = "pdf"):
    """
    生成Alignment-Uniformity二维散点图

    各w配置在A-U空间中的位置，标注帕累托前沿。
    """
    import matplotlib.pyplot as plt

    logger.info(f"[图4] 生成 A-U 散点图 -> {output_path}")

    if 'representation_distribution' not in results:
        logger.warning("[图4] 未找到表示分布数据，跳过")
        return

    data = results['representation_distribution']

    fig, ax = plt.subplots(figsize=(8, 6))

    w_values = sorted([float(k) for k in data.keys()])
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=min(w_values), vmax=max(w_values))

    for w in w_values:
        metrics = data[str(w)]
        uniformity = metrics.get('uniformity', 0)
        alignment = metrics.get('alignment', 0)
        color = cmap(norm(w))
        ax.scatter(uniformity, alignment, s=150, color=color, zorder=5, edgecolors='black', linewidths=0.5)
        ax.annotate(f'w={w}', (uniformity, alignment),
                    textcoords="offset points", xytext=(8, 8), fontsize=10)

    ax.set_xlabel("Uniformity (↓)", fontsize=12)
    ax.set_ylabel("Alignment (↓)", fontsize=12)
    ax.set_title("Alignment-Uniformity Trade-off", fontsize=14)
    ax.grid(True, alpha=0.3)

    # 左下角更好
    ax.annotate("← Better", xy=(0.05, 0.05), xycoords='axes fraction',
                fontsize=10, color='green', alpha=0.7)

    plt.tight_layout()
    fig.savefig(output_path, format=fmt, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# 图5: t-SNE 可视化（§5.3 fig:tsne_visualization）
# ============================================================
def generate_fig5_tsne(results: dict, output_path: str, fmt: str = "pdf"):
    """
    生成t-SNE对比可视化

    左图: w=0 (Baseline)，右图: w=0.6 (DACL-DR)
    """
    from analysis.visualization import Visualizer
    import matplotlib.pyplot as plt

    logger.info(f"[图5] 生成 t-SNE 可视化 -> {output_path}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, tag, title in [
        (ax1, 'baseline', 'Baseline (w=0)'),
        (ax2, 'dacl', 'DACL-DR (w=0.6)')
    ]:
        emb_key = f'embeddings_{tag}'
        if emb_key not in results:
            ax.text(0.5, 0.5, '数据未找到', ha='center', va='center', fontsize=14)
            ax.set_title(title)
            continue

        data = results[emb_key]
        embeddings = np.array(data['embeddings'])
        labels = np.array(data.get('labels', None))

        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
        emb_2d = tsne.fit_transform(embeddings)

        if labels is not None:
            scatter = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap='tab10', alpha=0.6, s=20)
        else:
            ax.scatter(emb_2d[:, 0], emb_2d[:, 1], alpha=0.6, s=20)

        ax.set_title(title, fontsize=14)
        ax.set_xlabel("t-SNE Dim 1", fontsize=11)
        ax.set_ylabel("t-SNE Dim 2", fontsize=11)

    plt.tight_layout()
    fig.savefig(output_path, format=fmt, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# 图6: Recall@100 vs ef_search（§5.4 fig:recall_vs_ef）
# ============================================================
def generate_fig6_recall_vs_ef(results: dict, output_path: str, fmt: str = "pdf"):
    """
    生成Recall@100随ef_search变化的曲线

    对比不同w值下Recall@100随ef_search的变化。
    """
    import matplotlib.pyplot as plt

    logger.info(f"[图6] 生成 Recall vs ef_search -> {output_path}")

    if 'ef_sensitivity' not in results:
        logger.warning("[图6] 未找到ef敏感度数据，跳过")
        return

    data = results['ef_sensitivity']
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
    markers = ['o', 's', 'D', '^', 'v', 'P']

    for idx, (method_name, ef_data) in enumerate(data.items()):
        ef_vals = sorted([int(k) for k in ef_data.keys()])
        recalls = [ef_data[str(ef)]['recall@100'] for ef in ef_vals]

        ax.plot(ef_vals, recalls, f'{markers[idx % len(markers)]}-',
                label=method_name, color=colors[idx % len(colors)],
                linewidth=2, markersize=8)

    ax.set_xlabel("ef_search", fontsize=12)
    ax.set_ylabel("Recall@100", fontsize=12)
    ax.set_title("Recall@100 vs ef_search", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, format=fmt, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# 图7: Latency vs ef_search（§5.4 fig:latency_vs_ef）
# ============================================================
def generate_fig7_latency_vs_ef(results: dict, output_path: str, fmt: str = "pdf"):
    """
    生成Latency随ef_search变化的曲线
    """
    import matplotlib.pyplot as plt

    logger.info(f"[图7] 生成 Latency vs ef_search -> {output_path}")

    if 'ef_sensitivity' not in results:
        logger.warning("[图7] 未找到ef敏感度数据，跳过")
        return

    data = results['ef_sensitivity']
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
    markers = ['o', 's', 'D', '^', 'v', 'P']

    for idx, (method_name, ef_data) in enumerate(data.items()):
        ef_vals = sorted([int(k) for k in ef_data.keys()])
        latencies = [ef_data[str(ef)]['latency_ms'] for ef in ef_vals]

        ax.plot(ef_vals, latencies, f'{markers[idx % len(markers)]}-',
                label=method_name, color=colors[idx % len(colors)],
                linewidth=2, markersize=8)

    ax.set_xlabel("ef_search", fontsize=12)
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title("Latency vs ef_search", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, format=fmt, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# 图8: Recall@100 vs Visited Nodes（§5.4 fig:recall_vs_visited）
# ============================================================
def generate_fig8_recall_vs_visited(results: dict, output_path: str, fmt: str = "pdf"):
    """
    生成Recall@100与Visited Nodes的关系曲线

    这是论文的核心实验结果图，验证命题P1。
    """
    import matplotlib.pyplot as plt

    logger.info(f"[图8] 生成 Recall vs Visited Nodes -> {output_path}")

    if 'ef_sensitivity' not in results:
        logger.warning("[图8] 未找到ef敏感度数据，跳过")
        return

    data = results['ef_sensitivity']
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
    markers = ['o', 's', 'D', '^', 'v', 'P']

    for idx, (method_name, ef_data) in enumerate(data.items()):
        ef_vals = sorted([int(k) for k in ef_data.keys()])
        visited = [ef_data[str(ef)]['visited_nodes'] for ef in ef_vals]
        recalls = [ef_data[str(ef)]['recall@100'] for ef in ef_vals]

        ax.plot(visited, recalls, f'{markers[idx % len(markers)]}-',
                label=method_name, color=colors[idx % len(colors)],
                linewidth=2, markersize=8)

    ax.set_xlabel("Avg Visited Nodes", fontsize=12)
    ax.set_ylabel("Recall@100", fontsize=12)
    ax.set_title("Recall@100 vs Visited Nodes", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, format=fmt, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# 图9: 消融 Pos Mean 柱状图（§5.5 fig:ablation_pos_mean）
# ============================================================
def generate_fig9_ablation_pos_mean(results: dict, output_path: str, fmt: str = "pdf"):
    """
    生成消融实验正样本平均相似度（Pos Mean）对比柱状图

    四组模型: A(Baseline), B(+L_dis), C(+Curriculum), D(Full)
    """
    import matplotlib.pyplot as plt

    logger.info(f"[图9] 生成消融 Pos Mean 柱状图 -> {output_path}")

    if 'ablation_representation' not in results:
        logger.warning("[图9] 未找到消融表示数据，跳过")
        return

    data = results['ablation_representation']
    models = list(data.keys())
    pos_means = [data[m]['pos_mean'] for m in models]

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red'][:len(models)]
    bars = ax.bar(models, pos_means, color=colors)

    for bar, val in zip(bars, pos_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel("Pos Mean (↑)", fontsize=12)
    ax.set_title("Positive Similarity Mean (Ablation)", fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig.savefig(output_path, format=fmt, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# 图10: 消融 Alignment-Uniformity 对比（§5.5 fig:ablation_align_uni）
# ============================================================
def generate_fig10_ablation_align_uni(results: dict, output_path: str, fmt: str = "pdf"):
    """
    生成消融实验Alignment和Uniformity对比图

    左图: Alignment柱状图，右图: Uniformity柱状图
    """
    import matplotlib.pyplot as plt

    logger.info(f"[图10] 生成消融 A-U 对比 -> {output_path}")

    if 'ablation_representation' not in results:
        logger.warning("[图10] 未找到消融表示数据，跳过")
        return

    data = results['ablation_representation']
    models = list(data.keys())
    alignments = [data[m]['alignment'] for m in models]
    uniformities = [data[m]['uniformity'] for m in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red'][:len(models)]

    # 左图: Alignment
    bars1 = ax1.bar(models, alignments, color=colors)
    for bar, val in zip(bars1, alignments):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    ax1.set_ylabel("Alignment (↓)", fontsize=12)
    ax1.set_title("(a) Alignment", fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')

    # 右图: Uniformity
    bars2 = ax2.bar(models, uniformities, color=colors)
    for bar, val in zip(bars2, uniformities):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    ax2.set_ylabel("Uniformity (↓)", fontsize=12)
    ax2.set_title("(b) Uniformity", fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(output_path, format=fmt, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# 图11: 消融 Recall vs ef_search 曲线（§5.5 fig:ablation_recall_ef）
# ============================================================
def generate_fig11_ablation_recall_ef(results: dict, output_path: str, fmt: str = "pdf"):
    """
    生成消融实验四组模型在不同ef_search下的Recall@10曲线
    """
    import matplotlib.pyplot as plt

    logger.info(f"[图11] 生成消融 Recall vs ef -> {output_path}")

    if 'ablation_ef_sensitivity' not in results:
        logger.warning("[图11] 未找到消融ef敏感度数据，跳过")
        return

    data = results['ablation_ef_sensitivity']
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    markers = ['o', 's', 'D', '^']

    for idx, (model_name, ef_data) in enumerate(data.items()):
        ef_vals = sorted([int(k) for k in ef_data.keys()])
        recalls = [ef_data[str(ef)]['recall@10'] for ef in ef_vals]

        ax.plot(ef_vals, recalls, f'{markers[idx % len(markers)]}-',
                label=model_name, color=colors[idx % len(colors)],
                linewidth=2, markersize=8)

    ax.set_xlabel("ef_search", fontsize=12)
    ax.set_ylabel("Recall@10", fontsize=12)
    ax.set_title("Recall@10 vs ef_search (Ablation)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, format=fmt, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# 图12: 消融 Visited Nodes 柱状图（§5.5 fig:ablation_visited）
# ============================================================
def generate_fig12_ablation_visited(results: dict, output_path: str, fmt: str = "pdf"):
    """
    生成消融实验平均访问节点数对比柱状图

    ef_search=100条件下四组模型的Visited Nodes对比。
    """
    import matplotlib.pyplot as plt

    logger.info(f"[图12] 生成消融 Visited Nodes 柱状图 -> {output_path}")

    if 'ablation_efficiency' not in results:
        logger.warning("[图12] 未找到消融效率数据，跳过")
        return

    data = results['ablation_efficiency']
    models = list(data.keys())
    visited = [data[m]['visited_nodes'] for m in models]

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red'][:len(models)]
    bars = ax.bar(models, visited, color=colors)

    for bar, val in zip(bars, visited):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{val:,.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel("Avg Visited Nodes (↓)", fontsize=12)
    ax.set_title("Visited Nodes (Ablation, ef_search=100)", fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig.savefig(output_path, format=fmt, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# 主函数
# ============================================================

# 图表编号到生成函数的映射（严格对应论文第五章）
FIGURE_GENERATORS = {
    '1': ('w_retrieval_curves', generate_fig1_w_retrieval_curves),
    '2': ('pos_mean_var', generate_fig2_pos_mean_var),
    '3': ('alignment_uniformity_curves', generate_fig3_alignment_uniformity_curves),
    '4': ('alignment_uniformity_scatter', generate_fig4_alignment_uniformity_scatter),
    '5': ('tsne_visualization', generate_fig5_tsne),
    '6': ('recall_vs_ef', generate_fig6_recall_vs_ef),
    '7': ('latency_vs_ef', generate_fig7_latency_vs_ef),
    '8': ('recall_vs_visited', generate_fig8_recall_vs_visited),
    '9': ('ablation_pos_mean', generate_fig9_ablation_pos_mean),
    '10': ('ablation_align_uni', generate_fig10_ablation_align_uni),
    '11': ('ablation_recall_ef', generate_fig11_ablation_recall_ef),
    '12': ('ablation_visited', generate_fig12_ablation_visited),
}


def main():
    """主函数"""
    args = parse_args()

    logger.info("=" * 50)
    logger.info("论文图表生成（严格对应第五章12张图）")
    logger.info("=" * 50)

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载结果并聚合
    raw_results = load_results(args.results_dir)
    logger.info(f"已加载原始结果文件: {len(raw_results)} 个")
    results = aggregate_results(raw_results)
    logger.info(f"聚合后可用数据键: {[k for k in results if not k.startswith('ablation_w') and not k.startswith('ef_sweep_') and not k.startswith('bm25_') and not k.startswith('tsne_')]}")

    # 确定要生成的图表
    if args.figures:
        figure_ids = args.figures
    else:
        figure_ids = list(FIGURE_GENERATORS.keys())

    fmt = args.format
    generated = 0
    skipped = 0

    for fig_id in figure_ids:
        if fig_id not in FIGURE_GENERATORS:
            logger.warning(f"未知图表编号: {fig_id}，跳过")
            skipped += 1
            continue

        name, generator = FIGURE_GENERATORS[fig_id]
        output_path = str(output_dir / f"fig{fig_id}_{name}.{fmt}")

        try:
            generator(results, output_path, fmt)
            generated += 1
        except Exception as e:
            logger.error(f"[图{fig_id}] 生成失败: {e}")
            skipped += 1

    logger.info("=" * 50)
    logger.info(f"图表生成完成: 成功 {generated} 张，跳过 {skipped} 张")
    logger.info(f"图表保存至 {output_dir}")


if __name__ == "__main__":
    main()