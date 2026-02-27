"""
可视化工具

本模块提供向量分布和实验结果的可视化功能，包括t-SNE降维可视化。

论文章节：第5章 - 实验结果可视化
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path


class Visualizer:
    """
    可视化工具

    提供向量空间和实验结果的可视化功能。

    Args:
        figsize: 默认图片大小
        style: matplotlib样式
    """

    def __init__(
        self,
        figsize: Tuple[int, int] = (10, 8),
        style: str = "seaborn-v0_8-whitegrid"
    ):
        self.figsize = figsize
        self.style = style

    def _setup_plot(self):
        """设置绘图环境"""
        import matplotlib.pyplot as plt
        try:
            plt.style.use(self.style)
        except Exception:
            pass
        return plt

    def plot_tsne(
        self,
        embeddings: np.ndarray,
        labels: Optional[np.ndarray] = None,
        title: str = "t-SNE Visualization",
        save_path: Optional[str] = None,
        perplexity: int = 30,
        n_iter: int = 1000
    ):
        """
        t-SNE可视化

        Args:
            embeddings: 向量数组 [num_samples, dim]
            labels: 标签数组
            title: 图标题
            save_path: 保存路径
            perplexity: t-SNE perplexity参数
            n_iter: 迭代次数

        Returns:
            matplotlib Figure对象
        """
        from sklearn.manifold import TSNE
        plt = self._setup_plot()

        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)

        fig, ax = plt.subplots(figsize=self.figsize)

        if labels is not None:
            scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                               c=labels, cmap='tab10', alpha=0.6)
            plt.colorbar(scatter, ax=ax, label='Label')
        else:
            ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6)

        ax.set_title(title)
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_distance_distribution(
        self,
        pos_distances: np.ndarray,
        neg_distances: np.ndarray,
        title: str = "Cosine Distance Distribution",
        save_path: Optional[str] = None
    ):
        """
        绘制余弦距离分布图

        对比正负样本的余弦距离分布（1 - cos_sim）。

        Args:
            pos_distances: 正样本余弦距离数组
            neg_distances: 负样本余弦距离数组
            title: 图标题
            save_path: 保存路径

        Returns:
            matplotlib Figure对象
        """
        plt = self._setup_plot()
        fig, ax = plt.subplots(figsize=self.figsize)

        ax.hist(pos_distances, bins=50, alpha=0.6, label='Positive', density=True)
        ax.hist(neg_distances, bins=50, alpha=0.6, label='Negative', density=True)

        ax.axvline(np.mean(pos_distances), color='blue', linestyle='--',
                   label=f'Pos Mean: {np.mean(pos_distances):.3f}')
        ax.axvline(np.mean(neg_distances), color='orange', linestyle='--',
                   label=f'Neg Mean: {np.mean(neg_distances):.3f}')

        ax.set_xlabel("Cosine Distance (1 - cos)")
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.legend()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_recall_vs_latency(
        self,
        results: Dict[str, Dict[str, float]],
        title: str = "Recall vs Latency Trade-off",
        save_path: Optional[str] = None
    ):
        """
        绘制召回率-延迟权衡图

        Args:
            results: 各方法的结果字典 {方法名: {latency, recall}}
            title: 图标题
            save_path: 保存路径

        Returns:
            matplotlib Figure对象
        """
        plt = self._setup_plot()
        fig, ax = plt.subplots(figsize=self.figsize)

        for method, metrics in results.items():
            ax.scatter(metrics['latency'], metrics['recall'], s=100, label=method)
            ax.annotate(method, (metrics['latency'], metrics['recall']))

        ax.set_xlabel("Latency (ms)")
        ax.set_ylabel("Recall@10")
        ax.set_title(title)
        ax.legend()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_training_curves(
        self,
        history: Dict[str, List[float]],
        title: str = "Training Curves",
        save_path: Optional[str] = None
    ):
        """
        绘制训练曲线

        Args:
            history: 训练指标历史 {指标名: [值列表]}
            title: 图标题
            save_path: 保存路径

        Returns:
            matplotlib Figure对象
        """
        plt = self._setup_plot()
        num_plots = len(history)
        fig, axes = plt.subplots(num_plots, 1,
                                figsize=(self.figsize[0], 4 * num_plots), sharex=True)

        if num_plots == 1:
            axes = [axes]

        for ax, (name, values) in zip(axes, history.items()):
            ax.plot(values)
            ax.set_ylabel(name)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Step")
        axes[0].set_title(title)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_ablation_results(
        self,
        results: Dict[str, Dict[str, float]],
        metric: str = "recall@10",
        title: str = "Ablation Study Results",
        save_path: Optional[str] = None
    ):
        """
        绘制消融实验结果

        Args:
            results: 各配置的结果字典 {配置名: {指标名: 值}}
            metric: 要展示的指标名
            title: 图标题
            save_path: 保存路径

        Returns:
            matplotlib Figure对象
        """
        plt = self._setup_plot()
        fig, ax = plt.subplots(figsize=self.figsize)

        names = list(results.keys())
        values = [results[name].get(metric, 0) for name in names]

        bars = ax.bar(names, values)
        ax.set_ylabel(metric)
        ax.set_title(title)

        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                   f'{value:.3f}', ha='center', va='bottom')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_pareto_frontier(
        self,
        results: Dict[str, Dict[str, float]],
        x_metric: str = "latency",
        y_metric: str = "recall",
        title: str = "Pareto Frontier",
        save_path: Optional[str] = None
    ):
        """
        绘制帕累托前沿图

        Args:
            results: 各方法的结果字典
            x_metric: X轴指标名
            y_metric: Y轴指标名
            title: 图标题
            save_path: 保存路径

        Returns:
            matplotlib Figure对象
        """
        plt = self._setup_plot()
        fig, ax = plt.subplots(figsize=self.figsize)

        x_vals = []
        y_vals = []
        names = []

        for name, metrics in results.items():
            x_vals.append(metrics.get(x_metric, 0))
            y_vals.append(metrics.get(y_metric, 0))
            names.append(name)

        ax.scatter(x_vals, y_vals, s=100)

        for i, name in enumerate(names):
            ax.annotate(name, (x_vals[i], y_vals[i]),
                       textcoords="offset points", xytext=(5, 5))

        ax.set_xlabel(x_metric)
        ax.set_ylabel(y_metric)
        ax.set_title(title)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_distance_weight_ablation(
        self,
        w_values: List[float],
        recall_values: List[float],
        mrr_values: Optional[List[float]] = None,
        title: str = "Distance Weight Ablation",
        save_path: Optional[str] = None
    ):
        """
        绘制距离权重w消融实验曲线

        专用于论文中w ∈ {0, 0.2, 0.4, 0.6, 0.8, 1.0}的消融分析。

        Args:
            w_values: 距离权重值列表
            recall_values: 对应的Recall@10值列表
            mrr_values: 对应的MRR@10值列表（可选，双Y轴展示）
            title: 图标题
            save_path: 保存路径

        Returns:
            matplotlib Figure对象
        """
        plt = self._setup_plot()
        fig, ax1 = plt.subplots(figsize=self.figsize)

        color_recall = 'tab:blue'
        ax1.plot(w_values, recall_values, 'o-', color=color_recall, label='Recall@10', linewidth=2)
        ax1.set_xlabel("Distance Weight (w)")
        ax1.set_ylabel("Recall@10", color=color_recall)
        ax1.tick_params(axis='y', labelcolor=color_recall)

        if mrr_values is not None:
            ax2 = ax1.twinx()
            color_mrr = 'tab:red'
            ax2.plot(w_values, mrr_values, 's--', color=color_mrr, label='MRR@10', linewidth=2)
            ax2.set_ylabel("MRR@10", color=color_mrr)
            ax2.tick_params(axis='y', labelcolor=color_mrr)

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')
        else:
            ax1.legend()

        ax1.set_title(title)
        ax1.set_xticks(w_values)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_three_stage_training(
        self,
        history: Dict[str, List[float]],
        stage_boundaries: List[int],
        title: str = "Three-Stage Training",
        save_path: Optional[str] = None
    ):
        """
        绘制三阶段训练过程图

        展示训练损失随步数变化，标注阶段切换边界。

        Args:
            history: 训练指标历史 {指标名: [值列表]}
            stage_boundaries: 阶段切换的步数列表 [stage1_end, stage2_end]
            title: 图标题
            save_path: 保存路径

        Returns:
            matplotlib Figure对象
        """
        plt = self._setup_plot()
        fig, ax = plt.subplots(figsize=self.figsize)

        colors = ['tab:blue', 'tab:orange', 'tab:green']
        for idx, (name, values) in enumerate(history.items()):
            ax.plot(values, label=name, color=colors[idx % len(colors)], alpha=0.8)

        stage_names = ["Stage 1: Warmup", "Stage 2: Distance Intro", "Stage 3: Joint Opt"]
        for i, boundary in enumerate(stage_boundaries):
            ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.7)
            ax.text(boundary, ax.get_ylim()[1] * 0.95, stage_names[i],
                   rotation=90, va='top', ha='right', fontsize=9, color='gray')

        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.legend()
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig