#!/usr/bin/env python3
"""
嵌入空间对比可视化

需要读取由前置脚本 analyse_embeddings.py 生成的 t-SNE .npz 文件和统计信息 .json 文件
输出文件（保存为 PDF 到 Image/ 目录）：

tsne_comparison.pdf —— 2×3 子图布局的 t-SNE 散点图
cosine_distribution.pdf —— 正样本余弦相似度分布的叠加直方图

Usage:
    python scripts/plot_embedding_compare.py \
        --npz_dir ./results/embedding/figures \
        --stats_dir ./results/embedding \
        --output_dir ./Image
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# 颜色与图例定义
MODELS = [
    ("DPR",           "#1f77b4"),
    ("DACL-DR_w0",    "#9467bd"),
    ("DACL-DR_w0.4",   "#d62728"),
    ("ANCE",          "#ff7f0e"),
    ("Contriever",    "#2ca02c"),
]

DISPLAY_LABELS = {
    "DPR": "DPR",
    "DACL-DR_w0": "DACL-DR (w=0)",
    "DACL-DR_w0.4": "DACL-DR (w=0.4)",
    "ANCE": "ANCE",
    "Contriever": "Contriever",
}


def find_npz(npz_dir, label):
    """Find the .npz file for a given model label."""
    pattern = os.path.join(npz_dir, "tsne_data_%s.npz" % label)
    if os.path.exists(pattern):
        return pattern
    candidates = glob.glob(os.path.join(npz_dir, "tsne_data_*%s*.npz" % label))
    if candidates:
        return candidates[0]
    return None


def find_stats(stats_dir, label):
    """Find the stats JSON for a given model label."""
    pattern = os.path.join(stats_dir, "stats_%s.json" % label.lower().replace("-", "_").replace(" ", "_"))
    if os.path.exists(pattern):
        return pattern
    candidates = glob.glob(os.path.join(stats_dir, "stats_*%s*.json" % label.lower().replace("-", "_")))
    if candidates:
        return candidates[0]
    return None


def plot_tsne_subplots(npz_dir, output_dir):
    """Generate a 2x3 grid of t-SNE scatter plots."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes_flat = axes.flatten()

    plotted = 0
    for i, (label, color) in enumerate(MODELS):
        npz_path = find_npz(npz_dir, label)
        if npz_path is None:
            print("  [SKIP] No .npz found for %s" % label)
            continue

        data = np.load(npz_path)
        q_coords = data["q_coords"]
        d_coords = data["d_coords"]
        n_samples = q_coords.shape[0]

        ax = axes_flat[plotted]

        for j in range(n_samples):
            ax.plot([q_coords[j, 0], d_coords[j, 0]],
                    [q_coords[j, 1], d_coords[j, 1]],
                    color="#CCCCCC", linewidth=0.3, alpha=0.4, zorder=1)

        ax.scatter(q_coords[:, 0], q_coords[:, 1],
                   c="#4A90D9", marker="o", s=12, alpha=0.6, label="Query", zorder=3)
        ax.scatter(d_coords[:, 0], d_coords[:, 1],
                   c="#E74C3C", marker="^", s=12, alpha=0.6, label="Document", zorder=3)

        # 计算查询与正样本对之间的平均距离
        pair_dists = np.sqrt(np.sum((q_coords - d_coords) ** 2, axis=1))
        mean_dist = np.mean(pair_dists)

        display = DISPLAY_LABELS.get(label, label)
        ax.set_title("%s (mean dist=%.1f)" % (display, mean_dist), fontsize=13, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        if plotted == 0:
            ax.legend(fontsize=9, loc="upper left")

        plotted += 1

    for j in range(plotted, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "tsne_comparison.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: %s" % out_path)


def plot_cosine_distribution(npz_dir, output_dir):
    """Overlay positive cosine similarity distributions for all models."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for label, color in MODELS:
        npz_path = find_npz(npz_dir, label)
        if npz_path is None:
            continue

        data = np.load(npz_path)
        pos_cos = data["pos_cos"]
        display = DISPLAY_LABELS.get(label, label)
        mean_val = np.mean(pos_cos)

        ax.hist(pos_cos, bins=50, color=color, alpha=0.35, edgecolor="none",
                label="%s (mean=%.3f)" % (display, mean_val))

    ax.set_xlabel("Cosine Similarity (q, d+)", fontsize=13)
    ax.set_ylabel("Count", fontsize=13)
    ax.tick_params(axis="both", labelsize=11)
    ax.legend(fontsize=10, frameon=True, fancybox=False, edgecolor="gray")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "cosine_distribution.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: %s" % out_path)


def main():
    parser = argparse.ArgumentParser(description="Combine embedding analysis into thesis figures")
    parser.add_argument("--npz_dir", type=str, default="./results/embedding/figures",
                        help="Directory containing .npz files from t-SNE runs")
    parser.add_argument("--stats_dir", type=str, default="./results/embedding",
                        help="Directory containing stats JSON files")
    parser.add_argument("--output_dir", type=str, default="./Image",
                        help="Output directory for PDF figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Generating t-SNE comparison subplot ...")
    plot_tsne_subplots(args.npz_dir, args.output_dir)

    print("Generating cosine distribution overlay ...")
    plot_cosine_distribution(args.npz_dir, args.output_dir)

    print("\nDone. Figures saved to:", args.output_dir)


if __name__ == "__main__":
    main()

