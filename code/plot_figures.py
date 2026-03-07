#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成论文图表

读取 experiments/results/ 下的 ef_sweep JSON, 生成 16 张图:
  - 每个数据集 (NQ, TriviaQA) x 8 种曲线:
    Recall@10  vs {efSearch, ndis, Latency, QPS}
    Recall@100 vs {efSearch, ndis, Latency, QPS}
  - 每张图 5 条曲线 (Wmax = 0.0, 0.4, 0.6, 0.8, 1.0)

用法:
    python plot_figures.py
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

WMAX_VALUES = [0.0, 0.4, 0.6, 0.8, 1.0]
DATASETS = {"nq": "Natural Questions", "trivia": "TriviaQA"}

COLORS = {0.0: "#1f77b4", 0.4: "#ff7f0e", 0.6: "#2ca02c", 0.8: "#d62728", 1.0: "#9467bd"}
MARKERS = {0.0: "o", 0.4: "s", 0.6: "D", 0.8: "^", 1.0: "v"}

# 8 张图的定义: (x_key, x_label, y_key, y_label, filename_suffix)
PLOT_SPECS = [
    ("efsearch",   "efSearch",               "recall@10",  "Recall@10",  "recall10_vs_efsearch"),
    ("efsearch",   "efSearch",               "recall@100", "Recall@100", "recall100_vs_efsearch"),
    ("ndis",       "Distance Computations",  "recall@10",  "Recall@10",  "recall10_vs_ndis"),
    ("ndis",       "Distance Computations",  "recall@100", "Recall@100", "recall100_vs_ndis"),
    ("latency_ms", "Latency (ms)",           "recall@10",  "Recall@10",  "recall10_vs_latency"),
    ("latency_ms", "Latency (ms)",           "recall@100", "Recall@100", "recall100_vs_latency"),
    ("qps",        "QPS (Queries Per Second)","recall@10",  "Recall@10",  "recall10_vs_qps"),
    ("qps",        "QPS (Queries Per Second)","recall@100", "Recall@100", "recall100_vs_qps"),
]


def load_sweep_results(results_dir: str, dataset: str):
    """加载某个数据集下所有 w 值的 ef_sweep 结果"""
    data = {}
    for w in WMAX_VALUES:
        path = Path(results_dir) / dataset / f"w{w}_ef_sweep.json"
        if path.exists():
            with open(path, "r") as f:
                data[w] = json.load(f)
        else:
            logger.warning(f"文件不存在: {path}")
    return data


def setup_style():
    """学术图表样式"""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "legend.fontsize": 10,
        "figure.figsize": (8, 5),
    })


def plot_single(data, x_key, x_label, y_key, y_label, dataset_name, output_path):
    """生成单张图: Y vs X, 5 条曲线"""
    fig, ax = plt.subplots(figsize=(8, 5))

    for w in WMAX_VALUES:
        if w not in data:
            continue
        sweep = data[w]
        efs = sorted([int(e) for e in sweep.keys()])

        if x_key == "efsearch":
            xs = efs
        else:
            xs = [sweep[str(e)][x_key] for e in efs]

        ys = [sweep[str(e)][y_key] for e in efs]

        label = f"$W_{{max}}={w}$" + (" (baseline)" if w == 0.0 else "")
        linestyle = "--" if w == 0.0 else "-"
        ax.plot(xs, ys, marker=MARKERS[w], color=COLORS[w], label=label,
                linestyle=linestyle, markersize=6, linewidth=1.5)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{dataset_name}: {y_label} vs {x_label}")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  已保存: {output_path}")


def main():
    p = argparse.ArgumentParser(description="生成论文图表")
    p.add_argument("--results_dir", type=str, default="experiments/results")
    p.add_argument("--output_dir", type=str, default="experiments/plots")
    p.add_argument("--format", type=str, default="pdf", choices=["pdf", "png"])
    args = p.parse_args()

    setup_style()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = args.format

    for ds_key, ds_name in DATASETS.items():
        logger.info(f"\n处理 {ds_name}...")
        data = load_sweep_results(args.results_dir, ds_key)
        if not data:
            logger.warning(f"  无数据, 跳过")
            continue

        for x_key, x_label, y_key, y_label, suffix in PLOT_SPECS:
            plot_single(
                data, x_key, x_label, y_key, y_label, ds_name,
                out_dir / f"{ds_key}_{suffix}.{ext}",
            )

    logger.info(f"\n所有图表已保存至: {out_dir}")


if __name__ == "__main__":
    main()
