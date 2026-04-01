#!/usr/bin/env python3
"""
Generate ANN retrieval efficiency comparison curves for dense retrieval experiments.
针对稠密检索实验产出ANN效率比较曲线图
需要读取5个模型在NQ测试集上的JSON形式的实验结果文件，产出6张PDF格式的图，用于分别比较HNSW和IVF索引性能。

"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results", "nq")
OUTPUT_DIR = os.path.join(BASE_DIR, "Image")

MODELS = [
    ("dpr_nq.json",           "DPR"),
    ("ance_nq.json",          "ANCE"),
    ("contriever_nq.json",    "Contriever"),
    ("dacl_dr_w0.0_nq.json",  "DACL-DR (w=0)"),
    ("dacl_dr_w0.4_nq.json",  "DACL-DR (w=0.4)"),
]

STYLES = {
    "DPR":              ("#1f77b4", "o",  1.5,  2),
    "ANCE":             ("#ff7f0e", "s",  1.5,  2),
    "Contriever":       ("#2ca02c", "^",  1.5,  2),
    "DACL-DR (w=0)":    ("#9467bd", "D",  1.5,  2),
    "DACL-DR (w=0.4)":  ("#d62728", "*",  2.8,  5),
}

# ---------------------------------------------------------------------------
# 加载数据
# ---------------------------------------------------------------------------

def load_all_data():
    """Return dict  label -> parsed JSON."""
    data = {}
    for fname, label in MODELS:
        path = os.path.join(RESULTS_DIR, fname)
        with open(path, "r") as f:
            data[label] = json.load(f)
    return data


def extract_hnsw(model_data):
    """Extract HNSW curves: ef_search, recall@100, latency, ndc, qps."""
    hnsw = model_data["indexes"]["hnsw"]
    ef_values = []
    recalls = []
    latencies = []
    ndcs = []
    qps_values = []
    for key in sorted(hnsw.keys(), key=lambda k: int(k.split("_")[1])):
        entry = hnsw[key]
        ef = int(key.split("_")[1])
        ef_values.append(ef)
        recalls.append(entry["recall"]["100"])
        latencies.append(entry["latency_ms"])
        ndcs.append(entry["avg_distance_computations"])
        qps_values.append(entry["qps"])
    return ef_values, recalls, latencies, ndcs, qps_values


def extract_ivf(model_data):
    """Extract IVF curves: nprobe, recall@100, latency."""
    ivf = model_data["indexes"]["ivf"]
    nprobe_values = []
    recalls = []
    latencies = []
    for key in sorted(ivf.keys(), key=lambda k: int(k.split("_")[1])):
        entry = ivf[key]
        nprobe = int(key.split("_")[1])
        nprobe_values.append(nprobe)
        recalls.append(entry["recall"]["100"])
        latencies.append(entry["latency_ms"])
    return nprobe_values, recalls, latencies

# ---------------------------------------------------------------------------
# 图片帮助输出
# ---------------------------------------------------------------------------

def setup_ax(ax, xlabel, ylabel, title=None):
    """Apply shared academic style to an axes."""
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    if title:
        ax.set_title(title, fontsize=14, pad=10)
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_facecolor("white")


def add_legend(ax):
    """Place legend outside the data area."""
    ax.legend(fontsize=10, frameon=True, fancybox=False, edgecolor="gray",
              loc="lower right")


def plot_curve(ax, xs, ys, label):
    """Plot a single model curve with its predefined style."""
    color, marker, lw, zo = STYLES[label]
    ax.plot(xs, ys, color=color, marker=marker, label=label,
            linewidth=lw, markersize=6, zorder=zo)


def savefig(fig, name):
    """Save figure as PDF."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")

# ---------------------------------------------------------------------------
# 单独的作图脚本
# ---------------------------------------------------------------------------

def plot_hnsw_recall_efsearch(all_data):
    """Fig 1: HNSW Recall@100 vs ef_search."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for _, label in MODELS:
        ef, recall, _, _, _ = extract_hnsw(all_data[label])
        plot_curve(ax, ef, recall, label)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_major_locator(ticker.FixedLocator([8, 16, 32, 64, 128, 256, 512]))
    setup_ax(ax, "ef_search", "Recall@100")
    add_legend(ax)
    savefig(fig, "hnsw_recall_efsearch.pdf")


def plot_hnsw_recall_latency(all_data):
    """Fig 2: HNSW Recall@100 vs Latency."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for _, label in MODELS:
        _, recall, latency, _, _ = extract_hnsw(all_data[label])
        plot_curve(ax, latency, recall, label)
    setup_ax(ax, "Latency (ms)", "Recall@100")
    add_legend(ax)
    savefig(fig, "hnsw_recall_latency.pdf")


def plot_hnsw_recall_ndc(all_data):
    """Fig 3: HNSW Recall@100 vs NDC."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for _, label in MODELS:
        _, recall, _, ndc, _ = extract_hnsw(all_data[label])
        plot_curve(ax, ndc, recall, label)
    setup_ax(ax, "Number of Distance Computations (NDC)", "Recall@100")
    add_legend(ax)
    savefig(fig, "hnsw_recall_ndc.pdf")


def plot_ivf_recall_nprobe(all_data):
    """Fig 4: IVF Recall@100 vs nprobe."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for _, label in MODELS:
        nprobe, recall, _ = extract_ivf(all_data[label])
        plot_curve(ax, nprobe, recall, label)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_major_locator(ticker.FixedLocator([1, 4, 8, 16, 32, 64, 128, 256]))
    setup_ax(ax, "nprobe", "Recall@100")
    add_legend(ax)
    savefig(fig, "ivf_recall_nprobe.pdf")


def plot_ivf_recall_latency(all_data):
    """Fig 5: IVF Recall@100 vs Latency."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for _, label in MODELS:
        _, recall, latency = extract_ivf(all_data[label])
        plot_curve(ax, latency, recall, label)
    setup_ax(ax, "Latency (ms)", "Recall@100")
    add_legend(ax)
    savefig(fig, "ivf_recall_latency.pdf")


def plot_hnsw_recall_qps(all_data):
    """Fig 6 (bonus): HNSW Recall@100 vs QPS."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for _, label in MODELS:
        _, recall, _, _, qps = extract_hnsw(all_data[label])
        plot_curve(ax, qps, recall, label)
    ax.invert_xaxis()  # higher QPS is better, show left-to-right improvement
    setup_ax(ax, "Queries per Second (QPS)", "Recall@100")
    add_legend(ax)
    savefig(fig, "hnsw_recall_qps.pdf")

def main():
    print("Loading experiment data ...")
    all_data = load_all_data()
    print(f"  Loaded {len(all_data)} models.\n")

    print("Generating figures:")
    plot_hnsw_recall_efsearch(all_data)
    plot_hnsw_recall_latency(all_data)
    plot_hnsw_recall_ndc(all_data)
    plot_ivf_recall_nprobe(all_data)
    plot_ivf_recall_latency(all_data)
    plot_hnsw_recall_qps(all_data)
    print("\nDone. All figures saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()

