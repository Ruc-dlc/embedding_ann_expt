#!/usr/bin/env python3
"""
零样本测试场景下，使用trivia数据集中的trivia-test.csv, 比较HNSW和IVF的性能，需要读取5个模型的结果，生成5个子图

"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results", "trivia")
OUTPUT_DIR = os.path.join(BASE_DIR, "Image")

MODELS = [
    ("dpr_trivia_zero_shot.json",                "DPR"),
    ("ance_trivia.json",                         "ANCE"),
    ("contriever_trivia.json",                   "Contriever"),
    ("dacl_dr_w0.0_trivia_zero_shot.json",       "DACL-DR (w=0)"),
    ("dacl_dr_w0.4_trivia_zero_shot.json",       "DACL-DR (w=0.4)"),
]

STYLES = {
    "DPR":              ("#1f77b4", "o",  1.5,  2),
    "ANCE":             ("#ff7f0e", "s",  1.5,  2),
    "Contriever":       ("#2ca02c", "^",  1.5,  2),
    "DACL-DR (w=0)":    ("#9467bd", "D",  1.5,  2),
    "DACL-DR (w=0.4)":  ("#d62728", "*",  2.8,  5),
}

def load_all_data():
    data = {}
    for fname, label in MODELS:
        path = os.path.join(RESULTS_DIR, fname)
        with open(path, "r") as f:
            data[label] = json.load(f)
    return data


def extract_hnsw(model_data):
    hnsw = model_data["indexes"]["hnsw"]
    ef_values = []
    recalls = []
    latencies = []
    ndcs = []
    for key in sorted(hnsw.keys(), key=lambda k: int(k.split("_")[1])):
        entry = hnsw[key]
        ef = int(key.split("_")[1])
        ef_values.append(ef)
        recalls.append(entry["recall"]["100"])
        latencies.append(entry["latency_ms"])
        ndcs.append(entry["avg_distance_computations"])
    return ef_values, recalls, latencies, ndcs


def extract_ivf(model_data):
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


def setup_ax(ax, xlabel, ylabel, title=None):
    """Apply shared academic style to an axes."""
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    if title:
        ax.set_title(title, fontsize=12, pad=8)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_facecolor("white")


def plot_curve(ax, xs, ys, label):
    """Plot a single model curve with its predefined style."""
    color, marker, lw, zo = STYLES[label]
    ax.plot(xs, ys, color=color, marker=marker, label=label,
            linewidth=lw, markersize=5, zorder=zo)



def plot_trivia_subplots(all_data):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("TriviaQA Zero-Shot: ANN Retrieval Efficiency", fontsize=14, y=0.98)

    ax = axes[0, 0]
    for _, label in MODELS:
        ef, recall, _, _ = extract_hnsw(all_data[label])
        plot_curve(ax, ef, recall, label)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_major_locator(ticker.FixedLocator([8, 16, 32, 64, 128, 256, 512]))
    setup_ax(ax, "ef_search", "Recall@100", "HNSW: Recall vs efSearch")
    ax.legend(fontsize=8, frameon=True, fancybox=False, edgecolor="gray", loc="lower right")

    ax = axes[0, 1]
    for _, label in MODELS:
        _, recall, _, ndc = extract_hnsw(all_data[label])
        plot_curve(ax, ndc, recall, label)
    setup_ax(ax, "Number of Distance Computations (NDC)", "Recall@100", "HNSW: Recall vs NDC")
    ax.legend(fontsize=8, frameon=True, fancybox=False, edgecolor="gray", loc="lower right")

    ax = axes[0, 2]
    for _, label in MODELS:
        _, recall, latency, _ = extract_hnsw(all_data[label])
        plot_curve(ax, latency, recall, label)
    setup_ax(ax, "Latency (ms)", "Recall@100", "HNSW: Recall vs Latency")
    ax.legend(fontsize=8, frameon=True, fancybox=False, edgecolor="gray", loc="lower right")

    ax = axes[1, 0]
    for _, label in MODELS:
        nprobe, recall, _ = extract_ivf(all_data[label])
        plot_curve(ax, nprobe, recall, label)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_major_locator(ticker.FixedLocator([1, 4, 8, 16, 32, 64, 128, 256]))
    setup_ax(ax, "nprobe", "Recall@100", "IVF: Recall vs nprobe")
    ax.legend(fontsize=8, frameon=True, fancybox=False, edgecolor="gray", loc="lower right")

    ax = axes[1, 1]
    for _, label in MODELS:
        _, recall, latency = extract_ivf(all_data[label])
        plot_curve(ax, latency, recall, label)
    setup_ax(ax, "Latency (ms)", "Recall@100", "IVF: Recall vs Latency")
    ax.legend(fontsize=8, frameon=True, fancybox=False, edgecolor="gray", loc="lower right")

    axes[1, 2].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "trivia_ann_curves_subplots.pdf")
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    print("Loading TriviaQA zero-shot experiment data ...")
    all_data = load_all_data()
    print(f"  Loaded {len(all_data)} models.\n")

    print("Generating TriviaQA subplots figure:")
    plot_trivia_subplots(all_data)
    print("\nDone. Figure saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()

