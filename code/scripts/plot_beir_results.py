#!/usr/bin/env python3
"""
针对零样本测试产出可视化数据结果，需要读取BEIR summary.json，并生成柱状图，比较整体排名和召回率

"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_FILE = os.path.join(BASE_DIR, "results", "beir", "summary.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "Image")

MODEL_ORDER = [
    ("dpr", "DPR"),
    ("dacl-dr-w0", "DACL-DR (w=0)"),
    ("dacl-dr-w04", "DACL-DR (w=0.4)"),
    ("ance", "ANCE*"),
    ("contriever", "Contriever*"),
]
DATASETS = ["scifact", "nfcorpus", "fiqa", "trec-covid", "fever"]
DATASET_LABELS = ["SciFact", "NFCorpus", "FiQA", "TREC-COVID", "FEVER"]

COLORS = {
    "DPR": "#1f77b4",
    "DACL-DR (w=0)": "#9467bd",
    "DACL-DR (w=0.4)": "#d62728",
    "ANCE*": "#ff7f0e",
    "Contriever*": "#2ca02c",
}

def load_beir_data():
    with open(RESULTS_FILE, "r") as f:
        return json.load(f)


def extract_metric_data(data, metric_key):
    result = {}
    for model_key, model_label in MODEL_ORDER:
        values = []
        for dataset in DATASETS:
            if model_key in data and dataset in data[model_key]:
                values.append(data[model_key][dataset][metric_key] * 100)
            else:
                values.append(0)
        result[model_label] = values
    return result


def plot_beir_comparison(data, metric_key, metric_name, filename):
    metric_data = extract_metric_data(data, metric_key)

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(DATASETS))
    width = 0.15

    for i, (model_key, model_label) in enumerate(MODEL_ORDER):
        offset = (i - 2) * width
        values = metric_data[model_label]
        ax.bar(x + offset, values, width, label=model_label,
               color=COLORS[model_label], edgecolor='black', linewidth=0.5)

    ax.set_xlabel("Dataset", fontsize=13)
    ax.set_ylabel(f"{metric_name} (%)", fontsize=13)
    ax.set_title(f"BEIR Zero-Shot Evaluation: {metric_name}", fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(DATASET_LABELS, fontsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.legend(fontsize=10, frameon=True, fancybox=False, edgecolor="gray",
              loc="upper right", ncol=2)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)

    fig.text(0.99, 0.01, "* Reference baselines (different training data/backbone)",
             ha="right", va="bottom", fontsize=9, style="italic", color="gray")

    plt.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_beir_average(data):
    """
    Create bar chart showing average performance across all datasets.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("BEIR Zero-Shot: Average Performance", fontsize=14, y=0.98)

    models = [label for _, label in MODEL_ORDER]

    ndcg_avgs = []
    for model_key, _ in MODEL_ORDER:
        if model_key in data and "Avg" in data[model_key]:
            ndcg_avgs.append(data[model_key]["Avg"]["NDCG@10"] * 100)
        else:
            ndcg_avgs.append(0)

    x = np.arange(len(models))
    bars1 = ax1.bar(x, ndcg_avgs, color=[COLORS[m] for m in models],
                    edgecolor='black', linewidth=0.5)
    ax1.set_ylabel("NDCG@10 (%)", fontsize=12)
    ax1.set_title("Average NDCG@10", fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=15, ha="right", fontsize=10)
    ax1.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    ax1.set_axisbelow(True)

    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=9)

    recall_avgs = []
    for model_key, _ in MODEL_ORDER:
        if model_key in data and "Avg" in data[model_key]:
            recall_avgs.append(data[model_key]["Avg"]["Recall@100"] * 100)
        else:
            recall_avgs.append(0)

    bars2 = ax2.bar(x, recall_avgs, color=[COLORS[m] for m in models],
                    edgecolor='black', linewidth=0.5)
    ax2.set_ylabel("Recall@100 (%)", fontsize=12)
    ax2.set_title("Average Recall@100", fontsize=13)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=15, ha="right", fontsize=10)
    ax2.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    ax2.set_axisbelow(True)

    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    fig.text(0.99, 0.01, "* Reference baselines (different training data/backbone)",
             ha="right", va="bottom", fontsize=9, style="italic", color="gray")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "beir_average.pdf")
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: {output_path}")



def main():
    print("Loading BEIR summary data ...")
    data = load_beir_data()
    print(f"  Loaded data for {len(data)} models.\n")

    print("Generating BEIR visualization figures:")
    plot_beir_comparison(data, "NDCG@10", "NDCG@10", "beir_ndcg10_comparison.pdf")
    plot_beir_comparison(data, "Recall@100", "Recall@100", "beir_recall100_comparison.pdf")
    plot_beir_average(data)
    print("\nDone. All figures saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()

