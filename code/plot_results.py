"""
结果可视化脚本

从 evaluate.py 产出的 JSON 文件中读取数据，生成对比图表：
  - Recall-Latency 曲线 (experiments.md 第 8.2 节)
  - Recall-QPS 曲线 (experiments.md 第 8.2 节)
  - efSearch / nprobe 敏感度曲线 (experiments.md 第 8.3 节)

用法：
  python plot_results.py \
    --result_files ./results/dacl-dr_nq.json ./results/dpr_nq.json ./results/ance_nq.json ./results/contriever_nq.json \
    --output_dir ./results/figures

参考：
  - experiments.md 第 8.2 节、第 8.3 节
"""

import argparse
import json
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# 各模型的颜色和标记
MODEL_STYLES = {
    "dacl-dr":    {"color": "#E74C3C", "marker": "o", "label": "DACL-DR (Ours)"},
    "dpr":        {"color": "#3498DB", "marker": "s", "label": "DPR"},
    "ance":       {"color": "#2ECC71", "marker": "^", "label": "ANCE"},
    "contriever": {"color": "#9B59B6", "marker": "D", "label": "Contriever"},
}


def get_args():
    parser = argparse.ArgumentParser(description="Plot evaluation results")
    parser.add_argument("--result_files", type=str, nargs="+", required=True,
                        help="evaluate.py 产出的 JSON 文件列表")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset_label", type=str, default=None,
                        help="图表标题中的数据集名称，默认从 JSON 中读取")
    return parser.parse_args()


def load_results(result_files):
    """加载所有评估结果文件。"""
    all_results = {}
    for fpath in result_files:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        model_type = data["model_type"]
        all_results[model_type] = data
        logger.info("Loaded results: %s (%s, %d queries)",
                    model_type, data["dataset"], data["num_queries"])
    return all_results


def extract_hnsw_data(model_results):
    """从 HNSW 结果中提取 (ef_values, recall_100, latency, qps, ndis)。"""
    hnsw = model_results.get("indexes", {}).get("hnsw", {})
    if not hnsw:
        return None

    efs = []
    recall_100 = []
    latency = []
    qps = []
    ndis = []

    for key in sorted(hnsw.keys(), key=lambda k: int(k.split("_")[1])):
        entry = hnsw[key]
        ef = int(key.split("_")[1])
        efs.append(ef)
        recall_100.append(entry.get("recall", {}).get("100", 0))
        latency.append(entry["latency_ms"])
        qps.append(entry["qps"])
        ndis.append(entry.get("avg_distance_computations", 0))

    return {
        "ef": efs,
        "recall_100": recall_100,
        "latency": latency,
        "qps": qps,
        "ndis": ndis,
    }


def extract_ivf_data(model_results, idx_name):
    """从 IVF/IVF-PQ 结果中提取 (nprobe_values, recall_100, latency, qps)。"""
    ivf = model_results.get("indexes", {}).get(idx_name, {})
    if not ivf:
        return None

    nprobes = []
    recall_100 = []
    latency = []
    qps = []

    for key in sorted(ivf.keys(), key=lambda k: int(k.split("_")[1])):
        entry = ivf[key]
        np_val = int(key.split("_")[1])
        nprobes.append(np_val)
        recall_100.append(entry.get("recall", {}).get("100", 0))
        latency.append(entry["latency_ms"])
        qps.append(entry["qps"])

    return {
        "nprobe": nprobes,
        "recall_100": recall_100,
        "latency": latency,
        "qps": qps,
    }


def plot_recall_latency(all_results, index_type, output_dir, dataset_label):
    """绘制 Recall@100 - Latency 曲线。"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    for model_type, results in all_results.items():
        if index_type == "hnsw":
            data = extract_hnsw_data(results)
        else:
            data = extract_ivf_data(results, index_type)

        if data is None:
            continue

        style = MODEL_STYLES.get(model_type, {"color": "gray", "marker": "x", "label": model_type})
        ax.plot(data["recall_100"], data["latency"],
                color=style["color"], marker=style["marker"],
                label=style["label"], linewidth=1.5, markersize=6)

    ax.set_xlabel("Recall@100", fontsize=12)
    ax.set_ylabel("Latency (ms/query)", fontsize=12)
    ax.set_title("%s Recall-Latency (%s)" % (index_type.upper(), dataset_label), fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "recall_latency_%s_%s.png" % (index_type, dataset_label))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)


def plot_recall_qps(all_results, index_type, output_dir, dataset_label):
    """绘制 Recall@100 - QPS 曲线。"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    for model_type, results in all_results.items():
        if index_type == "hnsw":
            data = extract_hnsw_data(results)
        else:
            data = extract_ivf_data(results, index_type)

        if data is None:
            continue

        style = MODEL_STYLES.get(model_type, {"color": "gray", "marker": "x", "label": model_type})
        ax.plot(data["recall_100"], data["qps"],
                color=style["color"], marker=style["marker"],
                label=style["label"], linewidth=1.5, markersize=6)

    ax.set_xlabel("Recall@100", fontsize=12)
    ax.set_ylabel("QPS", fontsize=12)
    ax.set_title("%s Recall-QPS (%s)" % (index_type.upper(), dataset_label), fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "recall_qps_%s_%s.png" % (index_type, dataset_label))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)


def plot_sensitivity_hnsw(all_results, output_dir, dataset_label):
    """绘制 HNSW efSearch 敏感度曲线。"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    for model_type, results in all_results.items():
        data = extract_hnsw_data(results)
        if data is None:
            continue

        style = MODEL_STYLES.get(model_type, {"color": "gray", "marker": "x", "label": model_type})
        ax.plot(data["ef"], data["recall_100"],
                color=style["color"], marker=style["marker"],
                label=style["label"], linewidth=1.5, markersize=6)

    ax.set_xlabel("efSearch", fontsize=12)
    ax.set_ylabel("Recall@100", fontsize=12)
    ax.set_title("HNSW efSearch Sensitivity (%s)" % dataset_label, fontsize=14)
    ax.set_xscale("log", base=2)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "sensitivity_hnsw_%s.png" % dataset_label)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)


def plot_sensitivity_ivf(all_results, idx_name, output_dir, dataset_label):
    """绘制 IVF/IVF-PQ nprobe 敏感度曲线。"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    for model_type, results in all_results.items():
        data = extract_ivf_data(results, idx_name)
        if data is None:
            continue

        style = MODEL_STYLES.get(model_type, {"color": "gray", "marker": "x", "label": model_type})
        ax.plot(data["nprobe"], data["recall_100"],
                color=style["color"], marker=style["marker"],
                label=style["label"], linewidth=1.5, markersize=6)

    ax.set_xlabel("nprobe", fontsize=12)
    ax.set_ylabel("Recall@100", fontsize=12)
    ax.set_title("%s nprobe Sensitivity (%s)" % (idx_name.upper(), dataset_label), fontsize=14)
    ax.set_xscale("log", base=2)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "sensitivity_%s_%s.png" % (idx_name, dataset_label))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)


def plot_distance_computations_hnsw(all_results, output_dir, dataset_label):
    """绘制 HNSW 距离计算次数对比。"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    for model_type, results in all_results.items():
        data = extract_hnsw_data(results)
        if data is None or not any(data["ndis"]):
            continue

        style = MODEL_STYLES.get(model_type, {"color": "gray", "marker": "x", "label": model_type})
        ax.plot(data["recall_100"], data["ndis"],
                color=style["color"], marker=style["marker"],
                label=style["label"], linewidth=1.5, markersize=6)

    ax.set_xlabel("Recall@100", fontsize=12)
    ax.set_ylabel("Avg Distance Computations", fontsize=12)
    ax.set_title("HNSW Distance Computations (%s)" % dataset_label, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "dist_comp_hnsw_%s.png" % dataset_label)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    all_results = load_results(args.result_files)

    # 从第一个结果文件中推断 dataset label
    first_result = list(all_results.values())[0]
    dataset_label = args.dataset_label or first_result["dataset"]

    # HNSW 图表
    plot_recall_latency(all_results, "hnsw", args.output_dir, dataset_label)
    plot_recall_qps(all_results, "hnsw", args.output_dir, dataset_label)
    plot_sensitivity_hnsw(all_results, args.output_dir, dataset_label)
    plot_distance_computations_hnsw(all_results, args.output_dir, dataset_label)

    # IVF 图表
    plot_recall_latency(all_results, "ivf", args.output_dir, dataset_label)
    plot_recall_qps(all_results, "ivf", args.output_dir, dataset_label)
    plot_sensitivity_ivf(all_results, "ivf", args.output_dir, dataset_label)

    # IVF-PQ 图表
    plot_recall_latency(all_results, "ivf_pq", args.output_dir, dataset_label)
    plot_recall_qps(all_results, "ivf_pq", args.output_dir, dataset_label)
    plot_sensitivity_ivf(all_results, "ivf_pq", args.output_dir, dataset_label)

    logger.info("All plots saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
