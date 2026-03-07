#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
t-SNE 向量可视化

从 dev set 采样 query + positive passage, 用各 checkpoint 编码后 t-SNE 降维,
可视化 query/doc 分布, 对比不同 Wmax 下的紧致度与均匀性。

用法:
    # 单个模型
    python visualize_tsne.py \
        --checkpoint experiments/models/w0.6/best_nq \
        --data_dir data_set \
        --output experiments/plots/tsne_w0.6_best_nq.pdf

    # 对比所有模型 (自动模式)
    python visualize_tsne.py --compare_all --output_dir experiments/plots
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
from transformers import AutoTokenizer
from sklearn.manifold import TSNE

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models import BiEncoder
from src.dataset import DPRDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

WMAX_VALUES = [0.0, 0.4, 0.6, 0.8, 1.0]


def sample_pairs(data_dir: str, n_samples: int = 500, seed: int = 42) -> List[Dict]:
    """从 NQ dev 采样 query + positive passage 对"""
    dataset = DPRDataset(f"{data_dir}/NQ/nq-dev.json")
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    indices = indices[:n_samples]

    pairs = []
    for idx in indices:
        sample = dataset[idx]
        pairs.append({
            "query": sample["query"],
            "positive": sample["positive"],
        })
    logger.info(f"采样 {len(pairs)} 个 query-positive 对")
    return pairs


@torch.no_grad()
def encode_texts(model, tokenizer, texts: List[str], max_len: int, device, encode_fn) -> np.ndarray:
    """批量编码文本"""
    model.eval()
    all_emb = []
    batch_size = 64
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch, max_length=max_len, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.float16):
            emb = encode_fn(input_ids, attention_mask)
        all_emb.append(emb.cpu().numpy())
    return np.concatenate(all_emb, axis=0).astype(np.float32)


def plot_tsne_single(
    q_emb: np.ndarray,
    d_emb: np.ndarray,
    title: str,
    output_path: str,
    perplexity: int = 30,
    seed: int = 42,
):
    """单个模型的 t-SNE 可视化"""
    n = q_emb.shape[0]
    combined = np.concatenate([q_emb, d_emb], axis=0)  # [2N, dim]

    logger.info(f"  t-SNE 降维 ({combined.shape[0]} 向量)...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed, n_iter=1000)
    coords = tsne.fit_transform(combined)

    q_coords = coords[:n]
    d_coords = coords[n:]

    # 计算统计量
    # 对齐度: query-positive 对之间的欧氏距离均值
    pair_dists = np.linalg.norm(q_coords - d_coords, axis=1)
    avg_pair_dist = pair_dists.mean()

    # 均匀性: 所有点的标准差 (越大越分散)
    all_std = coords.std(axis=0).mean()

    fig, ax = plt.subplots(figsize=(8, 8))

    # 画连线 (query → positive), 浅灰色
    for i in range(n):
        ax.plot(
            [q_coords[i, 0], d_coords[i, 0]],
            [q_coords[i, 1], d_coords[i, 1]],
            color="#cccccc", linewidth=0.3, alpha=0.3, zorder=1
        )

    ax.scatter(d_coords[:, 0], d_coords[:, 1], c="#2ca02c", s=10, alpha=0.5,
               label="Positive Passage", zorder=2)
    ax.scatter(q_coords[:, 0], q_coords[:, 1], c="#1f77b4", s=10, alpha=0.5,
               label="Query", zorder=3)

    ax.set_title(f"{title}\n(avg pair dist={avg_pair_dist:.2f}, spread={all_std:.2f})")
    ax.legend(loc="upper right")
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  已保存: {output_path}")

    return {"avg_pair_dist": round(float(avg_pair_dist), 4), "spread": round(float(all_std), 4)}


def plot_comparison(all_stats: dict, output_path: str):
    """对比不同 Wmax 的紧致度和均匀性"""
    if not all_stats:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    wmax_vals = sorted(all_stats.keys())
    pair_dists = [all_stats[w]["avg_pair_dist"] for w in wmax_vals]
    spreads = [all_stats[w]["spread"] for w in wmax_vals]

    labels = [f"$W_{{max}}$={w}" for w in wmax_vals]

    ax1.bar(labels, pair_dists, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"])
    ax1.set_ylabel("Avg Query-Positive Distance")
    ax1.set_title("Alignment Compactness (lower = tighter)")

    ax2.bar(labels, spreads, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"])
    ax2.set_ylabel("Embedding Spread (std)")
    ax2.set_title("Distribution Uniformity (higher = more uniform)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"对比图已保存: {output_path}")


def main():
    p = argparse.ArgumentParser(description="t-SNE 向量可视化")
    p.add_argument("--checkpoint", type=str, default=None, help="单个模型 checkpoint")
    p.add_argument("--data_dir", type=str, default="data_set")
    p.add_argument("--output", type=str, default=None, help="单个模型输出路径")
    p.add_argument("--compare_all", action="store_true", help="对比所有 Wmax 模型")
    p.add_argument("--output_dir", type=str, default="experiments/plots")
    p.add_argument("--n_samples", type=int, default=500)
    p.add_argument("--ckpt_type", type=str, default="best_nq", help="best_nq 或 best_trivia")
    p.add_argument("--format", type=str, default="pdf", choices=["pdf", "png"])
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 采样
    pairs = sample_pairs(args.data_dir, n_samples=args.n_samples)
    queries = [p["query"] for p in pairs]
    positives = [p["positive"] for p in pairs]

    if args.compare_all:
        # 对比所有 Wmax 模型
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ext = args.format
        all_stats = {}

        for w in WMAX_VALUES:
            ckpt_path = f"experiments/models/w{w}/{args.ckpt_type}"
            if not Path(ckpt_path).exists():
                logger.warning(f"Checkpoint 不存在: {ckpt_path}, 跳过")
                continue

            logger.info(f"\n处理 Wmax={w} ({ckpt_path})...")
            model = BiEncoder.from_pretrained(ckpt_path)
            model.eval()
            model.to(device)
            tokenizer = AutoTokenizer.from_pretrained(model.model_name)

            q_emb = encode_texts(model, tokenizer, queries, 64, device, model.encode_query)
            d_emb = encode_texts(model, tokenizer, positives, 256, device, model.encode_document)

            stats = plot_tsne_single(
                q_emb, d_emb,
                f"$W_{{max}}$={w}",
                str(out_dir / f"tsne_w{w}_{args.ckpt_type}.{ext}"),
            )
            all_stats[w] = stats

            # 释放
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 对比柱状图
        plot_comparison(all_stats, str(out_dir / f"tsne_comparison_{args.ckpt_type}.{ext}"))

        # 保存统计数据
        stats_path = out_dir / f"tsne_stats_{args.ckpt_type}.json"
        with open(stats_path, "w") as f:
            json.dump({str(k): v for k, v in all_stats.items()}, f, indent=2)
        logger.info(f"统计数据已保存: {stats_path}")

    elif args.checkpoint and args.output:
        # 单个模型
        logger.info(f"加载模型: {args.checkpoint}")
        model = BiEncoder.from_pretrained(args.checkpoint)
        model.eval()
        model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(model.model_name)

        q_emb = encode_texts(model, tokenizer, queries, 64, device, model.encode_query)
        d_emb = encode_texts(model, tokenizer, positives, 256, device, model.encode_document)

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plot_tsne_single(q_emb, d_emb, args.checkpoint, str(out_path))

    else:
        p.print_help()


if __name__ == "__main__":
    main()
