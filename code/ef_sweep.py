#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HNSW efSearch 参数扫描

对 efSearch in {16, 32, 64, 128, 256, 512} 逐条查询, 收集:
  - Recall@10, Recall@100     (ANN 召回率: HNSW vs Flat 精确搜索)
  - Accuracy@5/10/20/50/100   (Top-K 命中率: answer string matching)
  - Latency, QPS, ndis, nhops (效率指标)
  - avg_pos_sim_top10          (query 与 top-10 正例的平均相似度)

用法:
    python ef_sweep.py \
        --checkpoint experiments/models/w0.6/best_nq \
        --index experiments/indices/w0.6_best_nq \
        --embeddings experiments/embeddings/w0.6_best_nq \
        --test_file data_set/NQ/nq-test.csv \
        --corpus experiments/corpus/corpus_5m.tsv \
        --output experiments/results/nq/w0.6_ef_sweep.json
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
from transformers import AutoTokenizer

from src.models import BiEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

EF_VALUES = [16, 32, 64, 128, 256, 512]
ACCURACY_K_VALUES = [5, 10, 20, 50, 100]


def compute_ann_recall(hnsw_indices: np.ndarray, flat_indices: np.ndarray, k: int) -> float:
    """
    计算 ANN Recall@K

    对每个 query, 计算 |HNSW_topK ∩ Flat_topK| / K, 然后取所有 query 的平均。
    """
    n = hnsw_indices.shape[0]
    recalls = []
    for i in range(n):
        hnsw_set = set(hnsw_indices[i, :k].tolist())
        flat_set = set(flat_indices[i, :k].tolist())
        # 排除无效索引 (-1)
        hnsw_set.discard(-1)
        flat_set.discard(-1)
        if len(flat_set) == 0:
            continue
        recalls.append(len(hnsw_set & flat_set) / len(flat_set))
    return np.mean(recalls).item() if recalls else 0.0


def compute_accuracy_at_k(
    hnsw_indices: np.ndarray,
    doc_ids: List[str],
    answers_list: List[List[str]],
    corpus_texts: dict,
    k_values: List[int],
    answer_in_text_fn,
) -> dict:
    """
    计算 Top-K Accuracy (answer match)

    对每个 query, 检索 top-K passage, 若任一 passage 包含答案则命中。
    """
    results = {}
    for k in k_values:
        hits = 0
        total = 0
        for i, ans_list in enumerate(answers_list):
            if not ans_list:
                continue
            total += 1
            for idx in hnsw_indices[i, :k]:
                if 0 <= idx < len(doc_ids):
                    text = corpus_texts.get(doc_ids[idx], "")
                    if answer_in_text_fn(text, ans_list):
                        hits += 1
                        break
        results[f"accuracy@{k}"] = round(hits / max(total, 1), 4)
    return results


def compute_avg_pos_sim_top10(
    q_emb: np.ndarray,
    hnsw_indices: np.ndarray,
    hnsw_distances: np.ndarray,
    doc_ids: List[str],
    answers_list: List[List[str]],
    corpus_texts: dict,
    answer_in_text_fn,
) -> float:
    """
    计算 query 与 top-10 中命中正例的平均相似度

    对每个 query, 在 top-10 检索结果中找到包含答案的 passage,
    收集这些 passage 对应的相似度分数, 取平均。
    """
    all_sims = []
    for i, ans_list in enumerate(answers_list):
        if not ans_list:
            continue
        for j in range(min(10, hnsw_indices.shape[1])):
            idx = hnsw_indices[i, j]
            if 0 <= idx < len(doc_ids):
                text = corpus_texts.get(doc_ids[idx], "")
                if answer_in_text_fn(text, ans_list):
                    # FAISS IndexHNSWFlat 返回的 distance 是 L2 距离
                    # 但由于我们的向量已经 L2 归一化, 所以 inner product = 1 - L2^2/2
                    # 不过这里直接用 query 与 passage embedding 点积更准确
                    # distances 在 HNSW Flat 中是 L2 距离, 转换: sim = 1 - dist/2 (归一化向量)
                    dist = hnsw_distances[i, j]
                    sim = 1.0 - dist / 2.0
                    all_sims.append(sim)
    return round(float(np.mean(all_sims)), 4) if all_sims else 0.0


def main():
    p = argparse.ArgumentParser(description="efSearch 参数扫描")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--index", type=str, required=True)
    p.add_argument("--embeddings", type=str, required=True, help="嵌入目录 (含 embeddings.npy)")
    p.add_argument("--test_file", type=str, required=True)
    p.add_argument("--corpus", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--ef_values", type=int, nargs="+", default=EF_VALUES)
    p.add_argument("--top_k", type=int, default=100)
    p.add_argument("--max_query_len", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=256)
    args = p.parse_args()

    import faiss

    from evaluate import load_test_csv, load_corpus_texts, load_index, encode_queries, answer_in_text

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    logger.info(f"加载模型: {args.checkpoint}")
    model = BiEncoder.from_pretrained(args.checkpoint)
    model.eval()
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model.model_name)

    # 加载 HNSW 索引和数据
    logger.info("加载 HNSW 索引...")
    index, doc_ids = load_index(args.index)
    queries, answers = load_test_csv(args.test_file)
    corpus_texts = load_corpus_texts(args.corpus)

    # 加载嵌入, 构建 Flat 索引作为 ground truth
    emb_path = Path(args.embeddings) / "embeddings.npy"
    logger.info(f"加载嵌入构建 Flat 索引: {emb_path}")
    embeddings = np.load(emb_path)
    dim = embeddings.shape[1]
    flat_index = faiss.IndexFlatL2(dim)
    flat_index.add(embeddings)
    logger.info(f"Flat 索引: {flat_index.ntotal} 向量, dim={dim}")
    del embeddings  # 释放内存

    # 编码查询
    logger.info("编码查询...")
    q_emb = encode_queries(model, tokenizer, queries, args.max_query_len, args.batch_size, device)

    # Flat 精确搜索 (ground truth)
    logger.info("Flat 精确搜索 top-100 (ground truth)...")
    flat_distances, flat_indices = flat_index.search(q_emb, args.top_k)
    logger.info("Flat 搜索完成")
    del flat_index  # 释放内存

    # 确保单线程, 以获取准确的 hnsw_stats
    prev_threads = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(1)

    sweep_results = {}

    for ef in args.ef_values:
        logger.info(f"\n{'='*60}")
        logger.info(f"efSearch = {ef}")
        logger.info(f"{'='*60}")
        index.hnsw.efSearch = ef

        # --- 逐条查询: 收集延迟 + hnsw_stats ---
        faiss.cvar.hnsw_stats.reset()
        latencies = []

        for i in range(len(q_emb)):
            q = q_emb[i : i + 1]
            t0 = time.perf_counter()
            _, _ = index.search(q, args.top_k)
            latencies.append(time.perf_counter() - t0)

        ndis = faiss.cvar.hnsw_stats.ndis
        nhops = faiss.cvar.hnsw_stats.nhops
        n = len(q_emb)

        avg_latency_ms = float(np.mean(latencies)) * 1000
        qps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0.0
        avg_ndis = ndis / n
        avg_nhops = nhops / n

        # --- 批量搜索: 用于 recall + accuracy 计算 ---
        faiss.cvar.hnsw_stats.reset()
        hnsw_distances, hnsw_indices = index.search(q_emb, args.top_k)

        # ANN Recall@K (HNSW vs Flat)
        recall_10 = compute_ann_recall(hnsw_indices, flat_indices, 10)
        recall_100 = compute_ann_recall(hnsw_indices, flat_indices, 100)

        # Top-K Accuracy (answer match)
        acc = compute_accuracy_at_k(
            hnsw_indices, doc_ids, answers, corpus_texts, ACCURACY_K_VALUES, answer_in_text
        )

        # Query-正例相似度
        avg_pos_sim = compute_avg_pos_sim_top10(
            q_emb, hnsw_indices, hnsw_distances, doc_ids, answers, corpus_texts, answer_in_text
        )

        sweep_results[str(ef)] = {
            "recall@10": round(recall_10, 4),
            "recall@100": round(recall_100, 4),
            **acc,
            "latency_ms": round(avg_latency_ms, 3),
            "qps": round(qps, 1),
            "ndis": round(avg_ndis, 1),
            "nhops": round(avg_nhops, 1),
            "avg_pos_sim_top10": avg_pos_sim,
        }

        logger.info(f"  Recall@10={recall_10:.4f}, Recall@100={recall_100:.4f}")
        for k in ACCURACY_K_VALUES:
            logger.info(f"  Accuracy@{k}={acc[f'accuracy@{k}']:.4f}")
        logger.info(f"  Latency={avg_latency_ms:.3f}ms, QPS={qps:.1f}")
        logger.info(f"  ndis={avg_ndis:.1f}, nhops={avg_nhops:.1f}")
        logger.info(f"  avg_pos_sim_top10={avg_pos_sim:.4f}")

    # 恢复线程数
    faiss.omp_set_num_threads(prev_threads)

    # 保存
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sweep_results, f, indent=2)

    logger.info(f"\n结果已保存: {out_path}")

    # 汇总表格
    logger.info("\n" + "=" * 120)
    header = (
        f"{'efSearch':>8} | {'R@10':>6} | {'R@100':>6} | "
        f"{'A@5':>6} | {'A@10':>6} | {'A@20':>6} | {'A@50':>6} | {'A@100':>6} | "
        f"{'Lat(ms)':>8} | {'QPS':>8} | {'ndis':>8} | {'nhops':>6} | {'PosSim':>7}"
    )
    logger.info(header)
    logger.info("-" * 120)
    for ef in args.ef_values:
        r = sweep_results[str(ef)]
        logger.info(
            f"{ef:>8} | {r['recall@10']:>6.4f} | {r['recall@100']:>6.4f} | "
            f"{r['accuracy@5']:>6.4f} | {r['accuracy@10']:>6.4f} | {r['accuracy@20']:>6.4f} | "
            f"{r['accuracy@50']:>6.4f} | {r['accuracy@100']:>6.4f} | "
            f"{r['latency_ms']:>8.3f} | {r['qps']:>8.1f} | "
            f"{r['ndis']:>8.1f} | {r['nhops']:>6.1f} | {r['avg_pos_sim_top10']:>7.4f}"
        )
    logger.info("=" * 120)


if __name__ == "__main__":
    main()
