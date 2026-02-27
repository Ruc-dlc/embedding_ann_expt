#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检索评估脚本

使用训练好的编码器和构建的索引，在测试集上进行检索评估。
支持语义指标（Recall@K, MRR@K, NDCG@K）和效率指标（QPS, 延迟）。

使用方法:
    python scripts/run_evaluation.py \
        --encoder_path checkpoints/distance_aware/final_model \
        --index_path indices/hnsw_index \
        --test_file data_set/NQ/nq-test.csv \
        --output_path results/

论文章节：第5章 5.2节 - 实验评估
"""

import argparse
import csv
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="运行检索评估")

    parser.add_argument("--encoder_path", type=str, required=True,
                        help="训练好的BiEncoder模型目录")
    parser.add_argument("--index_path", type=str, required=True,
                        help="FAISS索引目录")
    parser.add_argument("--test_file", type=str, required=True,
                        help="测试集文件路径（CSV或JSON格式）")
    parser.add_argument("--output_path", type=str, required=True,
                        help="结果输出目录")
    parser.add_argument("--k_values", type=int, nargs='+',
                        default=[1, 5, 10, 20, 50, 100],
                        help="评估的K值列表")
    parser.add_argument("--hnsw_ef_search", type=int, default=128,
                        help="HNSW搜索时的ef参数")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="查询编码批次大小")
    parser.add_argument("--max_query_length", type=int, default=64,
                        help="查询最大token长度")
    parser.add_argument("--experiment_name", type=str, default="eval",
                        help="实验名称（用于输出文件名）")

    return parser.parse_args()


def load_encoder(encoder_path: str):
    """加载BiEncoder模型"""
    from src.models.bi_encoder import BiEncoder

    logger.info(f"加载编码器: {encoder_path}")
    model = BiEncoder.from_pretrained(encoder_path)
    model.eval()
    return model, model.model_name


def load_faiss_index(index_path: str):
    """加载FAISS索引和文档ID映射"""
    import faiss

    index_dir = Path(index_path)

    # 加载索引
    index_file = index_dir / "index.faiss"
    index = faiss.read_index(str(index_file))
    logger.info(f"FAISS索引已加载，包含 {index.ntotal} 个向量")

    # 加载文档ID
    doc_ids_file = index_dir / "doc_ids.txt"
    doc_ids = []
    with open(doc_ids_file, 'r', encoding='utf-8') as f:
        for line in f:
            doc_ids.append(line.strip())
    logger.info(f"文档ID映射已加载，共 {len(doc_ids)} 条")

    return index, doc_ids


def load_test_data(test_file: str):
    """
    加载测试数据

    支持两种格式：
    - CSV: question, answers（answers用制表符分隔）
    - JSON: DPR格式（含positive_ctxs）

    返回:
        queries: 查询文本列表
        answers: 答案列表（每个查询对应多个答案）
        positive_doc_ids: 正例文档ID列表（如果有）
    """
    path = Path(test_file)
    logger.info(f"加载测试数据: {path}")

    queries = []
    answers = []
    positive_doc_ids = []

    if path.suffix == '.csv':
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if len(row) >= 2:
                    queries.append(row[0])
                    answers.append(row[1].split('\t') if '\t' in row[1] else [row[1]])
                    positive_doc_ids.append([])
    elif path.suffix == '.json':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for item in data:
            queries.append(item.get('question', ''))
            answers.append(item.get('answers', []))
            pos_ids = []
            for ctx in item.get('positive_ctxs', []):
                pid = ctx.get('passage_id', ctx.get('psg_id', ''))
                if pid:
                    pos_ids.append(str(pid))
            positive_doc_ids.append(pos_ids)
    else:
        raise ValueError(f"不支持的测试文件格式: {path.suffix}")

    logger.info(f"测试数据加载完成，共 {len(queries)} 条查询")
    return queries, answers, positive_doc_ids


@torch.no_grad()
def encode_queries(encoder, tokenizer, queries, batch_size=64, max_query_length=64):
    """批量编码查询"""
    device = next(encoder.parameters()).device
    encoder.eval()

    all_embeddings = []

    for start_idx in range(0, len(queries), batch_size):
        end_idx = min(start_idx + batch_size, len(queries))
        batch = queries[start_idx:end_idx]

        encoded = tokenizer(
            batch,
            max_length=max_query_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        emb = encoder.encode_query(input_ids, attention_mask)
        all_embeddings.append(emb.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def search_index(index, query_embeddings, k, hnsw_ef_search=128):
    """在FAISS索引中检索，同时收集Visited Nodes统计"""
    import faiss

    # 设置HNSW搜索参数
    if hasattr(index, 'hnsw'):
        index.hnsw.efSearch = hnsw_ef_search

    query_embeddings = query_embeddings.astype(np.float32)
    num_queries = len(query_embeddings)

    # 重置HNSW搜索统计计数器
    if hasattr(faiss, 'cvar') and hasattr(faiss.cvar, 'hnsw_stats'):
        faiss.cvar.hnsw_stats.reset()

    start_time = time.time()
    scores, indices = index.search(query_embeddings, k)
    search_time = time.time() - start_time

    qps = num_queries / search_time if search_time > 0 else 0
    avg_latency = search_time / num_queries * 1000  # ms

    # 收集Visited Nodes（HNSW距离计算次数 ≈ 访问节点数）
    avg_visited_nodes = 0
    if hasattr(faiss, 'cvar') and hasattr(faiss.cvar, 'hnsw_stats'):
        total_ndis = faiss.cvar.hnsw_stats.ndis
        avg_visited_nodes = total_ndis / num_queries if num_queries > 0 else 0

    return scores, indices, qps, avg_latency, avg_visited_nodes


def compute_recall_at_k(retrieved_ids: List[List[str]], positive_ids: List[List[str]], k: int) -> float:
    """计算 Recall@K"""
    if not positive_ids:
        return 0.0

    hits = 0
    total = 0

    for retrieved, positives in zip(retrieved_ids, positive_ids):
        if not positives:
            continue
        total += 1
        top_k = set(retrieved[:k])
        if top_k & set(positives):
            hits += 1

    return hits / max(total, 1)


def compute_mrr_at_k(retrieved_ids: List[List[str]], positive_ids: List[List[str]], k: int) -> float:
    """计算 MRR@K"""
    if not positive_ids:
        return 0.0

    rr_sum = 0.0
    total = 0

    for retrieved, positives in zip(retrieved_ids, positive_ids):
        if not positives:
            continue
        total += 1
        pos_set = set(positives)
        for rank, doc_id in enumerate(retrieved[:k], 1):
            if doc_id in pos_set:
                rr_sum += 1.0 / rank
                break

    return rr_sum / max(total, 1)


def compute_ndcg_at_k(retrieved_ids: List[List[str]], positive_ids: List[List[str]], k: int) -> float:
    """计算 NDCG@K"""
    if not positive_ids:
        return 0.0

    ndcg_sum = 0.0
    total = 0

    for retrieved, positives in zip(retrieved_ids, positive_ids):
        if not positives:
            continue
        total += 1
        pos_set = set(positives)

        # 计算DCG
        dcg = 0.0
        for rank, doc_id in enumerate(retrieved[:k], 1):
            if doc_id in pos_set:
                dcg += 1.0 / math.log2(rank + 1)

        # 计算IDCG（理想排序）
        num_relevant = min(len(positives), k)
        idcg = sum(1.0 / math.log2(r + 1) for r in range(1, num_relevant + 1))

        if idcg > 0:
            ndcg_sum += dcg / idcg

    return ndcg_sum / max(total, 1)


def evaluate(
    index, doc_ids, query_embeddings, positive_doc_ids,
    k_values, hnsw_ef_search=128
):
    """
    执行完整评估

    返回:
        结果字典，包含语义指标和效率指标
    """
    max_k = max(k_values)

    # 检索
    scores, indices, qps, avg_latency, avg_visited_nodes = search_index(
        index, query_embeddings, max_k, hnsw_ef_search
    )

    # 转换为文档ID
    retrieved_ids = []
    for row in indices:
        ids = []
        for idx in row:
            if 0 <= idx < len(doc_ids):
                ids.append(doc_ids[idx])
            else:
                ids.append("")
        retrieved_ids.append(ids)

    # 计算各指标
    results = {
        "semantic": {},
        "efficiency": {
            "qps": qps,
            "avg_latency_ms": avg_latency,
            "avg_visited_nodes": avg_visited_nodes,
            "num_queries": len(query_embeddings),
            "hnsw_ef_search": hnsw_ef_search,
        }
    }

    for k in k_values:
        recall = compute_recall_at_k(retrieved_ids, positive_doc_ids, k)
        mrr = compute_mrr_at_k(retrieved_ids, positive_doc_ids, k)
        ndcg = compute_ndcg_at_k(retrieved_ids, positive_doc_ids, k)

        results["semantic"][f"Recall@{k}"] = recall
        results["semantic"][f"MRR@{k}"] = mrr
        results["semantic"][f"NDCG@{k}"] = ndcg

    return results


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("运行检索评估")
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 加载编码器
    encoder, model_name = load_encoder(args.encoder_path)
    encoder.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 加载索引
    index, doc_ids = load_faiss_index(args.index_path)

    # 加载测试数据
    queries, answers, positive_doc_ids = load_test_data(args.test_file)

    # 编码查询
    logger.info("编码查询...")
    query_embeddings = encode_queries(
        encoder, tokenizer, queries,
        batch_size=args.batch_size,
        max_query_length=args.max_query_length
    )

    # 评估
    logger.info("执行评估...")
    results = evaluate(
        index, doc_ids, query_embeddings, positive_doc_ids,
        args.k_values, args.hnsw_ef_search
    )

    # 保存结果
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{args.experiment_name}_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"结果已保存: {output_file}")

    # 打印汇总
    logger.info("\n" + "=" * 50)
    logger.info("评估结果汇总")
    logger.info("=" * 50)

    logger.info("语义指标:")
    for metric, value in results["semantic"].items():
        logger.info(f"  {metric}: {value:.4f}")

    logger.info("效率指标:")
    eff = results["efficiency"]
    logger.info(f"  QPS: {eff['qps']:.1f}")
    logger.info(f"  平均延迟: {eff['avg_latency_ms']:.2f} ms")
    logger.info(f"  平均访问节点数: {eff['avg_visited_nodes']:.0f}")

    logger.info("评估完成！")


if __name__ == "__main__":
    main()