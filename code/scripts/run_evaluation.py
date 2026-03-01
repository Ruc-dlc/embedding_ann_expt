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
    parser.add_argument("--mode", type=str, default="auto",
                        choices=["auto", "passage_id", "answer_match"],
                        help="评估模式: auto(根据文件格式自动选择), passage_id(JSON), answer_match(CSV)")
    parser.add_argument("--corpus_path", type=str, default=None,
                        help="语料库文件路径（用于答案匹配模式，默认自动推断）")

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


def load_test_data(test_file: str, mode: str = "auto"):
    """
    加载测试数据

    支持两种格式：
    - CSV: question, answers（answers用制表符或逗号分隔）
    - JSON: DPR格式（含positive_ctxs）

    参数:
        test_file: 测试文件路径
        mode: 评估模式 ("auto"|"passage_id"|"answer_match")

    返回:
        queries: 查询文本列表
        answers: 答案列表（每个查询对应多个答案）
        positive_doc_ids: 正例文档ID列表（如果有）
        mode: 实际使用的评估模式
    """
    path = Path(test_file)
    logger.info(f"加载测试数据: {path} (mode={mode})")

    queries = []
    answers = []
    positive_doc_ids = []

    if path.suffix == '.csv':
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if len(row) >= 2:
                    queries.append(row[0])
                    # 支持逗号分隔的多个答案
                    answers.append([ans.strip() for ans in row[1].split(',')])
                    positive_doc_ids.append([])
        # CSV文件强制使用answer_match模式
        if mode == "auto":
            mode = "answer_match"
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
        # JSON文件默认使用passage_id模式
        if mode == "auto":
            mode = "passage_id"
    else:
        raise ValueError(f"不支持的测试文件格式: {path.suffix}")

    logger.info(f"测试数据加载完成，共 {len(queries)} 条查询 (mode={mode})")
    return queries, answers, positive_doc_ids, mode


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

    # hnsw_stats 仅在单线程搜索时准确（FAISS官方文档要求）
    # 保存原线程数，临时设置为单线程以确保统计准确
    has_hnsw_stats = hasattr(faiss, 'cvar') and hasattr(faiss.cvar, 'hnsw_stats')
    prev_num_threads = None
    if has_hnsw_stats:
        prev_num_threads = faiss.omp_get_max_threads()
        faiss.omp_set_num_threads(1)
        faiss.cvar.hnsw_stats.reset()

    start_time = time.time()
    scores, indices = index.search(query_embeddings, k)
    search_time = time.time() - start_time

    qps = num_queries / search_time if search_time > 0 else 0
    avg_latency = search_time / num_queries * 1000  # ms

    # 收集Visited Nodes（HNSW距离计算次数 ≈ 访问节点数）
    avg_visited_nodes = 0
    if has_hnsw_stats:
        total_ndis = faiss.cvar.hnsw_stats.ndis
        avg_visited_nodes = total_ndis / num_queries if num_queries > 0 else 0
        # 恢复原线程数
        faiss.omp_set_num_threads(prev_num_threads)
        if avg_visited_nodes == 0 and num_queries > 0:
            logger.warning(
                "hnsw_stats.ndis 返回 0！HNSW Visited Nodes 统计不可用。"
                "请检查 FAISS 版本（需要 >=1.9.0 且包含 #3309 修复）。"
            )
    elif hasattr(index, 'hnsw'):
        logger.warning(
            "当前 FAISS 版本不支持 faiss.cvar.hnsw_stats，"
            "无法获取 Visited Nodes 统计。"
        )

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


def _answer_in_text(text: str, answers: List[str]) -> bool:
    """检查文档文本中是否包含任一答案字符串（大小写不敏感）"""
    text_lower = text.lower()
    for answer in answers:
        if answer.strip().lower() in text_lower:
            return True
    return False


def evaluate_answer_match(
    index, doc_ids, query_embeddings, ground_truth_answers,
    k_values, hnsw_ef_search=128, corpus_path=None
):
    """
    基于答案字符串匹配的评估

    对于CSV测试集，检查检索到的文档文本中是否包含正确答案字符串。

    参数:
        index: FAISS索引
        doc_ids: 文档ID列表
        query_embeddings: 查询向量
        ground_truth_answers: 每个查询的答案列表
        k_values: 评估的K值列表
        hnsw_ef_search: HNSW搜索ef参数
        corpus_path: 语料库文件路径（psgs_w100.tsv），用于获取文档文本

    返回:
        结果字典，包含语义指标和效率指标
    """
    max_k = max(k_values)

    # 检索
    scores, indices, qps, avg_latency, avg_visited_nodes = search_index(
        index, query_embeddings, max_k, hnsw_ef_search
    )

    # 加载文档文本（从 psgs_w100.tsv）
    doc_texts = {}
    if corpus_path and Path(corpus_path).exists():
        logger.info(f"加载语料库文本: {corpus_path}")
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if line_num == 0:
                    continue  # 跳过表头
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    doc_id = parts[0]
                    text = parts[1]
                    if len(parts) >= 3:
                        text = parts[1] + " " + parts[2]  # title + text
                    doc_texts[doc_id] = text
        logger.info(f"语料库已加载: {len(doc_texts)} 篇文档")
    else:
        logger.warning(f"语料库文件不存在或未指定: {corpus_path}，答案匹配将无法进行")

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

    # 答案匹配评估
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
        hits = 0
        total = 0
        rr_sum = 0.0

        for q_idx, (ret_ids, ans_list) in enumerate(zip(retrieved_ids, ground_truth_answers)):
            if not ans_list:
                continue
            total += 1
            found = False
            for rank, doc_id in enumerate(ret_ids[:k], 1):
                doc_text = doc_texts.get(doc_id, "")
                if _answer_in_text(doc_text, ans_list):
                    if not found:
                        rr_sum += 1.0 / rank
                        found = True
                    break
            if found:
                hits += 1

        recall = hits / max(total, 1)
        mrr = rr_sum / max(total, 1)

        results["semantic"][f"Recall@{k}"] = recall
        results["semantic"][f"MRR@{k}"] = mrr

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
    queries, answers, positive_doc_ids, eval_mode = load_test_data(args.test_file, args.mode)

    # 编码查询
    logger.info("编码查询...")
    query_embeddings = encode_queries(
        encoder, tokenizer, queries,
        batch_size=args.batch_size,
        max_query_length=args.max_query_length
    )

    # 评估
    logger.info(f"执行评估 (mode={eval_mode})...")
    if eval_mode == "answer_match":
        # CSV测试集使用答案匹配模式
        corpus_path = args.corpus_path
        if corpus_path is None:
            corpus_path = str(Path(args.test_file).parent.parent / "psgs_w100.tsv")
        results = evaluate_answer_match(
            index, doc_ids, query_embeddings, answers,
            args.k_values, args.hnsw_ef_search,
            corpus_path=corpus_path
        )
    else:
        # JSON测试集使用passage_id模式
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