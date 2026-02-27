#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BM25基线评估脚本

使用Pyserini在NQ和TriviaQA测试集上运行BM25检索基线。
BM25参数：k1=0.82, b=0.68（与论文一致）。

产出图表：
    - 表5.6 tab:w_retrieval_results — BM25行

使用方法:
    python scripts/run_bm25_eval.py \
        --corpus_file data_set/psgs_w100.tsv \
        --test_file data_set/NQ/nq-test.csv \
        --output_path results/bm25/ \
        --dataset_name NQ

论文章节：第5章 5.1节 - 基准方法（BM25）
"""

import argparse
import csv
import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# BM25参数（论文规定）
BM25_K1 = 0.82
BM25_B = 0.68


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="BM25基线评估")

    parser.add_argument("--corpus_file", type=str, required=True,
                        help="语料库文件路径（psgs_w100.tsv格式：id\\ttext\\ttitle）")
    parser.add_argument("--test_file", type=str, required=True,
                        help="测试集文件路径（CSV或JSON格式）")
    parser.add_argument("--output_path", type=str, required=True,
                        help="结果输出目录")
    parser.add_argument("--dataset_name", type=str, default="NQ",
                        help="数据集名称")
    parser.add_argument("--k_values", type=int, nargs='+',
                        default=[1, 5, 10, 20, 100],
                        help="评估的K值列表")
    parser.add_argument("--num_results", type=int, default=100,
                        help="BM25返回结果数")
    parser.add_argument("--use_rank_bm25", action="store_true",
                        help="使用rank_bm25库（无需Pyserini）")

    return parser.parse_args()


def load_test_data(test_file: str) -> Tuple[List[str], List[List[str]]]:
    """加载测试数据，返回查询和正例文档ID"""
    path = Path(test_file)

    queries = []
    positive_doc_ids = []

    if path.suffix == '.csv':
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if len(row) >= 2:
                    queries.append(row[0])
                    positive_doc_ids.append([])
    elif path.suffix == '.json':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for item in data:
            queries.append(item.get('question', ''))
            pos_ids = []
            for ctx in item.get('positive_ctxs', []):
                pid = ctx.get('passage_id', ctx.get('psg_id', ''))
                if pid:
                    pos_ids.append(str(pid))
            positive_doc_ids.append(pos_ids)

    logger.info(f"加载了 {len(queries)} 条查询")
    return queries, positive_doc_ids


def build_bm25_index(corpus_file: str, max_docs: int = None):
    """
    构建BM25索引

    使用rank_bm25库（轻量级，无需Java环境）。
    """
    from rank_bm25 import BM25Okapi

    logger.info(f"加载语料库: {corpus_file}")

    doc_ids = []
    tokenized_corpus = []
    count = 0

    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                doc_id, text, title = parts[0], parts[1], parts[2]
            elif len(parts) >= 2:
                doc_id, text = parts[0], parts[1]
                title = ""
            else:
                continue

            full_text = f"{title} {text}".strip() if title else text
            doc_ids.append(str(doc_id))
            tokenized_corpus.append(full_text.lower().split())

            count += 1
            if count % 1000000 == 0:
                logger.info(f"  已加载 {count} 篇文档...")
            if max_docs and count >= max_docs:
                break

    logger.info(f"语料库加载完成，共 {len(doc_ids)} 篇文档")
    logger.info("构建BM25索引...")

    # rank_bm25不直接支持自定义k1/b，但BM25Okapi的默认参数接近
    # 我们通过传入参数实现
    bm25 = BM25Okapi(tokenized_corpus, k1=BM25_K1, b=BM25_B)

    logger.info("BM25索引构建完成")
    return bm25, doc_ids


def search_bm25(bm25, doc_ids, queries, num_results=100):
    """BM25检索"""
    logger.info(f"执行BM25检索 (top-{num_results})...")

    all_retrieved_ids = []
    for i, query in enumerate(queries):
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)
        top_indices = scores.argsort()[-num_results:][::-1]
        retrieved = [doc_ids[idx] for idx in top_indices]
        all_retrieved_ids.append(retrieved)

        if (i + 1) % 500 == 0:
            logger.info(f"  已检索 {i+1}/{len(queries)} 条查询")

    return all_retrieved_ids


def compute_metrics(retrieved_ids, positive_ids, k_values):
    """计算Recall@K, MRR@K, NDCG@K"""
    results = {}

    for k in k_values:
        # Recall@K
        hits = 0
        total = 0
        for retrieved, positives in zip(retrieved_ids, positive_ids):
            if not positives:
                continue
            total += 1
            if set(retrieved[:k]) & set(positives):
                hits += 1
        results[f"recall@{k}"] = hits / max(total, 1)

        # MRR@K
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
        results[f"mrr@{k}"] = rr_sum / max(total, 1)

        # NDCG@K
        ndcg_sum = 0.0
        total = 0
        for retrieved, positives in zip(retrieved_ids, positive_ids):
            if not positives:
                continue
            total += 1
            pos_set = set(positives)
            dcg = sum(1.0 / math.log2(r + 1) for r, d in enumerate(retrieved[:k], 1) if d in pos_set)
            num_rel = min(len(positives), k)
            idcg = sum(1.0 / math.log2(r + 1) for r in range(1, num_rel + 1))
            if idcg > 0:
                ndcg_sum += dcg / idcg
        results[f"ndcg@{k}"] = ndcg_sum / max(total, 1)

    return results


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info(f"BM25基线评估 — {args.dataset_name}")
    logger.info(f"参数: k1={BM25_K1}, b={BM25_B}")
    logger.info("=" * 60)

    # 加载测试数据
    queries, positive_doc_ids = load_test_data(args.test_file)

    # 构建BM25索引
    bm25, doc_ids = build_bm25_index(args.corpus_file)

    # 检索
    retrieved_ids = search_bm25(bm25, doc_ids, queries, args.num_results)

    # 计算指标
    results = compute_metrics(retrieved_ids, positive_doc_ids, args.k_values)

    # 保存
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "method": "BM25",
        "dataset": args.dataset_name,
        "params": {"k1": BM25_K1, "b": BM25_B},
        "metrics": results,
    }

    output_file = output_dir / f"bm25_{args.dataset_name}_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f"\n结果已保存: {output_file}")

    # 打印结果
    logger.info("\n" + "=" * 50)
    logger.info(f"BM25 — {args.dataset_name} 结果:")
    for metric, value in results.items():
        logger.info(f"  {metric}: {value:.4f}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()