#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ef_search 敏感度扫参脚本

对指定模型在不同 ef_search 值下评估 Recall@100、Latency、Visited Nodes，
产出论文第五章 §5.4 ANN搜索效率分析所需的全部数据。

产出图表：
    - 表5.9 tab:ef_sensitivity — Recall@100随ef_search的变化
    - 表5.10 tab:latency — 平均查询延迟随ef_search的变化
    - 表5.11 tab:visited_nodes — 平均访问节点数随ef_search的变化
    - 图5.6 fig:recall_vs_ef — Recall@100 vs ef_search
    - 图5.7 fig:latency_vs_ef — Latency vs ef_search
    - 图5.8 fig:recall_vs_visited — Recall@100 vs Visited Nodes

使用方法:
    python scripts/run_ef_sweep.py \
        --encoder_path checkpoints/ablation_w0.6/final_model \
        --index_path indices/hnsw_w0.6 \
        --test_file data_set/NQ/nq-test.csv \
        --output_path results/ef_sweep/ \
        --experiment_name w0.6_nq

论文章节：第5章 5.4节 - ANN搜索效率分析
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 论文中使用的ef_search参数范围
DEFAULT_EF_VALUES = [16, 32, 64, 128, 256, 512]


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="ef_search敏感度扫参")

    parser.add_argument("--encoder_path", type=str, required=True,
                        help="训练好的BiEncoder模型目录")
    parser.add_argument("--index_path", type=str, required=True,
                        help="FAISS索引目录")
    parser.add_argument("--test_file", type=str, required=True,
                        help="测试集文件路径")
    parser.add_argument("--output_path", type=str, required=True,
                        help="结果输出目录")
    parser.add_argument("--ef_values", type=int, nargs='+',
                        default=DEFAULT_EF_VALUES,
                        help="要扫描的ef_search值列表")
    parser.add_argument("--k", type=int, default=100,
                        help="评估Recall@K中的K值（默认100）")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="查询编码批次大小")
    parser.add_argument("--max_query_length", type=int, default=64,
                        help="查询最大token长度")
    parser.add_argument("--experiment_name", type=str, default="ef_sweep",
                        help="实验名称")

    return parser.parse_args()


def main():
    args = parse_args()

    # 复用 run_evaluation.py 中的加载函数
    from scripts.run_evaluation import (
        load_encoder, load_faiss_index, load_test_data,
        encode_queries, search_index, compute_recall_at_k
    )

    logger.info("=" * 60)
    logger.info("ef_search 敏感度扫参")
    logger.info(f"ef_values: {args.ef_values}")
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型和数据
    encoder, model_name = load_encoder(args.encoder_path)
    encoder.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    index, doc_ids = load_faiss_index(args.index_path)
    queries, answers, positive_doc_ids = load_test_data(args.test_file)

    logger.info("编码查询...")
    query_embeddings = encode_queries(
        encoder, tokenizer, queries,
        batch_size=args.batch_size,
        max_query_length=args.max_query_length
    )

    # 对每个ef_search值进行评估
    sweep_results = {}

    for ef in args.ef_values:
        logger.info(f"\n--- ef_search = {ef} ---")

        scores, indices, qps, avg_latency, avg_visited_nodes = search_index(
            index, query_embeddings, args.k, hnsw_ef_search=ef
        )

        # 转换为文档ID并计算Recall@K
        retrieved_ids = []
        for row in indices:
            ids = [doc_ids[idx] if 0 <= idx < len(doc_ids) else "" for idx in row]
            retrieved_ids.append(ids)

        recall_at_k = compute_recall_at_k(retrieved_ids, positive_doc_ids, args.k)

        sweep_results[ef] = {
            f"recall@{args.k}": recall_at_k,
            "avg_latency_ms": avg_latency,
            "avg_visited_nodes": avg_visited_nodes,
            "qps": qps,
        }

        logger.info(f"  Recall@{args.k}: {recall_at_k:.4f}")
        logger.info(f"  Latency: {avg_latency:.2f} ms")
        logger.info(f"  Visited Nodes: {avg_visited_nodes:.0f}")

    # 保存结果
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "experiment_name": args.experiment_name,
        "k": args.k,
        "ef_values": args.ef_values,
        "results": {str(k): v for k, v in sweep_results.items()},
    }

    output_file = output_dir / f"{args.experiment_name}_ef_sweep.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f"\n结果已保存: {output_file}")

    # 打印汇总表格
    logger.info("\n" + "=" * 70)
    logger.info(f"{'ef_search':>10} | {'Recall@'+str(args.k):>12} | {'Latency(ms)':>12} | {'Visited Nodes':>14}")
    logger.info("-" * 70)
    for ef in args.ef_values:
        r = sweep_results[ef]
        logger.info(
            f"{ef:>10} | {r[f'recall@{args.k}']:>12.4f} | "
            f"{r['avg_latency_ms']:>12.2f} | {r['avg_visited_nodes']:>14.0f}"
        )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()