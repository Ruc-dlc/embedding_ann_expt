#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
消融实验评估脚本

对4组消融模型（A/B/C/D）进行完整评估，收集检索性能、表示空间结构、ANN效率三类指标。

消融模型配置：
    A (Baseline): InfoNCE, w=0, 单阶段(In-Batch)
    B (+L_dis):   InfoNCE + L_dis, w=0.6, 单阶段(In-Batch)
    C (+Curriculum): InfoNCE, w=0, 三阶段
    D (Full):     InfoNCE + L_dis, w=0.6, 三阶段

产出图表：
    - 表5.13 tab:ablation_retrieval — 消融实验检索性能对比
    - 表5.14 tab:ablation_representation — 消融实验表示空间对比
    - 表5.15 tab:ablation_efficiency — 消融实验ANN效率对比
    - 图5.9 fig:ablation_pos_mean — Pos Mean柱状图
    - 图5.10 fig:ablation_align_uni — A-U对比图
    - 图5.11 fig:ablation_recall_ef — Recall@10 vs ef_search消融曲线
    - 图5.12 fig:ablation_visited — Visited Nodes柱状图

使用方法:
    python scripts/run_ablation.py \
        --model_dirs A=checkpoints/baseline B=checkpoints/plus_ldis \
                     C=checkpoints/ablation_w0.0 D=checkpoints/ablation_w0.6 \
        --index_dirs A=indices/baseline B=indices/plus_ldis \
                     C=indices/w0.0 D=indices/w0.6 \
        --test_file data_set/NQ/nq-test.csv \
        --dev_file data_set/NQ/nq-dev.json \
        --output_path results/ablation/

论文章节：第5章 5.5节 - 消融实验
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 消融模型标签
ABLATION_MODELS = ['A', 'B', 'C', 'D']
ABLATION_LABELS = {
    'A': 'Baseline (InfoNCE)',
    'B': '+L_dis (w=0.6)',
    'C': '+Curriculum (三阶段)',
    'D': 'Full Model',
}

# ef_search扫参值（用于消融Recall vs ef曲线）
EF_VALUES = [16, 32, 64, 128, 256, 512]

# 消融效率评估使用的ef_search（论文表5.15使用ef=100）
ABLATION_EF_SEARCH = 100


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="消融实验评估")

    parser.add_argument("--model_dirs", type=str, nargs='+', required=True,
                        help="模型目录，格式: A=path/to/modelA B=path/to/modelB ...")
    parser.add_argument("--index_dirs", type=str, nargs='+', required=True,
                        help="索引目录，格式: A=path/to/indexA B=path/to/indexB ...")
    parser.add_argument("--test_file", type=str, required=True,
                        help="测试集文件路径")
    parser.add_argument("--dev_file", type=str, default=None,
                        help="开发集文件路径（用于表示空间分析）")
    parser.add_argument("--output_path", type=str, required=True,
                        help="结果输出目录")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="编码批次大小")
    parser.add_argument("--num_repr_samples", type=int, default=10000,
                        help="表示空间分析采样数量")
    parser.add_argument("--dataset_name", type=str, default="NQ",
                        help="数据集名称（NQ或TriviaQA）")

    return parser.parse_args()


def parse_kv_args(kv_list):
    """解析 KEY=VALUE 格式的参数列表"""
    result = {}
    for item in kv_list:
        key, value = item.split('=', 1)
        result[key.strip()] = value.strip()
    return result


def evaluate_retrieval(encoder, tokenizer, index, doc_ids, test_data, k_values, ef_search, batch_size, device):
    """评估检索性能"""
    from scripts.run_evaluation import (
        encode_queries, search_index,
        compute_recall_at_k, compute_mrr_at_k, compute_ndcg_at_k
    )

    queries, answers, positive_doc_ids = test_data
    query_emb = encode_queries(encoder, tokenizer, queries, batch_size=batch_size)

    max_k = max(k_values)
    scores, indices, qps, avg_latency, avg_visited_nodes = search_index(
        index, query_emb, max_k, hnsw_ef_search=ef_search
    )

    retrieved_ids = []
    for row in indices:
        ids = [doc_ids[idx] if 0 <= idx < len(doc_ids) else "" for idx in row]
        retrieved_ids.append(ids)

    results = {}
    for k in k_values:
        results[f"recall@{k}"] = compute_recall_at_k(retrieved_ids, positive_doc_ids, k)
        results[f"mrr@{k}"] = compute_mrr_at_k(retrieved_ids, positive_doc_ids, k)

    results["avg_latency_ms"] = avg_latency
    results["avg_visited_nodes"] = avg_visited_nodes
    results["qps"] = qps

    return results


def evaluate_ef_sweep(encoder, tokenizer, index, doc_ids, test_data, ef_values, k, batch_size):
    """对不同ef_search值评估Recall和Visited Nodes"""
    from scripts.run_evaluation import (
        encode_queries, search_index, compute_recall_at_k
    )

    queries, answers, positive_doc_ids = test_data
    query_emb = encode_queries(encoder, tokenizer, queries, batch_size=batch_size)

    sweep = {}
    for ef in ef_values:
        scores, indices, qps, avg_latency, avg_visited_nodes = search_index(
            index, query_emb, k, hnsw_ef_search=ef
        )

        retrieved_ids = []
        for row in indices:
            ids = [doc_ids[idx] if 0 <= idx < len(doc_ids) else "" for idx in row]
            retrieved_ids.append(ids)

        recall = compute_recall_at_k(retrieved_ids, positive_doc_ids, k)
        sweep[ef] = {
            f"recall@{k}": recall,
            "avg_latency_ms": avg_latency,
            "avg_visited_nodes": avg_visited_nodes,
        }

    return sweep


def evaluate_representation(encoder, tokenizer, dev_file, num_samples, batch_size, device):
    """评估表示空间指标"""
    from scripts.run_representation_eval import (
        load_dev_pairs, encode_texts,
        compute_cosine_similarity, compute_alignment, compute_uniformity
    )

    queries, pos_texts = load_dev_pairs(dev_file, num_samples)

    query_emb = encode_texts(encoder, tokenizer, queries, batch_size, 64, 'encode_query')
    pos_emb = encode_texts(encoder, tokenizer, pos_texts, batch_size, 256, 'encode_document')

    pos_cos_sim = compute_cosine_similarity(query_emb, pos_emb)
    alignment = compute_alignment(query_emb, pos_emb)
    all_emb = np.concatenate([query_emb, pos_emb], axis=0)
    uniformity = compute_uniformity(all_emb)

    return {
        "pos_mean": float(np.mean(pos_cos_sim)),
        "pos_var": float(np.var(pos_cos_sim)),
        "alignment": alignment,
        "uniformity": uniformity,
    }


def main():
    args = parse_args()
    from src.models.bi_encoder import BiEncoder
    from scripts.run_evaluation import load_faiss_index, load_test_data

    model_dirs = parse_kv_args(args.model_dirs)
    index_dirs = parse_kv_args(args.index_dirs)

    logger.info("=" * 60)
    logger.info("消融实验评估 (ABCD)")
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载测试数据
    test_data = load_test_data(args.test_file)

    all_results = {}

    for model_id in ABLATION_MODELS:
        if model_id not in model_dirs or model_id not in index_dirs:
            logger.warning(f"模型 {model_id} 缺少路径配置，跳过")
            continue

        logger.info(f"\n{'='*50}")
        logger.info(f"评估模型 {model_id}: {ABLATION_LABELS[model_id]}")
        logger.info(f"{'='*50}")

        # 加载模型和索引
        encoder = BiEncoder.from_pretrained(model_dirs[model_id])
        encoder.to(device)
        encoder.eval()
        tokenizer = AutoTokenizer.from_pretrained(encoder.model_name)

        index, doc_ids = load_faiss_index(index_dirs[model_id])

        model_results = {"label": ABLATION_LABELS[model_id]}

        # 1. 检索性能 (tab:ablation_retrieval)
        logger.info("评估检索性能...")
        retrieval = evaluate_retrieval(
            encoder, tokenizer, index, doc_ids, test_data,
            k_values=[10, 100], ef_search=128,
            batch_size=args.batch_size, device=device
        )
        model_results["retrieval"] = retrieval

        # 2. ANN效率 (tab:ablation_efficiency)
        logger.info(f"评估ANN效率 (ef_search={ABLATION_EF_SEARCH})...")
        efficiency = evaluate_retrieval(
            encoder, tokenizer, index, doc_ids, test_data,
            k_values=[10], ef_search=ABLATION_EF_SEARCH,
            batch_size=args.batch_size, device=device
        )
        model_results["efficiency"] = efficiency

        # 3. ef_search扫参 (fig:ablation_recall_ef)
        logger.info("评估ef_search敏感度...")
        ef_sweep = evaluate_ef_sweep(
            encoder, tokenizer, index, doc_ids, test_data,
            ef_values=EF_VALUES, k=10, batch_size=args.batch_size
        )
        model_results["ef_sweep"] = {str(k): v for k, v in ef_sweep.items()}

        # 4. 表示空间 (tab:ablation_representation)
        if args.dev_file:
            logger.info("评估表示空间...")
            representation = evaluate_representation(
                encoder, tokenizer, args.dev_file,
                args.num_repr_samples, args.batch_size, device
            )
            model_results["representation"] = representation

        all_results[model_id] = model_results

        # 打印当前模型结果
        logger.info(f"\n模型 {model_id} 结果:")
        logger.info(f"  Recall@10: {retrieval.get('recall@10', 0):.4f}")
        logger.info(f"  MRR@10:    {retrieval.get('mrr@10', 0):.4f}")
        logger.info(f"  Recall@100:{retrieval.get('recall@100', 0):.4f}")
        logger.info(f"  Visited:   {efficiency.get('avg_visited_nodes', 0):.0f}")
        logger.info(f"  Latency:   {efficiency.get('avg_latency_ms', 0):.2f} ms")
        if "representation" in model_results:
            r = model_results["representation"]
            logger.info(f"  Pos Mean:  {r['pos_mean']:.4f}")
            logger.info(f"  Alignment: {r['alignment']:.4f}")
            logger.info(f"  Uniformity:{r['uniformity']:.4f}")

    # 保存结果
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"ablation_{args.dataset_name}_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info(f"\n结果已保存: {output_file}")

    # 打印汇总对比表
    logger.info("\n" + "=" * 80)
    logger.info("消融实验汇总:")
    logger.info(f"{'模型':>20} | {'R@10':>8} | {'MRR@10':>8} | {'R@100':>8} | {'Visited':>8} | {'Latency':>8}")
    logger.info("-" * 80)
    for mid in ABLATION_MODELS:
        if mid not in all_results:
            continue
        r = all_results[mid]
        ret = r.get("retrieval", {})
        eff = r.get("efficiency", {})
        logger.info(
            f"{ABLATION_LABELS[mid]:>20} | "
            f"{ret.get('recall@10', 0):>8.4f} | "
            f"{ret.get('mrr@10', 0):>8.4f} | "
            f"{ret.get('recall@100', 0):>8.4f} | "
            f"{eff.get('avg_visited_nodes', 0):>8.0f} | "
            f"{eff.get('avg_latency_ms', 0):>8.2f}"
        )
    logger.info("=" * 80)


if __name__ == "__main__":
    main()