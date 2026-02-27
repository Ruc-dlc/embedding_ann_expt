#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
表示空间分析脚本

对指定模型计算向量分布特性指标：Pos Mean、Pos Var、Neg Mean、Neg Var、
Alignment、Uniformity，产出论文 §5.3 所需数据。

产出图表：
    - 表5.7 tab:representation_distribution — 向量分布特性随w的变化
    - 图5.2 fig:pos_mean_var_curves — Pos Mean/Var vs w
    - 图5.3 fig:alignment_uniformity_curves — A-U vs w
    - 图5.4 fig:alignment_uniformity_scatter — A-U二维散点

使用方法:
    python scripts/run_representation_eval.py \
        --encoder_path checkpoints/ablation_w0.6/final_model \
        --dev_file data_set/NQ/nq-dev.json \
        --output_path results/representation/ \
        --experiment_name w0.6_nq \
        --num_samples 10000

论文章节：第5章 5.3节 - 表示空间分析
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="表示空间分析")

    parser.add_argument("--encoder_path", type=str, required=True,
                        help="训练好的BiEncoder模型目录")
    parser.add_argument("--dev_file", type=str, required=True,
                        help="开发集JSON文件路径（DPR格式，含positive_ctxs）")
    parser.add_argument("--output_path", type=str, required=True,
                        help="结果输出目录")
    parser.add_argument("--num_samples", type=int, default=10000,
                        help="采样数量（论文使用10K）")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="编码批次大小")
    parser.add_argument("--max_query_length", type=int, default=64,
                        help="查询最大token长度")
    parser.add_argument("--max_doc_length", type=int, default=256,
                        help="文档最大token长度")
    parser.add_argument("--experiment_name", type=str, default="repr_eval",
                        help="实验名称")

    return parser.parse_args()


def load_dev_pairs(dev_file: str, num_samples: int):
    """
    从DPR格式的开发集加载查询-正例-负例对

    返回:
        queries: 查询文本列表
        pos_texts: 正例文档文本列表
        neg_texts: 负例文档文本列表（来自in-batch其他正例）
    """
    with open(dev_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    queries = []
    pos_texts = []

    for item in data:
        question = item.get('question', '')
        positive_ctxs = item.get('positive_ctxs', [])

        if not positive_ctxs:
            continue

        # 取第一个正例
        pos_ctx = positive_ctxs[0]
        title = pos_ctx.get('title', '')
        text = pos_ctx.get('text', '')
        pos_text = f"{title} {text}".strip() if title else text

        queries.append(question)
        pos_texts.append(pos_text)

        if len(queries) >= num_samples:
            break

    logger.info(f"加载了 {len(queries)} 个查询-正例对")
    return queries, pos_texts


@torch.no_grad()
def encode_texts(encoder, tokenizer, texts, batch_size, max_length, encode_fn_name):
    """批量编码文本"""
    device = next(encoder.parameters()).device
    encoder.eval()
    all_embeddings = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        encoded = tokenizer(
            batch, max_length=max_length, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        encode_fn = getattr(encoder, encode_fn_name)
        emb = encode_fn(input_ids, attention_mask)
        all_embeddings.append(emb.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def compute_cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """批量计算余弦相似度"""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return np.sum(a_norm * b_norm, axis=1)


def compute_alignment(query_emb: np.ndarray, pos_emb: np.ndarray) -> float:
    """
    计算Alignment指标
    L_align = E[ ||q - d+||^2 ]
    """
    diffs = query_emb - pos_emb
    l2_sq = np.sum(diffs ** 2, axis=1)
    return float(np.mean(l2_sq))


def compute_uniformity(embeddings: np.ndarray, t: float = 2.0) -> float:
    """
    计算Uniformity指标
    L_uniform = log E[ e^{-t * ||x - y||^2} ]
    采样估计：从embeddings中随机采样对
    """
    n = len(embeddings)
    if n < 2:
        return 0.0

    # 随机采样10000对（避免O(n^2)）
    num_pairs = min(10000, n * (n - 1) // 2)
    idx1 = np.random.randint(0, n, size=num_pairs)
    idx2 = np.random.randint(0, n, size=num_pairs)
    # 排除自身配对
    mask = idx1 != idx2
    idx1, idx2 = idx1[mask], idx2[mask]

    diffs = embeddings[idx1] - embeddings[idx2]
    l2_sq = np.sum(diffs ** 2, axis=1)
    return float(np.log(np.mean(np.exp(-t * l2_sq))))


def main():
    args = parse_args()

    from src.models.bi_encoder import BiEncoder

    logger.info("=" * 60)
    logger.info("表示空间分析")
    logger.info(f"采样数量: {args.num_samples}")
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    logger.info(f"加载编码器: {args.encoder_path}")
    encoder = BiEncoder.from_pretrained(args.encoder_path)
    encoder.to(device)
    encoder.eval()
    tokenizer = AutoTokenizer.from_pretrained(encoder.model_name)

    # 加载数据
    queries, pos_texts = load_dev_pairs(args.dev_file, args.num_samples)

    # 编码
    logger.info("编码查询...")
    query_emb = encode_texts(
        encoder, tokenizer, queries,
        args.batch_size, args.max_query_length, 'encode_query'
    )

    logger.info("编码正例文档...")
    pos_emb = encode_texts(
        encoder, tokenizer, pos_texts,
        args.batch_size, args.max_doc_length, 'encode_document'
    )

    # 计算指标
    logger.info("计算表示空间指标...")

    # 正例余弦相似度
    pos_cos_sim = compute_cosine_similarity(query_emb, pos_emb)
    pos_mean = float(np.mean(pos_cos_sim))
    pos_var = float(np.var(pos_cos_sim))

    # 负例余弦相似度（in-batch其他正例作为负例）
    # 随机打乱pos_emb作为负例近似
    neg_indices = np.random.permutation(len(pos_emb))
    # 确保不与自身配对
    for i in range(len(neg_indices)):
        if neg_indices[i] == i:
            swap_idx = (i + 1) % len(neg_indices)
            neg_indices[i], neg_indices[swap_idx] = neg_indices[swap_idx], neg_indices[i]

    neg_emb = pos_emb[neg_indices]
    neg_cos_sim = compute_cosine_similarity(query_emb, neg_emb)
    neg_mean = float(np.mean(neg_cos_sim))
    neg_var = float(np.var(neg_cos_sim))

    # Alignment & Uniformity
    alignment = compute_alignment(query_emb, pos_emb)
    all_emb = np.concatenate([query_emb, pos_emb], axis=0)
    uniformity = compute_uniformity(all_emb)

    results = {
        "experiment_name": args.experiment_name,
        "num_samples": len(queries),
        "pos_mean": pos_mean,
        "pos_var": pos_var,
        "neg_mean": neg_mean,
        "neg_var": neg_var,
        "alignment": alignment,
        "uniformity": uniformity,
    }

    # 保存
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{args.experiment_name}_representation.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"结果已保存: {output_file}")

    # 打印汇总
    logger.info("\n" + "=" * 50)
    logger.info("表示空间指标:")
    logger.info(f"  Pos Mean:    {pos_mean:.4f}")
    logger.info(f"  Pos Var:     {pos_var:.4f}")
    logger.info(f"  Neg Mean:    {neg_mean:.4f}")
    logger.info(f"  Neg Var:     {neg_var:.4f}")
    logger.info(f"  Alignment:   {alignment:.4f}")
    logger.info(f"  Uniformity:  {uniformity:.4f}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()