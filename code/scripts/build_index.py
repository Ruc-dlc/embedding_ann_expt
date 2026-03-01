#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
向量索引构建脚本

使用训练好的BiEncoder编码语料库（psgs_w100.tsv），
构建FAISS向量索引（HNSW / Flat），用于后续检索评估。

使用方法:
    python scripts/build_index.py \
        --encoder_path checkpoints/distance_aware/final_model \
        --corpus_path data_set/psgs_w100.tsv \
        --output_path indices/hnsw_index \
        --index_type hnsw

论文章节：第5章 5.2节 - 索引构建
"""

import argparse
import logging
import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="构建向量索引")

    parser.add_argument("--encoder_path", type=str, required=True,
                        help="训练好的BiEncoder模型目录（含bi_encoder_config.json）")
    parser.add_argument("--corpus_path", type=str, required=True,
                        help="语料库文件路径（psgs_w100.tsv格式：id\\ttext\\ttitle）")
    parser.add_argument("--output_path", type=str, required=True,
                        help="索引输出目录")
    parser.add_argument("--index_type", type=str, default="hnsw",
                        choices=["hnsw", "flat"],
                        help="索引类型（hnsw: 近似搜索，flat: 精确搜索）")
    parser.add_argument("--hnsw_m", type=int, default=32,
                        help="HNSW M参数（每个节点的连接数）")
    parser.add_argument("--hnsw_ef_construction", type=int, default=200,
                        help="HNSW构建时的ef参数")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="编码批次大小")
    parser.add_argument("--max_doc_length", type=int, default=256,
                        help="文档最大token长度")
    parser.add_argument("--max_docs", type=int, default=None,
                        help="最大编码文档数（调试用）")

    return parser.parse_args()


def load_encoder(encoder_path: str):
    """加载训练好的BiEncoder模型"""
    from src.models.bi_encoder import BiEncoder

    logger.info(f"加载编码器: {encoder_path}")
    model = BiEncoder.from_pretrained(encoder_path)
    model.eval()

    # 获取模型名称用于加载tokenizer
    model_name = model.model_name

    return model, model_name


def load_corpus_tsv(corpus_path: str, max_docs=None):
    """
    加载psgs_w100.tsv格式的语料库

    格式：id\\ttext\\ttitle（第一行为表头）

    参数:
        corpus_path: TSV文件路径
        max_docs: 最大加载文档数

    返回:
        doc_ids: 文档ID列表
        doc_texts: 文档文本列表（title + " " + text）
    """
    logger.info(f"加载语料库: {corpus_path}")
    path = Path(corpus_path)

    if not path.exists():
        raise FileNotFoundError(f"语料库文件不存在: {corpus_path}")

    file_size_gb = path.stat().st_size / (1024 ** 3)
    logger.info(f"文件大小: {file_size_gb:.2f} GB")

    doc_ids = []
    doc_texts = []

    with open(corpus_path, 'r', encoding='utf-8') as f:
        # 跳过表头
        header = f.readline().strip()
        logger.info(f"表头: {header}")

        for line_num, line in enumerate(f):
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue

            doc_id = parts[0]
            text = parts[1] if len(parts) > 1 else ""
            title = parts[2] if len(parts) > 2 else ""

            # 拼接标题和正文
            doc_text = f"{title} {text}".strip() if title else text
            doc_ids.append(doc_id)
            doc_texts.append(doc_text)

            if max_docs is not None and len(doc_ids) >= max_docs:
                break

            if (line_num + 1) % 1000000 == 0:
                logger.info(f"  已加载 {line_num + 1} 篇文档...")

    logger.info(f"语料库加载完成，共 {len(doc_ids)} 篇文档")
    return doc_ids, doc_texts


@torch.no_grad()
def encode_corpus(encoder, tokenizer, doc_texts, batch_size=256, max_doc_length=256):
    """
    批量编码语料库文档

    参数:
        encoder: BiEncoder模型
        tokenizer: HuggingFace tokenizer
        doc_texts: 文档文本列表
        batch_size: 编码批次大小
        max_doc_length: 文档最大token长度

    返回:
        embeddings: numpy数组 [num_docs, embedding_dim]
    """
    device = next(encoder.parameters()).device
    encoder.eval()

    all_embeddings = []
    total = len(doc_texts)

    logger.info(f"开始编码 {total} 篇文档，batch_size={batch_size}")

    for start_idx in range(0, total, batch_size):
        end_idx = min(start_idx + batch_size, total)
        batch_texts = doc_texts[start_idx:end_idx]

        encoded = tokenizer(
            batch_texts,
            max_length=max_doc_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        embeddings = encoder.encode_document(input_ids, attention_mask)
        all_embeddings.append(embeddings.cpu().numpy())

        if ((start_idx // batch_size + 1) % 200 == 0) or (end_idx == total):
            logger.info(f"  编码进度: {end_idx}/{total} ({100*end_idx/total:.1f}%)")

    result = np.concatenate(all_embeddings, axis=0)
    logger.info(f"编码完成，向量矩阵形状: {result.shape}")
    return result


def build_faiss_index(embeddings, index_type, hnsw_m=32, hnsw_ef_construction=200):
    """
    构建FAISS索引

    参数:
        embeddings: 向量矩阵 [num_docs, dim]
        index_type: 索引类型 ("hnsw" 或 "flat")
        hnsw_m: HNSW M参数
        hnsw_ef_construction: HNSW ef_construction参数

    返回:
        FAISS索引对象
    """
    import faiss

    dim = embeddings.shape[1]
    embeddings = embeddings.astype(np.float32)

    logger.info(f"构建 {index_type} 索引，维度={dim}，文档数={embeddings.shape[0]}")

    start_time = time.time()

    if index_type == "hnsw":
        # HNSW索引（默认L2距离）
        # 因向量已L2归一化：||a-b||² = 2(1-⟨a,b⟩)，
        # L2距离排序与内积/余弦相似度排序完全等价
        index = faiss.IndexHNSWFlat(dim, hnsw_m)
        index.hnsw.efConstruction = hnsw_ef_construction
        index.add(embeddings)
    elif index_type == "flat":
        # Flat精确索引（用于计算ground truth）
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
    else:
        raise ValueError(f"不支持的索引类型: {index_type}")

    build_time = time.time() - start_time
    logger.info(f"索引构建完成，耗时 {build_time:.1f} 秒，索引包含 {index.ntotal} 个向量")

    return index


def save_index(index, doc_ids, output_path, config):
    """保存索引和相关元数据"""
    import faiss

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存FAISS索引
    index_file = output_dir / "index.faiss"
    faiss.write_index(index, str(index_file))
    logger.info(f"FAISS索引已保存: {index_file}")

    # 保存文档ID映射
    doc_ids_file = output_dir / "doc_ids.txt"
    with open(doc_ids_file, 'w', encoding='utf-8') as f:
        for doc_id in doc_ids:
            f.write(f"{doc_id}\n")
    logger.info(f"文档ID映射已保存: {doc_ids_file}")

    # 保存索引配置
    config_file = output_dir / "index_config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    logger.info(f"索引配置已保存: {config_file}")


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("构建向量索引")
    logger.info(f"  索引类型: {args.index_type}")
    logger.info(f"  编码器: {args.encoder_path}")
    logger.info(f"  语料库: {args.corpus_path}")
    logger.info("=" * 60)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 加载编码器
    encoder, model_name = load_encoder(args.encoder_path)
    encoder.to(device)

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 加载语料库
    doc_ids, doc_texts = load_corpus_tsv(args.corpus_path, args.max_docs)

    # 编码语料库
    embeddings = encode_corpus(
        encoder, tokenizer, doc_texts,
        batch_size=args.batch_size,
        max_doc_length=args.max_doc_length
    )

    # 构建索引
    index = build_faiss_index(
        embeddings, args.index_type,
        hnsw_m=args.hnsw_m,
        hnsw_ef_construction=args.hnsw_ef_construction
    )

    # 保存
    config = {
        "index_type": args.index_type,
        "num_docs": len(doc_ids),
        "embedding_dim": embeddings.shape[1],
        "encoder_path": args.encoder_path,
        "corpus_path": args.corpus_path,
        "hnsw_m": args.hnsw_m if args.index_type == "hnsw" else None,
        "hnsw_ef_construction": args.hnsw_ef_construction if args.index_type == "hnsw" else None,
    }
    save_index(index, doc_ids, args.output_path, config)

    logger.info("索引构建完成！")


if __name__ == "__main__":
    main()