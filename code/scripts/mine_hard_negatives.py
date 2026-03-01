#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage 3 难负例挖掘脚本

使用 Stage 2 训练好的 BiEncoder，在训练数据文档池（2~5M docs）上
进行向量检索，为每个训练查询挖掘模型难负例，用于 Stage 3 训练。

流程:
    1. 加载训练数据 JSON（DPR格式），收集所有文档（positive_ctxs + hard_negative_ctxs + negative_ctxs）
    2. 按 passage_id 去重，构建训练数据文档池
    3. 使用 Stage 2 模型编码文档池，构建 FAISS Flat 索引（精确检索）
    4. 编码训练查询，检索 top-200 候选
    5. 过滤正例 passage_id，保留前 num_negatives 个作为挖掘的难负例
    6. 输出新的 DPR 格式 JSON 文件（hard_negative_ctxs 替换为挖掘结果）

使用方法:
    python scripts/mine_hard_negatives.py \
        --encoder_path checkpoints/distance_aware/stage2_model \
        --train_file data_set/NQ/nq-train.json \
        --output_file data_set/NQ/nq-train-mined.json \
        --batch_size 256

论文章节：第4章 4.2节 - 难负例挖掘
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Stage 3 难负例挖掘")

    parser.add_argument("--encoder_path", type=str, required=True,
                        help="Stage 2 训练好的 BiEncoder 模型目录")
    parser.add_argument("--train_file", type=str, required=True,
                        help="训练数据文件路径（DPR JSON 格式）")
    parser.add_argument("--output_file", type=str, required=True,
                        help="输出文件路径（挖掘后的 DPR JSON）")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="编码批次大小")
    parser.add_argument("--max_doc_length", type=int, default=256,
                        help="文档最大 token 长度")
    parser.add_argument("--max_query_length", type=int, default=64,
                        help="查询最大 token 长度")
    parser.add_argument("--top_k", type=int, default=200,
                        help="每个查询检索的候选数量")
    parser.add_argument("--num_negatives", type=int, default=50,
                        help="每个查询保留的难负例数量")

    return parser.parse_args()


def load_train_data(train_file: str) -> List[Dict]:
    """加载 DPR 格式训练数据"""
    logger.info(f"加载训练数据: {train_file}")
    with open(train_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f"训练数据加载完成，共 {len(data)} 条样本")
    return data


def build_document_pool(train_data: List[Dict]) -> Tuple[List[str], List[str], Dict[str, Dict]]:
    """
    从训练数据中收集所有文档，构建文档池

    收集 positive_ctxs 和 hard_negative_ctxs 中的所有文档，
    按 passage_id 去重。

    返回:
        doc_ids: 去重后的文档 ID 列表
        doc_texts: 对应的文档文本列表（title + text）
        doc_meta: passage_id -> {title, text} 的映射（用于输出）
    """
    logger.info("构建训练数据文档池...")
    doc_meta: Dict[str, Dict] = {}

    for item in train_data:
        # 收集正例文档
        for ctx in item.get('positive_ctxs', []):
            pid = str(ctx.get('passage_id', ctx.get('psg_id', '')))
            if pid and pid not in doc_meta:
                doc_meta[pid] = {
                    'title': ctx.get('title', ''),
                    'text': ctx.get('text', ''),
                }

        # 收集难负例文档
        for ctx in item.get('hard_negative_ctxs', []):
            pid = str(ctx.get('passage_id', ctx.get('psg_id', '')))
            if pid and pid not in doc_meta:
                doc_meta[pid] = {
                    'title': ctx.get('title', ''),
                    'text': ctx.get('text', ''),
                }

        # 收集普通负例文档（如果存在）
        for ctx in item.get('negative_ctxs', []):
            pid = str(ctx.get('passage_id', ctx.get('psg_id', '')))
            if pid and pid not in doc_meta:
                doc_meta[pid] = {
                    'title': ctx.get('title', ''),
                    'text': ctx.get('text', ''),
                }

    doc_ids = list(doc_meta.keys())
    doc_texts = []
    for pid in doc_ids:
        meta = doc_meta[pid]
        title = meta['title']
        text = meta['text']
        doc_text = f"{title} {text}".strip() if title else text
        doc_texts.append(doc_text)

    logger.info(f"文档池构建完成: {len(doc_ids)} 篇去重文档")
    return doc_ids, doc_texts, doc_meta


@torch.no_grad()
def encode_texts(encoder, tokenizer, texts, batch_size, max_length, encode_fn_name):
    """
    批量编码文本

    参数:
        encoder: BiEncoder 模型
        tokenizer: HuggingFace tokenizer
        texts: 文本列表
        batch_size: 批次大小
        max_length: 最大 token 长度
        encode_fn_name: 编码函数名 ('encode_query' 或 'encode_document')

    返回:
        embeddings: numpy 数组 [num_texts, embedding_dim]
    """
    device = next(encoder.parameters()).device
    encoder.eval()
    encode_fn = getattr(encoder, encode_fn_name)

    all_embeddings = []
    total = len(texts)

    for start_idx in range(0, total, batch_size):
        end_idx = min(start_idx + batch_size, total)
        batch_texts = texts[start_idx:end_idx]

        encoded = tokenizer(
            batch_texts,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        embeddings = encode_fn(input_ids, attention_mask)
        all_embeddings.append(embeddings.cpu().numpy())

        if ((start_idx // batch_size + 1) % 200 == 0) or (end_idx == total):
            logger.info(f"  编码进度: {end_idx}/{total} ({100*end_idx/total:.1f}%)")

    return np.concatenate(all_embeddings, axis=0)


def build_faiss_index(embeddings):
    """构建 FAISS Flat 索引（精确检索，内积度量）"""
    import faiss

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    logger.info(f"FAISS Flat 索引已构建，包含 {index.ntotal} 个向量，维度={dim}")
    return index


def mine_negatives(
    index, query_embeddings, doc_ids, train_data,
    top_k, num_negatives, doc_meta
) -> List[Dict]:
    """
    为每个训练查询挖掘难负例

    参数:
        index: FAISS 索引
        query_embeddings: 查询向量 [num_queries, dim]
        doc_ids: 文档池 ID 列表
        train_data: 原始训练数据
        top_k: 检索候选数量
        num_negatives: 保留的难负例数量
        doc_meta: passage_id -> {title, text} 映射

    返回:
        更新后的训练数据列表
    """
    logger.info(f"开始挖掘难负例: top_k={top_k}, num_negatives={num_negatives}")

    query_embeddings = query_embeddings.astype(np.float32)
    scores, indices = index.search(query_embeddings, top_k)

    mined_data = []
    total_mined = 0

    for q_idx, item in enumerate(train_data):
        # 获取正例 passage_id 集合
        positive_ids: Set[str] = set()
        for ctx in item.get('positive_ctxs', []):
            pid = str(ctx.get('passage_id', ctx.get('psg_id', '')))
            if pid:
                positive_ids.add(pid)

        # 从检索结果中筛选难负例（排除正例）
        hard_negatives = []
        for rank_idx in range(top_k):
            doc_idx = indices[q_idx][rank_idx]
            if doc_idx < 0 or doc_idx >= len(doc_ids):
                continue

            pid = doc_ids[doc_idx]
            if pid in positive_ids:
                continue

            meta = doc_meta.get(pid, {})
            hard_negatives.append({
                'passage_id': pid,
                'title': meta.get('title', ''),
                'text': meta.get('text', ''),
            })

            if len(hard_negatives) >= num_negatives:
                break

        total_mined += len(hard_negatives)

        # 构建输出条目（保留原始结构，替换 hard_negative_ctxs）
        mined_item = {
            'question': item.get('question', ''),
            'answers': item.get('answers', []),
            'positive_ctxs': item.get('positive_ctxs', []),
            'hard_negative_ctxs': hard_negatives,
        }
        mined_data.append(mined_item)

        if (q_idx + 1) % 10000 == 0:
            logger.info(f"  挖掘进度: {q_idx + 1}/{len(train_data)}")

    avg_mined = total_mined / max(len(train_data), 1)
    logger.info(f"难负例挖掘完成: 平均每查询 {avg_mined:.1f} 个难负例")
    return mined_data


def run_mining(
    encoder,
    tokenizer,
    train_file: str,
    output_file: str,
    batch_size: int = 256,
    max_doc_length: int = 256,
    max_query_length: int = 64,
    top_k: int = 200,
    num_negatives: int = 50,
) -> str:
    """
    核心挖掘函数，可被 CLI 和 Trainer 调用

    使用给定的编码器，从训练数据文档池中为每个查询挖掘模型难负例。
    编码器应已加载到目标设备并处于 eval 模式。

    参数:
        encoder: BiEncoder 模型（已加载到设备）
        tokenizer: HuggingFace tokenizer
        train_file: 训练数据 JSON 文件路径
        output_file: 输出 JSON 文件路径
        batch_size: 编码批次大小
        max_doc_length: 文档最大 token 长度
        max_query_length: 查询最大 token 长度
        top_k: 每个查询检索的候选数量
        num_negatives: 每个查询保留的难负例数量

    返回:
        输出文件路径
    """
    logger.info(f"挖掘参数: train_file={train_file}, top_k={top_k}, num_negatives={num_negatives}")

    # 加载训练数据
    train_data = load_train_data(train_file)

    # 构建训练数据文档池
    doc_ids, doc_texts, doc_meta = build_document_pool(train_data)

    # 编码文档池
    logger.info("编码文档池...")
    start_time = time.time()
    doc_embeddings = encode_texts(
        encoder, tokenizer, doc_texts,
        batch_size=batch_size,
        max_length=max_doc_length,
        encode_fn_name='encode_document'
    )
    logger.info(f"文档编码完成，耗时 {time.time() - start_time:.1f} 秒，矩阵形状: {doc_embeddings.shape}")

    # 构建 FAISS 索引
    index = build_faiss_index(doc_embeddings)

    # 编码训练查询
    queries = [item.get('question', '') for item in train_data]
    logger.info(f"编码 {len(queries)} 个训练查询...")
    start_time = time.time()
    query_embeddings = encode_texts(
        encoder, tokenizer, queries,
        batch_size=batch_size,
        max_length=max_query_length,
        encode_fn_name='encode_query'
    )
    logger.info(f"查询编码完成，耗时 {time.time() - start_time:.1f} 秒")

    # 挖掘难负例
    mined_data = mine_negatives(
        index, query_embeddings, doc_ids, train_data,
        top_k=top_k,
        num_negatives=num_negatives,
        doc_meta=doc_meta
    )

    # 保存结果
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(mined_data, f, indent=2, ensure_ascii=False)
    logger.info(f"挖掘结果已保存: {output_path} ({len(mined_data)} 条样本)")

    return str(output_path)


def main():
    args = parse_args()

    from src.models.bi_encoder import BiEncoder

    logger.info("=" * 60)
    logger.info("Stage 3 难负例挖掘")
    logger.info(f"  编码器: {args.encoder_path}")
    logger.info(f"  训练数据: {args.train_file}")
    logger.info(f"  输出文件: {args.output_file}")
    logger.info(f"  top_k: {args.top_k}, num_negatives: {args.num_negatives}")
    logger.info("=" * 60)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 加载编码器
    logger.info("加载 BiEncoder 模型...")
    encoder = BiEncoder.from_pretrained(args.encoder_path)
    encoder.to(device)
    encoder.eval()
    tokenizer = AutoTokenizer.from_pretrained(encoder.model_name)

    # 执行挖掘
    run_mining(
        encoder=encoder,
        tokenizer=tokenizer,
        train_file=args.train_file,
        output_file=args.output_file,
        batch_size=args.batch_size,
        max_doc_length=args.max_doc_length,
        max_query_length=args.max_query_length,
        top_k=args.top_k,
        num_negatives=args.num_negatives,
    )

    logger.info("Stage 3 难负例挖掘完成！")


if __name__ == "__main__":
    main()