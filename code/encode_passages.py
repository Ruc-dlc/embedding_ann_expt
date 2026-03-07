#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
编码 passage 嵌入

用训练好的 BiEncoder 的 doc_encoder 编码 corpus_5m.tsv 中的所有 passage,
将嵌入存为 numpy 文件, 供后续建索引使用。

用法:
    python encode_passages.py \
        --checkpoint experiments/models/w0.6/best_nq \
        --corpus experiments/corpus/corpus_5m.tsv \
        --output experiments/embeddings/w0.6_best_nq
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer

from src.models import BiEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def iter_corpus(corpus_path: str, max_docs: int = None) -> Iterator[Tuple[str, str]]:
    """逐行读取 TSV 语料, 返回 (doc_id, doc_text)"""
    with open(corpus_path, "r", encoding="utf-8") as f:
        f.readline()  # 跳过表头
        count = 0
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            doc_id = parts[0]
            text = parts[1]
            title = parts[2] if len(parts) > 2 else ""
            doc_text = f"{title} {text}".strip() if title else text
            yield doc_id, doc_text
            count += 1
            if max_docs and count >= max_docs:
                break


def iter_batches(corpus_path: str, batch_size: int, max_docs: int = None):
    """按 batch 切分语料"""
    ids, texts = [], []
    for doc_id, doc_text in iter_corpus(corpus_path, max_docs):
        ids.append(doc_id)
        texts.append(doc_text)
        if len(ids) >= batch_size:
            yield ids, texts
            ids, texts = [], []
    if ids:
        yield ids, texts


@torch.no_grad()
def encode_batch(model, tokenizer, texts: List[str], max_len: int, device) -> np.ndarray:
    """编码一个 batch 的文本"""
    encoded = tokenizer(texts, max_length=max_len, padding="max_length", truncation=True, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.cuda.amp.autocast(enabled=device.type == "cuda", dtype=torch.float16):
        emb = model.encode_document(input_ids, attention_mask)

    return emb.cpu().numpy().astype(np.float32)


def main():
    p = argparse.ArgumentParser(description="编码 passage 嵌入")
    p.add_argument("--checkpoint", type=str, required=True, help="BiEncoder checkpoint 目录")
    p.add_argument("--corpus", type=str, required=True, help="语料库 TSV 文件")
    p.add_argument("--output", type=str, required=True, help="输出目录")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--max_doc_len", type=int, default=256)
    p.add_argument("--max_docs", type=int, default=None, help="最大文档数(调试用)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"设备: {device}")

    # 加载模型
    logger.info(f"加载模型: {args.checkpoint}")
    model = BiEncoder.from_pretrained(args.checkpoint)
    model.eval()
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model.model_name)

    # 输出目录
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 编码
    logger.info(f"开始编码: {args.corpus}")
    start = time.time()
    all_embeddings = []
    all_ids = []
    total = 0

    for batch_idx, (ids, texts) in enumerate(iter_batches(args.corpus, args.batch_size, args.max_docs), 1):
        emb = encode_batch(model, tokenizer, texts, args.max_doc_len, device)
        all_embeddings.append(emb)
        all_ids.extend(ids)
        total += len(ids)

        if batch_idx % 500 == 0:
            elapsed = time.time() - start
            logger.info(f"  {total} docs 编码完成, {elapsed / 60:.1f} min")

    elapsed = time.time() - start
    logger.info(f"编码完成: {total} docs, {elapsed / 60:.1f} min")

    # 拼接并保存
    embeddings = np.concatenate(all_embeddings, axis=0)
    logger.info(f"嵌入 shape: {embeddings.shape}")

    np.save(out_dir / "embeddings.npy", embeddings)
    with open(out_dir / "doc_ids.txt", "w", encoding="utf-8") as f:
        for did in all_ids:
            f.write(f"{did}\n")

    logger.info(f"已保存至: {out_dir}")


if __name__ == "__main__":
    main()
