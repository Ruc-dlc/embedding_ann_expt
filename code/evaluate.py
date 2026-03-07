#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试集检索评估

使用 answer string matching 评估检索效果:
加载 encoder → 编码 test queries → FAISS 搜索 → 答案匹配 → Accuracy@K

用法:
    python evaluate.py \
        --checkpoint experiments/models/w0.6/best_nq \
        --index experiments/indices/w0.6_best_nq \
        --test_file data_set/NQ/nq-test.csv \
        --corpus experiments/corpus/corpus_5m.tsv \
        --output experiments/results/nq/w0.6.json
"""

import argparse
import csv
import json
import logging
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoTokenizer

from src.models import BiEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_test_csv(test_file: str) -> tuple:
    """加载测试集 CSV (query\\tanswers)

    answers 列格式为 Python list 字面量, 如:  ['answer1', 'answer2']
    使用 ast.literal_eval 安全解析。
    """
    queries = []
    answers = []
    with open(test_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) >= 2:
                queries.append(row[0])
                raw = row[1].strip()
                try:
                    import ast
                    ans_list = ast.literal_eval(raw)
                    if isinstance(ans_list, list):
                        answers.append([str(a).strip() for a in ans_list])
                    else:
                        answers.append([str(ans_list).strip()])
                except (ValueError, SyntaxError):
                    # 兜底: 简单逗号分割
                    answers.append([a.strip() for a in raw.split(",")])
    logger.info(f"测试集: {len(queries)} 条查询")
    return queries, answers


def load_corpus_texts(corpus_path: str) -> Dict[str, str]:
    """加载语料库文本 (doc_id -> text)"""
    logger.info(f"加载语料库文本: {corpus_path}")
    texts = {}
    with open(corpus_path, "r", encoding="utf-8") as f:
        f.readline()  # 表头
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                doc_id = parts[0]
                text = parts[1]
                title = parts[2] if len(parts) > 2 else ""
                texts[doc_id] = f"{title} {text}".strip() if title else text
    logger.info(f"语料库: {len(texts)} 篇文档")
    return texts


def load_index(index_path: str):
    """加载 FAISS 索引和 doc_ids"""
    import faiss

    idx_dir = Path(index_path)
    index = faiss.read_index(str(idx_dir / "index.faiss"))
    logger.info(f"索引: {index.ntotal} 向量")

    doc_ids = []
    with open(idx_dir / "doc_ids.txt", "r", encoding="utf-8") as f:
        for line in f:
            doc_ids.append(line.strip())

    return index, doc_ids


@torch.no_grad()
def encode_queries(model, tokenizer, queries: List[str], max_len: int, batch_size: int, device) -> np.ndarray:
    """批量编码查询"""
    model.eval()
    all_emb = []
    for start in range(0, len(queries), batch_size):
        batch = queries[start : start + batch_size]
        encoded = tokenizer(batch, max_length=max_len, padding="max_length", truncation=True, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.cuda.amp.autocast(enabled=device.type == "cuda", dtype=torch.float16):
            emb = model.encode_query(input_ids, attention_mask)
        all_emb.append(emb.cpu().numpy())

    return np.concatenate(all_emb, axis=0).astype(np.float32)


def answer_in_text(text: str, answers: List[str]) -> bool:
    """检查文本是否包含任一答案 (case-insensitive)"""
    text_lower = text.lower()
    return any(a.strip().lower() in text_lower for a in answers)


def compute_accuracy(retrieved_ids, doc_ids, answers_list, corpus_texts, k_values):
    """计算 Top-K Accuracy (answer match)

    对每个 query, 检索 top-K passage, 若至少一个 passage 包含答案则命中 (1),
    否则未命中 (0), 所有 query 取平均。
    """
    results = {}
    for k in k_values:
        hits = 0
        total = 0
        for i, (ret_indices, ans_list) in enumerate(zip(retrieved_ids, answers_list)):
            if not ans_list:
                continue
            total += 1
            for idx in ret_indices[:k]:
                if 0 <= idx < len(doc_ids):
                    doc_text = corpus_texts.get(doc_ids[idx], "")
                    if answer_in_text(doc_text, ans_list):
                        hits += 1
                        break
        results[f"accuracy@{k}"] = hits / max(total, 1)
    return results


def main():
    p = argparse.ArgumentParser(description="测试集检索评估")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--index", type=str, required=True)
    p.add_argument("--test_file", type=str, required=True)
    p.add_argument("--corpus", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--ef_search", type=int, default=128)
    p.add_argument("--top_k", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--max_query_len", type=int, default=64)
    args = p.parse_args()

    import faiss

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载
    model = BiEncoder.from_pretrained(args.checkpoint)
    model.eval()
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model.model_name)

    index, doc_ids = load_index(args.index)
    queries, answers = load_test_csv(args.test_file)
    corpus_texts = load_corpus_texts(args.corpus)

    # 设置 efSearch
    if hasattr(index, "hnsw"):
        index.hnsw.efSearch = args.ef_search

    # 编码查询
    logger.info("编码查询...")
    q_emb = encode_queries(model, tokenizer, queries, args.max_query_len, args.batch_size, device)

    # 检索
    logger.info(f"检索 Top-{args.top_k} (efSearch={args.ef_search})...")
    start = time.time()
    _, indices = index.search(q_emb, args.top_k)
    search_time = time.time() - start
    logger.info(f"检索完成: {search_time:.2f}s")

    # 评估
    results = compute_accuracy(indices, doc_ids, answers, corpus_texts, [5, 10, 20, 50, 100])
    results["ef_search"] = args.ef_search
    results["num_queries"] = len(queries)

    # 保存
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("=" * 50)
    for k, v in results.items():
        logger.info(f"  {k}: {v}")
    logger.info(f"结果已保存: {out_path}")


if __name__ == "__main__":
    main()
