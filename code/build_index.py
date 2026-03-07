#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
构建 FAISS HNSW 索引

从已编码的 embeddings.npy 构建 HNSW 索引。

用法:
    python build_index.py \
        --embeddings experiments/embeddings/w0.6_best_nq \
        --output experiments/indices/w0.6_best_nq
"""

import argparse
import json
import logging
import shutil
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser(description="构建 FAISS HNSW 索引")
    p.add_argument("--embeddings", type=str, required=True, help="嵌入目录 (含 embeddings.npy, doc_ids.txt)")
    p.add_argument("--output", type=str, required=True, help="索引输出目录")
    p.add_argument("--hnsw_m", type=int, default=32)
    p.add_argument("--ef_construction", type=int, default=200)
    args = p.parse_args()

    import faiss

    emb_dir = Path(args.embeddings)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 加载嵌入
    logger.info(f"加载嵌入: {emb_dir / 'embeddings.npy'}")
    embeddings = np.load(emb_dir / "embeddings.npy")
    logger.info(f"嵌入 shape: {embeddings.shape}, dtype: {embeddings.dtype}")

    dim = embeddings.shape[1]

    # 构建 HNSW 索引
    logger.info(f"构建 IndexHNSWFlat(dim={dim}, M={args.hnsw_m}, efConstruction={args.ef_construction})")
    index = faiss.IndexHNSWFlat(dim, args.hnsw_m)
    index.hnsw.efConstruction = args.ef_construction

    start = time.time()
    index.add(embeddings)
    build_time = time.time() - start
    logger.info(f"索引构建完成: {index.ntotal} 向量, 耗时 {build_time / 60:.1f} min")

    # 保存
    faiss.write_index(index, str(out_dir / "index.faiss"))
    shutil.copy(emb_dir / "doc_ids.txt", out_dir / "doc_ids.txt")

    config = {
        "index_type": "hnsw",
        "num_docs": int(index.ntotal),
        "dim": dim,
        "hnsw_m": args.hnsw_m,
        "ef_construction": args.ef_construction,
    }
    with open(out_dir / "index_config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"索引已保存: {out_dir}")


if __name__ == "__main__":
    main()
