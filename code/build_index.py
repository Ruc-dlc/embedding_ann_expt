"""
FAISS 索引构建脚本

从编码好的 passage embeddings 构建 4 种 FAISS 索引：
  - Flat (暴力精确搜索)
  - HNSW (M=32, efConstruction=200)
  - IVF (nlist=4096)
  - IVF-PQ (nlist=4096, m=48, nbits=8)

用法：
  python build_index.py --embeddings_dir ./embeddings/dacl-dr --index_type all
  python build_index.py --embeddings_dir ./embeddings/dpr --index_type hnsw

参考：
  - experiments.md 第六节
"""

import argparse
import logging
import os
import time

import faiss
import numpy as np

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(description="Build FAISS Index")
    parser.add_argument("--embeddings_dir", type=str, required=True,
                        help="包含 passage_embeddings.npy 的目录")
    parser.add_argument("--index_type", type=str, required=True,
                        choices=["flat", "hnsw", "ivf", "ivf_pq", "all"])
    parser.add_argument("--output_dir", type=str, default=None,
                        help="索引输出目录（默认与 embeddings_dir 同级 indexes/ 目录）")

    # HNSW 参数
    parser.add_argument("--hnsw_m", type=int, default=32)
    parser.add_argument("--hnsw_ef_construction", type=int, default=200)

    # IVF 参数
    parser.add_argument("--ivf_nlist", type=int, default=4096)

    # IVF-PQ 参数
    parser.add_argument("--pq_m", type=int, default=48, help="PQ 子向量数")
    parser.add_argument("--pq_nbits", type=int, default=8, help="PQ 每子向量 bits")

    # IVF 训练
    parser.add_argument("--ivf_train_size", type=int, default=1000000,
                        help="IVF 聚类训练采样数")

    return parser.parse_args()


def load_embeddings(embeddings_dir):
    """加载 passage embeddings（若 float16 则转 float32）。"""
    emb_path = os.path.join(embeddings_dir, "passage_embeddings.npy")
    logger.info("Loading embeddings from %s ...", emb_path)
    embs = np.load(emb_path)
    if embs.dtype == np.float16:
        logger.info("Converting float16 -> float32 for FAISS")
        embs = embs.astype(np.float32)
    logger.info("Embeddings shape: %s, dtype: %s", embs.shape, embs.dtype)
    return embs


def build_flat(embs, dim):
    """构建 Flat (暴力精确) 索引。"""
    logger.info("Building Flat index (dim=%d)...", dim)
    # 向量已 L2 归一化，用 IP 等价于 cosine/L2
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    logger.info("Flat index built: %d vectors", index.ntotal)
    return index


def build_hnsw(embs, dim, M, ef_construction):
    """构建 HNSW 索引（Inner Product metric，与 Flat/IVF 一致）。"""
    logger.info("Building HNSW index (dim=%d, M=%d, efConstruction=%d, metric=IP)...", dim, M, ef_construction)
    index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    t = time.time()
    index.add(embs)
    logger.info("HNSW index built: %d vectors, %.1f seconds", index.ntotal, time.time() - t)
    return index


def build_ivf(embs, dim, nlist, train_size):
    """构建 IVF (倒排) 索引。"""
    logger.info("Building IVF index (dim=%d, nlist=%d)...", dim, nlist)
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

    # 训练
    n = embs.shape[0]
    if n > train_size:
        train_indices = np.random.choice(n, train_size, replace=False)
        train_data = embs[train_indices]
    else:
        train_data = embs

    logger.info("Training IVF with %d vectors...", train_data.shape[0])
    t = time.time()
    index.train(train_data)
    logger.info("IVF training done: %.1f seconds", time.time() - t)

    logger.info("Adding vectors to IVF index...")
    t = time.time()
    index.add(embs)
    logger.info("IVF index built: %d vectors, %.1f seconds", index.ntotal, time.time() - t)
    return index


def build_ivf_pq(embs, dim, nlist, pq_m, pq_nbits, train_size):
    """构建 IVF-PQ (倒排+乘积量化) 索引。"""
    logger.info("Building IVF-PQ index (dim=%d, nlist=%d, m=%d, nbits=%d)...",
                dim, nlist, pq_m, pq_nbits)
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, pq_m, pq_nbits, faiss.METRIC_INNER_PRODUCT)

    # 训练
    n = embs.shape[0]
    if n > train_size:
        train_indices = np.random.choice(n, train_size, replace=False)
        train_data = embs[train_indices]
    else:
        train_data = embs

    logger.info("Training IVF-PQ with %d vectors...", train_data.shape[0])
    t = time.time()
    index.train(train_data)
    logger.info("IVF-PQ training done: %.1f seconds", time.time() - t)

    logger.info("Adding vectors to IVF-PQ index...")
    t = time.time()
    index.add(embs)
    logger.info("IVF-PQ index built: %d vectors, %.1f seconds", index.ntotal, time.time() - t)
    return index


def save_index(index, output_dir, name):
    """保存 FAISS 索引到磁盘。"""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "%s.index" % name)
    faiss.write_index(index, path)
    size_gb = os.path.getsize(path) / 1e9
    logger.info("Saved index to %s (%.2f GB)", path, size_gb)


def main():
    args = get_args()
    embs = load_embeddings(args.embeddings_dir)
    dim = embs.shape[1]

    output_dir = args.output_dir or os.path.join(args.embeddings_dir, "indexes")

    types_to_build = []
    if args.index_type == "all":
        types_to_build = ["flat", "hnsw", "ivf", "ivf_pq"]
    else:
        types_to_build = [args.index_type]

    for idx_type in types_to_build:
        logger.info("=" * 50)

        if idx_type == "flat":
            index = build_flat(embs, dim)
            save_index(index, output_dir, "flat")

        elif idx_type == "hnsw":
            index = build_hnsw(embs, dim, args.hnsw_m, args.hnsw_ef_construction)
            save_index(index, output_dir, "hnsw_M%d" % args.hnsw_m)

        elif idx_type == "ivf":
            index = build_ivf(embs, dim, args.ivf_nlist, args.ivf_train_size)
            save_index(index, output_dir, "ivf_nlist%d" % args.ivf_nlist)

        elif idx_type == "ivf_pq":
            index = build_ivf_pq(embs, dim, args.ivf_nlist, args.pq_m, args.pq_nbits, args.ivf_train_size)
            save_index(index, output_dir, "ivf_pq_nlist%d_m%d" % (args.ivf_nlist, args.pq_m))

        del index  # 释放内存

    logger.info("All indexes built successfully.")


if __name__ == "__main__":
    main()
