"""
难负例挖掘器

本模块实现难负例挖掘策略：
- BM25静态难负例：基于词汇匹配（用于开源完整性，实际训练中阶段2使用DPR预存难负例）
- 模型动态难负例：基于当前模型的FAISS索引检索（阶段3使用）

论文章节：第4章 4.2节 - 难负例挖掘
"""

import math
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any, Set

import numpy as np
import torch

logger = logging.getLogger(__name__)


class BM25Index:
    """
    BM25稀疏检索索引

    基于BM25算法的轻量级检索索引，用于静态难负例挖掘。

    参数:
        k1: BM25参数k1，控制词频饱和度
        b: BM25参数b，控制文档长度归一化
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.num_docs = 0
        self.avg_doc_length = 0.0
        self.doc_lengths: List[int] = []
        self.doc_freqs: Dict[str, int] = defaultdict(int)
        # 倒排索引: token -> [(doc_idx, term_freq)]
        self.inverted_index: Dict[str, List[Tuple[int, int]]] = defaultdict(list)

    def _tokenize(self, text: str) -> List[str]:
        """简单空格分词（小写化）"""
        return text.lower().split()

    def build(self, documents: List[str]) -> None:
        """
        构建BM25索引

        参数:
            documents: 文档文本列表
        """
        self.num_docs = len(documents)
        logger.info(f"正在构建BM25索引，文档数: {self.num_docs}")

        total_length = 0

        for doc_idx, doc in enumerate(documents):
            tokens = self._tokenize(doc)
            doc_length = len(tokens)
            self.doc_lengths.append(doc_length)
            total_length += doc_length

            # 统计词频
            term_freqs: Dict[str, int] = defaultdict(int)
            for token in tokens:
                term_freqs[token] += 1

            # 更新倒排索引和文档频率
            for token, freq in term_freqs.items():
                self.inverted_index[token].append((doc_idx, freq))
                self.doc_freqs[token] += 1

            if (doc_idx + 1) % 100000 == 0:
                logger.info(f"  已索引 {doc_idx + 1}/{self.num_docs} 篇文档")

        self.avg_doc_length = total_length / max(self.num_docs, 1)
        logger.info(f"BM25索引构建完成，平均文档长度: {self.avg_doc_length:.1f}")

    def _compute_bm25_score(
        self,
        term_freq: int,
        doc_freq: int,
        doc_length: int
    ) -> float:
        """
        计算单个词项的BM25分数

        参数:
            term_freq: 词频
            doc_freq: 文档频率
            doc_length: 文档长度

        返回:
            BM25分数
        """
        # IDF部分
        idf = math.log(
            (self.num_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0
        )

        # TF归一化部分
        tf_norm = (term_freq * (self.k1 + 1)) / (
            term_freq + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
        )

        return idf * tf_norm

    def search(self, query: str, top_k: int = 100) -> List[Tuple[int, float]]:
        """
        BM25检索

        参数:
            query: 查询文本
            top_k: 返回数量

        返回:
            (文档索引, BM25分数) 列表，按分数降序
        """
        query_tokens = self._tokenize(query)

        # 累计每个文档的BM25分数
        doc_scores: Dict[int, float] = defaultdict(float)

        for token in query_tokens:
            if token not in self.inverted_index:
                continue

            df = self.doc_freqs[token]
            for doc_idx, tf in self.inverted_index[token]:
                score = self._compute_bm25_score(tf, df, self.doc_lengths[doc_idx])
                doc_scores[doc_idx] += score

        # 排序取top_k
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:top_k]


class HardNegativeMiner:
    """
    难负例挖掘器

    支持两种挖掘策略：
    - BM25静态负例：基于词汇匹配的高分但非正例文档
    - 模型动态负例：使用当前模型编码后通过FAISS索引检索

    注意：阶段2的难负例已在DPR数据中预存（hard_negative_ctxs字段），
    本模块主要用于阶段3的动态挖掘以及开源代码的完整性。

    参数:
        corpus_texts: 文档文本列表
        corpus_ids: 文档ID列表（与corpus_texts一一对应）
        num_negatives: 每个查询的负例数量
        bm25_candidates: BM25候选数量
    """

    def __init__(
        self,
        corpus_texts: Optional[List[str]] = None,
        corpus_ids: Optional[List[str]] = None,
        num_negatives: int = 7,
        bm25_candidates: int = 100
    ):
        self.corpus_texts = corpus_texts or []
        self.corpus_ids = corpus_ids or []
        self.num_negatives = num_negatives
        self.bm25_candidates = bm25_candidates

        self.bm25_index: Optional[BM25Index] = None
        self.doc_embeddings: Optional[np.ndarray] = None
        self.faiss_index: Optional[Any] = None

    def build_bm25_index(self) -> None:
        """构建BM25索引"""
        if not self.corpus_texts:
            logger.warning("语料库为空，无法构建BM25索引")
            return

        self.bm25_index = BM25Index()
        self.bm25_index.build(self.corpus_texts)

    def mine_bm25_negatives(
        self,
        query: str,
        positive_ids: Set[str],
        num_negatives: Optional[int] = None
    ) -> List[int]:
        """
        使用BM25挖掘难负例

        检索高分但非正例的文档作为难负例。

        参数:
            query: 查询文本
            positive_ids: 正例文档ID集合
            num_negatives: 需要的负例数量

        返回:
            难负例的语料库索引列表
        """
        if self.bm25_index is None:
            logger.warning("BM25索引未构建，请先调用 build_bm25_index()")
            return []

        if num_negatives is None:
            num_negatives = self.num_negatives

        # BM25检索
        candidates = self.bm25_index.search(query, top_k=self.bm25_candidates)

        # 过滤正例
        negatives = []
        for doc_idx, score in candidates:
            if doc_idx < len(self.corpus_ids):
                doc_id = self.corpus_ids[doc_idx]
                if doc_id not in positive_ids:
                    negatives.append(doc_idx)
                    if len(negatives) >= num_negatives:
                        break

        return negatives

    def mine_dynamic_negatives(
        self,
        query_embedding: np.ndarray,
        positive_ids: Set[str],
        num_negatives: Optional[int] = None
    ) -> List[int]:
        """
        使用当前模型动态挖掘难负例

        通过FAISS索引在向量空间中找到与query最相似但非正例的文档。

        参数:
            query_embedding: 查询向量 [embedding_dim]
            positive_ids: 正例文档ID集合
            num_negatives: 需要的负例数量

        返回:
            难负例的语料库索引列表
        """
        if self.faiss_index is None or self.doc_embeddings is None:
            logger.warning("FAISS索引未构建，请先调用 update_doc_embeddings()")
            return []

        if num_negatives is None:
            num_negatives = self.num_negatives

        # FAISS检索（多检索一些候选以便过滤正例后还有足够数量）
        search_k = num_negatives * 3
        query_vec = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self.faiss_index.search(query_vec, search_k)

        # 过滤正例
        negatives = []
        for idx in indices[0]:
            if idx < 0:
                continue
            if idx < len(self.corpus_ids):
                doc_id = self.corpus_ids[idx]
                if doc_id not in positive_ids:
                    negatives.append(int(idx))
                    if len(negatives) >= num_negatives:
                        break

        return negatives

    @torch.no_grad()
    def update_doc_embeddings(
        self,
        encoder: Any,
        tokenizer: Any,
        batch_size: int = 256,
        max_doc_length: int = 256
    ) -> None:
        """
        使用当前模型重新编码全部文档并重建FAISS索引

        参数:
            encoder: BiEncoder模型
            tokenizer: HuggingFace tokenizer
            batch_size: 编码批次大小
            max_doc_length: 文档最大token长度
        """
        try:
            import faiss
        except ImportError:
            logger.error("请安装 faiss-cpu: conda install -c conda-forge faiss-cpu")
            return

        if not self.corpus_texts:
            logger.warning("语料库为空，无法更新文档向量")
            return

        logger.info(f"开始编码语料库，共 {len(self.corpus_texts)} 篇文档")

        encoder.eval()
        device = next(encoder.parameters()).device
        all_embeddings = []

        for start_idx in range(0, len(self.corpus_texts), batch_size):
            end_idx = min(start_idx + batch_size, len(self.corpus_texts))
            batch_texts = self.corpus_texts[start_idx:end_idx]

            # Tokenize
            encoded = tokenizer(
                batch_texts,
                max_length=max_doc_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)

            # 编码
            embeddings = encoder.encode_document(input_ids, attention_mask)
            all_embeddings.append(embeddings.cpu().numpy())

            if (start_idx // batch_size + 1) % 100 == 0:
                logger.info(f"  已编码 {end_idx}/{len(self.corpus_texts)} 篇文档")

        self.doc_embeddings = np.concatenate(all_embeddings, axis=0)
        logger.info(f"文档编码完成，向量矩阵形状: {self.doc_embeddings.shape}")

        # 重建FAISS索引（使用内积，因为向量已L2归一化）
        dim = self.doc_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(self.doc_embeddings.astype(np.float32))
        logger.info(f"FAISS索引已重建，索引大小: {self.faiss_index.ntotal}")