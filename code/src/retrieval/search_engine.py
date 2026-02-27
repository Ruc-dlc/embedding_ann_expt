"""
端到端检索引擎

本模块实现完整的端到端检索引擎，包括索引构建、检索和重排序。

论文章节：第5章 - 检索系统架构
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import logging
import time

logger = logging.getLogger(__name__)

class SearchEngine:
    """
    搜索引擎
    
    提供完整的检索流程：
    1. 文档编码与索引构建
    2. 在线查询检索
    3. 可选的重排序
    
    Args:
        encoder_service: 编码服务
        index: 向量索引
        corpus: 文档语料库（可选）
    """
    
    def __init__(
        self,
        encoder_service: Any,
        index: Any,
        corpus: Optional[Dict[str, str]] = None
    ):
        self.encoder_service = encoder_service
        self.index = index
        self.corpus = corpus or {}
        
        self.doc_ids: List[str] = []
        self.is_indexed = False
        
    def build_index(
        self,
        documents: Optional[Dict[str, str]] = None,
        batch_size: int = 64,
        show_progress: bool = True
    ) -> None:
        """
        构建索引
        
        Args:
            documents: 文档字典 {doc_id: doc_text}
            batch_size: 编码批次大小
            show_progress: 是否显示进度
        """
        if documents is not None:
            self.corpus = documents
            
        if not self.corpus:
            raise ValueError("No documents provided for indexing")
            
        logger.info(f"Building index for {len(self.corpus)} documents")
        start_time = time.time()
        
        # 获取文档ID和文本
        self.doc_ids = list(self.corpus.keys())
        doc_texts = list(self.corpus.values())
        
        # 编码文档
        doc_embeddings = self.encoder_service.encode_documents(
            doc_texts,
            batch_size=batch_size,
            show_progress=show_progress
        )
        
        # 构建索引
        self.index.build(doc_embeddings)
        self.is_indexed = True
        
        elapsed = time.time() - start_time
        logger.info(f"Index built in {elapsed:.2f}s")
        
    def search(
        self,
        query: Union[str, List[str]],
        top_k: int = 10,
        return_text: bool = False
    ) -> List[Dict[str, Any]]:
        """
        搜索
        
        Args:
            query: 查询文本或列表
            top_k: 返回数量
            return_text: 是否返回文档文本
            
        Returns:
            检索结果列表
        """
        if not self.is_indexed:
            raise RuntimeError("Index not built. Call build_index() first.")
            
        is_single = isinstance(query, str)
        queries = [query] if is_single else query
        
        # 编码查询
        query_embeddings = self.encoder_service.encode_queries(queries)
        
        # 索引搜索
        start_time = time.time()
        distances, indices = self.index.search(query_embeddings, top_k)
        search_time = time.time() - start_time
        
        # 格式化结果
        results = []
        for i, (q_indices, q_distances) in enumerate(zip(indices, distances)):
            query_results = []
            for rank, (idx, dist) in enumerate(zip(q_indices, q_distances)):
                if idx < 0:
                    continue
                    
                doc_id = self.doc_ids[idx]
                result = {
                    'doc_id': doc_id,
                    'rank': rank + 1,
                    'distance': float(dist),
                    'score': float(-dist) if self.index.metric == 'l2' else float(dist)
                }
                
                if return_text and doc_id in self.corpus:
                    result['text'] = self.corpus[doc_id]
                    
                query_results.append(result)
            results.append(query_results)
            
        # 添加搜索时间统计
        if is_single:
            return results[0]
        return results
    
    def search_with_stats(
        self,
        query: Union[str, List[str]],
        top_k: int = 10
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """
        搜索并返回统计信息
        
        Args:
            query: 查询
            top_k: 返回数量
            
        Returns:
            (检索结果, 统计信息)
        """
        is_single = isinstance(query, str)
        queries = [query] if is_single else query
        
        # 编码计时
        encode_start = time.time()
        query_embeddings = self.encoder_service.encode_queries(queries)
        encode_time = time.time() - encode_start
        
        # 搜索计时
        search_start = time.time()
        distances, indices = self.index.search(query_embeddings, top_k)
        search_time = time.time() - search_start
        
        # 格式化结果
        results = []
        for i, (q_indices, q_distances) in enumerate(zip(indices, distances)):
            query_results = []
            for rank, (idx, dist) in enumerate(zip(q_indices, q_distances)):
                if idx < 0:
                    continue
                result = {
                    'doc_id': self.doc_ids[idx],
                    'rank': rank + 1,
                    'distance': float(dist),
                    'score': float(-dist) if self.index.metric == 'l2' else float(dist)
                }
                query_results.append(result)
            results.append(query_results)
            
        stats = {
            'encode_time_ms': encode_time * 1000,
            'search_time_ms': search_time * 1000,
            'total_time_ms': (encode_time + search_time) * 1000,
            'num_queries': len(queries),
            'qps': len(queries) / (encode_time + search_time)
        }
        
        if is_single:
            return results[0], stats
        return results, stats
    
    def save(self, path: str) -> None:
        """
        保存搜索引擎
        
        Args:
            path: 保存目录
        """
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存索引
        self.index.save(str(save_dir / "index.faiss"))
        
        # 保存文档ID
        with open(save_dir / "doc_ids.txt", 'w') as f:
            for doc_id in self.doc_ids:
                f.write(f"{doc_id}\n")
                
        logger.info(f"Search engine saved to {path}")
        
    def load_index(self, path: str) -> None:
        """
        加载索引
        
        Args:
            path: 索引目录
        """
        load_dir = Path(path)
        
        # 加载索引
        self.index.load(str(load_dir / "index.faiss"))
        
        # 加载文档ID
        with open(load_dir / "doc_ids.txt", 'r') as f:
            self.doc_ids = [line.strip() for line in f]
            
        self.is_indexed = True
        logger.info(f"Search engine loaded from {path}")