"""
编码服务（批量编码）

本模块提供高效的批量文本编码服务，支持大规模文档集合的向量化。

论文章节：第5章 - 系统实现
"""

import torch
import numpy as np
from typing import List, Optional, Union, Any, Iterator
from tqdm import tqdm

class EncoderService:
    """
    编码服务
    
    提供Query和Document的批量编码功能。
    
    Args:
        query_encoder: Query编码器
        doc_encoder: Document编码器（如果与query_encoder相同可以为None）
        tokenizer: Tokenizer实例
        device: 计算设备
        max_query_length (int): Query最大长度
        max_doc_length (int): Document最大长度
    """
    
    def __init__(
        self,
        query_encoder: Any,
        doc_encoder: Optional[Any] = None,
        tokenizer: Any = None,
        device: Optional[torch.device] = None,
        max_query_length: int = 64,
        max_doc_length: int = 256
    ):
        self.query_encoder = query_encoder
        self.doc_encoder = doc_encoder or query_encoder
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_query_length = max_query_length
        self.max_doc_length = max_doc_length
        
        self.query_encoder.to(self.device)
        self.doc_encoder.to(self.device)
        
    def encode_queries(
        self,
        queries: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        编码查询
        
        Args:
            queries: 查询文本或列表
            batch_size: 批次大小
            show_progress: 是否显示进度条
            
        Returns:
            查询向量 [num_queries, embedding_dim]
        """
        if isinstance(queries, str):
            queries = [queries]
            
        self.query_encoder.eval()
        all_embeddings = []
        
        iterator = range(0, len(queries), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding queries")
            
        with torch.no_grad():
            for start_idx in iterator:
                batch = queries[start_idx:start_idx + batch_size]
                
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_query_length,
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                embeddings = self.query_encoder(**inputs)
                all_embeddings.append(embeddings.cpu().numpy())
                
        return np.vstack(all_embeddings)
    
    def encode_documents(
        self,
        documents: Union[str, List[str]],
        batch_size: int = 64,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        编码文档
        
        Args:
            documents: 文档文本或列表
            batch_size: 批次大小
            show_progress: 是否显示进度条
            
        Returns:
            文档向量 [num_documents, embedding_dim]
        """
        if isinstance(documents, str):
            documents = [documents]
            
        self.doc_encoder.eval()
        all_embeddings = []
        
        iterator = range(0, len(documents), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding documents")
            
        with torch.no_grad():
            for start_idx in iterator:
                batch = documents[start_idx:start_idx + batch_size]
                
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_doc_length,
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                embeddings = self.doc_encoder(**inputs)
                all_embeddings.append(embeddings.cpu().numpy())
                
        return np.vstack(all_embeddings)
    
    def encode_documents_streaming(
        self,
        documents: Iterator[str],
        batch_size: int = 64
    ) -> Iterator[np.ndarray]:
        """
        流式编码文档
        
        适用于超大规模文档集合，避免一次性加载到内存。
        
        Args:
            documents: 文档迭代器
            batch_size: 批次大小
            
        Yields:
            文档向量批次
        """
        self.doc_encoder.eval()
        batch = []
        
        with torch.no_grad():
            for doc in documents:
                batch.append(doc)
                
                if len(batch) >= batch_size:
                    inputs = self.tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=self.max_doc_length,
                        return_tensors="pt"
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    embeddings = self.doc_encoder(**inputs)
                    yield embeddings.cpu().numpy()
                    batch = []
                    
            # 处理剩余文档
            if batch:
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_doc_length,
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                embeddings = self.doc_encoder(**inputs)
                yield embeddings.cpu().numpy()
                
    def compute_similarity(
        self,
        queries: Union[str, List[str]],
        documents: Union[str, List[str]]
    ) -> np.ndarray:
        """
        计算Query-Document相似度
        
        Args:
            queries: 查询文本
            documents: 文档文本
            
        Returns:
            相似度矩阵 [num_queries, num_documents]
        """
        query_embs = self.encode_queries(queries)
        doc_embs = self.encode_documents(documents)
        
        # 计算余弦相似度
        query_embs = query_embs / np.linalg.norm(query_embs, axis=1, keepdims=True)
        doc_embs = doc_embs / np.linalg.norm(doc_embs, axis=1, keepdims=True)
        
        return np.dot(query_embs, doc_embs.T)