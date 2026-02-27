"""
检索器主类

本模块实现核心检索功能，整合编码器和索引进行端到端检索。

论文章节：第5章 - 检索系统
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path

class Retriever:
    """
    检索器
    
    整合Query编码器和向量索引，提供完整的检索接口。
    
    Args:
        encoder: Query编码器
        index: 向量索引
        tokenizer: Tokenizer实例
        device: 计算设备
    """
    
    def __init__(
        self,
        encoder: Any,
        index: Any,
        tokenizer: Any,
        device: Optional[torch.device] = None
    ):
        self.encoder = encoder
        self.index = index
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.encoder.to(self.device)
        self.encoder.eval()
        
        # 文档ID映射
        self.doc_ids: List[str] = []
        
    def encode_query(self, query: Union[str, List[str]]) -> np.ndarray:
        """
        编码查询
        
        Args:
            query: 查询文本或查询列表
            
        Returns:
            查询向量 [num_queries, embedding_dim]
        """
        if isinstance(query, str):
            query = [query]
            
        with torch.no_grad():
            inputs = self.tokenizer(
                query,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            embeddings = self.encoder(**inputs)
            
        return embeddings.cpu().numpy()
    
    def search(
        self,
        query: Union[str, List[str]],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        检索相关文档
        
        Args:
            query: 查询文本或查询列表
            top_k: 返回的文档数量
            
        Returns:
            检索结果列表，每个结果包含 doc_id, score, rank
        """
        # 编码查询
        query_embeddings = self.encode_query(query)
        
        # 索引搜索
        distances, indices = self.index.search(query_embeddings, top_k)
        
        # 格式化结果
        results = []
        is_single = isinstance(query, str)
        num_queries = 1 if is_single else len(query)
        
        for i in range(num_queries):
            query_results = []
            for rank, (idx, dist) in enumerate(zip(indices[i], distances[i])):
                if idx < 0:
                    continue
                    
                result = {
                    'doc_id': self.doc_ids[idx] if self.doc_ids else str(idx),
                    'index': int(idx),
                    'score': float(-dist) if self.index.metric == 'l2' else float(dist),
                    'distance': float(dist),
                    'rank': rank + 1
                }
                query_results.append(result)
            results.append(query_results)
            
        return results[0] if is_single else results
    
    def batch_search(
        self,
        queries: List[str],
        top_k: int = 10,
        batch_size: int = 32
    ) -> List[List[Dict[str, Any]]]:
        """
        批量检索
        
        Args:
            queries: 查询列表
            top_k: 返回数量
            batch_size: 批次大小
            
        Returns:
            每个查询的检索结果
        """
        all_results = []
        
        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i:i + batch_size]
            batch_results = self.search(batch_queries, top_k)
            all_results.extend(batch_results)
            
        return all_results
    
    def set_doc_ids(self, doc_ids: List[str]) -> None:
        """
        设置文档ID映射
        
        Args:
            doc_ids: 文档ID列表，与索引中的向量顺序对应
        """
        self.doc_ids = doc_ids
        
    def save(self, path: str) -> None:
        """
        保存检索器
        
        Args:
            path: 保存目录
        """
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存编码器
        torch.save(self.encoder.state_dict(), save_dir / "encoder.pt")
        
        # 保存索引
        self.index.save(str(save_dir / "index.faiss"))
        
        # 保存文档ID
        with open(save_dir / "doc_ids.txt", 'w') as f:
            for doc_id in self.doc_ids:
                f.write(f"{doc_id}\n")
                
    @classmethod
    def load(cls, path: str, encoder: Any, tokenizer: Any) -> "Retriever":
        """
        加载检索器
        
        Args:
            path: 保存目录
            encoder: 编码器实例
            tokenizer: Tokenizer实例
            
        Returns:
            检索器实例
        """
        # TODO: 实现加载逻辑
        pass