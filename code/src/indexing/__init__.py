"""
索引模块初始化

本模块包含向量索引的封装，支持多种ANN索引类型。

论文章节：第4章 4.4节 - ANN索引
"""

from .faiss_wrapper import FaissIndex
from .hnsw_index import HNSWIndex
from .flat_index import FlatIndex
from .index_factory import IndexFactory, create_index

__all__ = [
    'FaissIndex',
    'HNSWIndex',
    'FlatIndex',
    'IndexFactory',
    'create_index',
]