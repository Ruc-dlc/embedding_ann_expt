"""
索引工厂类

本模块提供索引的工厂模式创建接口，支持通过配置创建不同类型的索引。

论文章节：第5章 - 实验实现
"""

from typing import Dict, Any, Optional, Union
from .faiss_wrapper import FaissIndex
from .hnsw_index import HNSWIndex
from .ivf_index import IVFIndex
from .flat_index import FlatIndex

class IndexFactory:
    """
    索引工厂
    
    统一的索引创建接口，根据配置创建相应类型的索引。
    
    支持的索引类型：
    - flat: 精确索引
    - hnsw: HNSW图索引
    - ivf: IVF倒排索引
    - ivf_pq: IVF+PQ量化索引
    """
    
    # 注册的索引类型
    _index_types = {
        'flat': FlatIndex,
        'hnsw': HNSWIndex,
        'ivf': IVFIndex,
        'ivf_pq': IVFIndex,
    }
    
    @classmethod
    def create(
        cls,
        index_type: str,
        dimension: int,
        metric: str = "l2",
        **kwargs
    ) -> FaissIndex:
        """
        创建索引
        
        Args:
            index_type: 索引类型
            dimension: 向量维度
            metric: 距离度量
            **kwargs: 索引特定参数
            
        Returns:
            索引实例
            
        Raises:
            ValueError: 未知的索引类型
        """
        index_type = index_type.lower()
        
        if index_type not in cls._index_types:
            raise ValueError(
                f"Unknown index type: {index_type}. "
                f"Supported types: {list(cls._index_types.keys())}"
            )
            
        index_class = cls._index_types[index_type]
        
        # 特殊处理IVF+PQ
        if index_type == 'ivf_pq':
            kwargs['use_pq'] = True
            
        return index_class(dimension=dimension, metric=metric, **kwargs)
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> FaissIndex:
        """
        从配置字典创建索引
        
        Args:
            config: 配置字典，需包含 'type' 和 'dimension' 字段
            
        Returns:
            索引实例
        """
        config = config.copy()
        index_type = config.pop('type', 'flat')
        dimension = config.pop('dimension')
        metric = config.pop('metric', 'l2')
        
        return cls.create(index_type, dimension, metric, **config)
    
    @classmethod
    def register(cls, name: str, index_class: type) -> None:
        """
        注册新的索引类型
        
        Args:
            name: 索引类型名称
            index_class: 索引类
        """
        cls._index_types[name.lower()] = index_class
        
    @classmethod
    def list_types(cls) -> list:
        """
        列出所有支持的索引类型
        
        Returns:
            索引类型列表
        """
        return list(cls._index_types.keys())

def create_index(
    index_type: str,
    dimension: int,
    metric: str = "l2",
    **kwargs
) -> FaissIndex:
    """
    创建索引（便捷函数）
    
    IndexFactory.create 的便捷包装。
    
    Args:
        index_type: 索引类型 ("flat", "hnsw", "ivf", "ivf_pq")
        dimension: 向量维度
        metric: 距离度量 ("l2" 或 "ip")
        **kwargs: 索引特定参数
        
    Returns:
        索引实例
        
    Examples:
        >>> # 创建HNSW索引
        >>> # 创建IVF索引
    """
    return IndexFactory.create(index_type, dimension, metric, **kwargs)

def create_index_from_string(index_string: str, dimension: int) -> FaissIndex:
    """
    从FAISS索引字符串创建索引
    
    支持FAISS的索引工厂字符串格式。
    
    Args:
        index_string: FAISS索引字符串（如 "HNSW32", "IVF100,Flat"）
        dimension: 向量维度
        
    Returns:
        索引实例
    """
    import faiss
    
    index = faiss.index_factory(dimension, index_string)
    
    # 包装为FaissIndex
    wrapper = FaissIndex.__new__(FaissIndex)
    wrapper.dimension = dimension
    wrapper.index = index
    wrapper.is_trained = False
    wrapper.num_vectors = 0
    wrapper.metric = "l2"
    
    return wrapper