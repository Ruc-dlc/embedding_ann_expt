"""
模型模块初始化

本模块包含双塔编码器和相关组件的定义。

论文章节：第4章 4.1节 - 模型架构
"""

from .bi_encoder import BiEncoder
from .query_encoder import QueryEncoder
from .doc_encoder import DocEncoder
from .pooling import PoolingLayer, CLSPooling, MeanPooling, MaxPooling

__all__ = [
    'BiEncoder',
    'QueryEncoder',
    'DocEncoder',
    'PoolingLayer',
    'CLSPooling',
    'MeanPooling',
    'MaxPooling',
]