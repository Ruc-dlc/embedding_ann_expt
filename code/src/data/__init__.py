"""
数据模块初始化

本模块负责数据集加载、预处理和难负例挖掘。

论文章节：第4章 4.2节 - 数据处理
"""

from .dataset import NQDataset, TriviaQADataset, ConcatRetrievalDataset, BaseDataset
from .dataloader import ThreeStageDataLoader, collate_fn
from .preprocessor import TextPreprocessor, QueryPreprocessor, DocumentPreprocessor
from .hard_negative_miner import HardNegativeMiner

__all__ = [
    'NQDataset',
    'TriviaQADataset',
    'ConcatRetrievalDataset',
    'BaseDataset',
    'ThreeStageDataLoader',
    'collate_fn',
    'TextPreprocessor',
    'QueryPreprocessor',
    'DocumentPreprocessor',
    'HardNegativeMiner',
]