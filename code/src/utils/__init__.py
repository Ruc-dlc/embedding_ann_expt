"""
工具模块初始化

包含数据预处理、分词器、答案匹配等评估/训练所需的工具函数。
"""

from .data_utils import normalize_question, normalize_passage, has_answer
from .tokenizers import SimpleTokenizer

__all__ = [
    'normalize_question',
    'normalize_passage',
    'has_answer',
    'SimpleTokenizer',
]
