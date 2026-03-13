"""
数据模块初始化
"""

from .dataset import DPRDataset, BiEncoderCollator

__all__ = [
    'DPRDataset',
    'BiEncoderCollator',
]
