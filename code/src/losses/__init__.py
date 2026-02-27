"""
损失函数模块初始化

本模块包含距离感知对比学习所需的损失函数。

论文章节：第4章 4.1节 - 损失函数设计
"""

from .infonce_loss import InfoNCELoss
from .distance_loss import DistanceLoss
from .combined_loss import CombinedLoss

__all__ = [
    'InfoNCELoss',
    'DistanceLoss',
    'CombinedLoss',
]