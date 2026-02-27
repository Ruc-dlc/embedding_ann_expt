"""
训练模块初始化

本模块包含训练流程相关的组件。

论文章节：第4章 4.3节 - 训练策略
"""

from .trainer import Trainer
from .training_args import TrainingArguments
from .scheduler import WarmupScheduler, ThreeStageScheduler
from .callbacks import (
    TrainingCallback,
    CheckpointCallback,
    ValidationCallback,
    LoggingCallback
)

__all__ = [
    'Trainer',
    'TrainingArguments',
    'WarmupScheduler',
    'ThreeStageScheduler',
    'TrainingCallback',
    'CheckpointCallback',
    'ValidationCallback',
    'LoggingCallback',
]