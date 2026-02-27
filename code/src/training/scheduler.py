"""
学习率调度器

本模块实现多种学习率调度策略，包括：
- 线性预热
- 余弦退火
- 三阶段调度

论文章节：第4章 4.3节 - 训练策略
"""

import math
from typing import Optional
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

class WarmupScheduler(_LRScheduler):
    """
    线性预热调度器
    
    在预热阶段线性增加学习率，之后保持恒定或线性衰减。
    
    Args:
        optimizer: 优化器
        warmup_steps (int): 预热步数
        total_steps (int): 总步数
        last_epoch (int): 上一个epoch（用于恢复训练）
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self):
        """
        计算当前学习率
        """
        if self.last_epoch < self.warmup_steps:
            # 预热阶段：线性增加
            warmup_factor = self.last_epoch / max(1, self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # 衰减阶段：线性衰减
            decay_steps = self.total_steps - self.warmup_steps
            decay_factor = max(0, (self.total_steps - self.last_epoch) / decay_steps)
            return [base_lr * decay_factor for base_lr in self.base_lrs]

class CosineWarmupScheduler(_LRScheduler):
    """
    余弦退火预热调度器
    
    预热后使用余弦退火策略衰减学习率。
    
    Args:
        optimizer: 优化器
        warmup_steps (int): 预热步数
        total_steps (int): 总步数
        min_lr (float): 最小学习率
        last_epoch (int): 上一个epoch
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self):
        """
        计算当前学习率
        """
        if self.last_epoch < self.warmup_steps:
            # 预热阶段
            warmup_factor = self.last_epoch / max(1, self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # 余弦退火
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_factor
                for base_lr in self.base_lrs
            ]

class ThreeStageScheduler(_LRScheduler):
    """
    三阶段调度器
    
    针对三阶段训练策略的学习率调度：
    - 阶段1：预热，学习率线性增加
    - 阶段2：稳定，保持峰值学习率
    - 阶段3：衰减，余弦退火
    
    Args:
        optimizer: 优化器
        stage1_steps (int): 第一阶段步数
        stage2_steps (int): 第二阶段步数
        stage3_steps (int): 第三阶段步数
        warmup_ratio (float): 第一阶段中预热占比
        min_lr (float): 最小学习率
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        stage1_steps: int,
        stage2_steps: int,
        stage3_steps: int,
        warmup_ratio: float = 0.5,
        min_lr: float = 0.0,
        last_epoch: int = -1
    ):
        self.stage1_steps = stage1_steps
        self.stage2_steps = stage2_steps
        self.stage3_steps = stage3_steps
        self.warmup_ratio = warmup_ratio
        self.min_lr = min_lr
        
        self.warmup_steps = int(stage1_steps * warmup_ratio)
        self.total_steps = stage1_steps + stage2_steps + stage3_steps
        
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self):
        """
        计算当前学习率
        """
        step = self.last_epoch
        
        if step < self.warmup_steps:
            # 预热
            factor = step / max(1, self.warmup_steps)
        elif step < self.stage1_steps:
            # 阶段1剩余部分：保持
            factor = 1.0
        elif step < self.stage1_steps + self.stage2_steps:
            # 阶段2：保持
            factor = 1.0
        else:
            # 阶段3：余弦衰减
            stage3_progress = (step - self.stage1_steps - self.stage2_steps) / self.stage3_steps
            factor = 0.5 * (1 + math.cos(math.pi * stage3_progress))
            
        return [
            self.min_lr + (base_lr - self.min_lr) * factor
            for base_lr in self.base_lrs
        ]
    
    def get_current_stage(self) -> int:
        """
        获取当前训练阶段
        
        Returns:
            当前阶段 (1, 2, 或 3)
        """
        step = self.last_epoch
        
        if step < self.stage1_steps:
            return 1
        elif step < self.stage1_steps + self.stage2_steps:
            return 2
        else:
            return 3

def get_scheduler(
    scheduler_type: str,
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    **kwargs
) -> _LRScheduler:
    """
    获取调度器实例
    
    工厂函数，根据类型创建调度器。
    
    Args:
        scheduler_type: 调度器类型 ("linear", "cosine", "three_stage")
        optimizer: 优化器
        warmup_steps: 预热步数
        total_steps: 总步数
        **kwargs: 额外参数
        
    Returns:
        调度器实例
    """
    scheduler_type = scheduler_type.lower()
    
    if scheduler_type == "linear":
        return WarmupScheduler(optimizer, warmup_steps, total_steps)
    elif scheduler_type == "cosine":
        return CosineWarmupScheduler(
            optimizer, warmup_steps, total_steps,
            min_lr=kwargs.get("min_lr", 0.0)
        )
    elif scheduler_type == "three_stage":
        # 需要额外的阶段步数参数
        return ThreeStageScheduler(
            optimizer,
            stage1_steps=kwargs.get("stage1_steps", warmup_steps),
            stage2_steps=kwargs.get("stage2_steps", total_steps // 3),
            stage3_steps=kwargs.get("stage3_steps", total_steps // 3),
            min_lr=kwargs.get("min_lr", 0.0)
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")