"""
随机种子设置

本模块提供随机种子的统一设置，确保实验可复现。

论文章节：第5章 5.1节 - 实验设置
"""

import random
import numpy as np
import torch
from typing import Optional

def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    设置随机种子
    
    统一设置Python、NumPy和PyTorch的随机种子。
    
    Args:
        seed: 随机种子值
        deterministic: 是否启用确定性计算（可能影响性能）
    """
    # Python内置随机
    random.seed(seed)
    
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    # 确定性设置
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        if hasattr(torch, 'use_deterministic_algorithms'):
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass

def get_random_state() -> dict:
    """
    获取当前随机状态
    
    用于保存和恢复随机状态。
    
    Returns:
        包含各库随机状态的字典
    """
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
        
    return state

def set_random_state(state: dict) -> None:
    """
    设置随机状态
    
    恢复之前保存的随机状态。
    
    Args:
        state: 随机状态字典
    """
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    
    if 'cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])

class SeedContext:
    """
    随机种子上下文管理器
    
    在特定代码块中临时设置随机种子。
    
    Example:
        ...     # 这里的随机操作是可复现的
    """
    
    def __init__(self, seed: int):
        self.seed = seed
        self.saved_state = None
        
    def __enter__(self):
        # 保存当前状态
        self.saved_state = get_random_state()
        # 设置新种子
        set_seed(self.seed)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 恢复之前的状态
        set_random_state(self.saved_state)
        return False