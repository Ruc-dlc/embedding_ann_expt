"""
池化层（CLS、Mean、Max等）

本模块实现多种池化策略，将Transformer输出的token序列聚合为固定维度的向量。

论文章节：第4章 4.1节 - 向量表示
"""

import torch
import torch.nn as nn
from typing import Optional
from abc import ABC, abstractmethod

class PoolingLayer(nn.Module, ABC):
    """
    池化层基类
    
    定义池化层的通用接口。
    """
    
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            hidden_states: Transformer隐藏状态 [batch_size, seq_length, hidden_size]
            attention_mask: 注意力掩码 [batch_size, seq_length]
            
        Returns:
            池化后的向量 [batch_size, hidden_size]
        """
        pass

class CLSPooling(PoolingLayer):
    """
    CLS Token池化
    
    使用[CLS] token的隐藏状态作为整个序列的表示。
    
    这是BERT等模型最常用的池化方式。
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        提取CLS token表示
        
        Args:
            attention_mask: [batch_size, seq_length]（未使用）
            
        Returns:
            CLS向量 [batch_size, hidden_size]
        """
        return hidden_states[:, 0, :]

class MeanPooling(PoolingLayer):
    """
    平均池化
    
    对所有有效token的隐藏状态取平均。
    
    相比CLS池化，平均池化能更好地捕获整个序列的语义信息。
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        计算masked平均池化
        
        Args:
            
        Returns:
            平均向量 [batch_size, hidden_size]
        """
        # 扩展attention_mask以匹配hidden_states的维度
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        
        # 计算加权和
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
        
        # 计算有效token数量
        sum_mask = mask_expanded.sum(dim=1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)  # 防止除零
        
        return sum_embeddings / sum_mask

class MaxPooling(PoolingLayer):
    """
    最大池化
    
    对所有有效token的隐藏状态取最大值。
    
    最大池化能捕获序列中最显著的特征。
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        计算masked最大池化
        
        Args:
            
        Returns:
            最大池化向量 [batch_size, hidden_size]
        """
        # 将padding位置的值设为很小的数，使其不会被选中
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
        hidden_states_masked = hidden_states.masked_fill(mask_expanded == 0, -1e9)
        
        # 沿序列维度取最大值
        max_embeddings, _ = torch.max(hidden_states_masked, dim=1)
        
        return max_embeddings

class AttentionPooling(PoolingLayer):
    """
    注意力池化
    
    使用可学习的注意力机制对token进行加权聚合。
    
    Args:
        hidden_size (int): 隐藏层维度
    """
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        计算注意力加权池化
        
        Args:
            
        Returns:
            注意力池化向量 [batch_size, hidden_size]
        """
        # 计算注意力分数
        attn_scores = self.attention(hidden_states).squeeze(-1)
        
        # 应用mask
        attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e9)
        
        # Softmax归一化
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # 加权求和
        pooled = torch.bmm(attn_weights.unsqueeze(1), hidden_states).squeeze(1)
        
        return pooled

def get_pooling_layer(pooling_type: str, hidden_size: Optional[int] = None) -> PoolingLayer:
    """
    获取池化层实例
    
    工厂函数，根据类型名创建对应的池化层。
    
    Args:
        pooling_type: 池化类型 ("cls", "mean", "max", "attention")
        hidden_size: 隐藏层维度（仅attention池化需要）
        
    Returns:
        池化层实例
        
    Raises:
        ValueError: 未知的池化类型
    """
    pooling_type = pooling_type.lower()
    
    if pooling_type == "cls":
        return CLSPooling()
    elif pooling_type == "mean":
        return MeanPooling()
    elif pooling_type == "max":
        return MaxPooling()
    elif pooling_type == "attention":
        if hidden_size is None:
            raise ValueError("hidden_size is required for attention pooling")
        return AttentionPooling(hidden_size)
    else:
        raise ValueError(f"Unknown pooling type: {pooling_type}")