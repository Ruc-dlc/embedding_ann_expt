"""
Document编码器

本模块实现Document专用的编码器，将文档文本转换为稠密向量表示。

论文章节：第4章 4.1节 - Document编码
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Any
from transformers import AutoModel

class DocEncoder(nn.Module):
    """
    Document编码器
    
    将文档文本编码为固定维度的稠密向量。
    
    Args:
        backbone: 预训练Transformer模型
        pooling: 池化层
        projection_dim (int): 投影维度（可选）
        normalize (bool): 是否L2归一化
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        pooling: nn.Module,
        projection_dim: Optional[int] = None,
        normalize: bool = True
    ):
        super().__init__()
        
        self.backbone = backbone
        self.pooling = pooling
        self.normalize = normalize
        
        # 获取backbone输出维度
        self.hidden_size = backbone.config.hidden_size
        
        # 可选的投影层
        self.projection: Optional[nn.Linear] = None
        if projection_dim is not None and projection_dim != self.hidden_size:
            self.projection = nn.Linear(self.hidden_size, projection_dim)
            self.output_dim = projection_dim
        else:
            self.output_dim = self.hidden_size
            
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            attention_mask: 注意力掩码 [batch_size, seq_length]
            token_type_ids: Token类型IDs（可选）
            
        Returns:
            Document向量 [batch_size, output_dim]
        """
        # 获取Transformer输出
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs
        )
        
        # 池化
        hidden_states = outputs.last_hidden_state
        embeddings = self.pooling(hidden_states, attention_mask)
        
        # 投影（如果有）
        if self.projection is not None:
            embeddings = self.projection(embeddings)
            
        # L2归一化
        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
            
        return embeddings
    
    def get_output_dim(self) -> int:
        """
        获取输出维度
        
        Returns:
            输出向量维度
        """
        return self.output_dim
    
    @torch.no_grad()
    def encode_corpus(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        batch_size: int = 256
    ) -> torch.Tensor:
        """
        批量编码语料库
        
        用于大规模文档集合的高效编码。
        
        Args:
            input_ids: 所有文档的Token IDs [num_docs, seq_length]
            attention_mask: 注意力掩码 [num_docs, seq_length]
            batch_size: 批处理大小
            
        Returns:
            文档向量 [num_docs, output_dim]
        """
        self.eval()
        all_embeddings = []
        
        num_docs = input_ids.size(0)
        for start_idx in range(0, num_docs, batch_size):
            end_idx = min(start_idx + batch_size, num_docs)
            batch_input_ids = input_ids[start_idx:end_idx]
            batch_attention_mask = attention_mask[start_idx:end_idx]
            
            embeddings = self.forward(batch_input_ids, batch_attention_mask)
            all_embeddings.append(embeddings)
            
        return torch.cat(all_embeddings, dim=0)

class DocEncoderWithPrefix(DocEncoder):
    """
    带前缀的Document编码器
    
    在Document前添加特殊前缀以区分查询和文档。
    
    Args:
        backbone: 预训练Transformer模型
        pooling: 池化层
        prefix (str): Document前缀文本
        tokenizer: Tokenizer实例
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        pooling: nn.Module,
        prefix: str = "passage: ",
        tokenizer: Any = None,
        normalize: bool = True
    ):
        super().__init__(backbone, pooling, normalize=normalize)
        self.prefix = prefix
        self.tokenizer = tokenizer
        self._prefix_ids: Optional[torch.Tensor] = None
        
        if tokenizer is not None:
            self._init_prefix_ids()
            
    def _init_prefix_ids(self) -> None:
        """
        初始化前缀token IDs
        """
        # TODO: 实现前缀初始化
        pass
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        前向传播（带前缀）
        
        Args:
            attention_mask: 注意力掩码 [batch_size, seq_length]
            
        Returns:
            Document向量 [batch_size, output_dim]
        """
        # TODO: 实现带前缀的编码
        return super().forward(input_ids, attention_mask, **kwargs)