"""
文本预处理与Tokenization

本模块提供文本预处理功能，包括：
- 文本清洗和规范化
- 查询和文档的Tokenization封装
- 批量Tokenization支持

论文章节：第5章 5.1节 - 数据预处理
"""

import re
from typing import Dict, List, Optional, Union, Any

import torch


class TextPreprocessor:
    """
    文本预处理器

    统一处理Query和Document的文本预处理与Tokenization流程。
    Query最大长度默认64，Document最大长度默认256（标题+正文拼接）。

    参数:
        tokenizer: HuggingFace Tokenizer实例
        max_query_length: 查询最大token长度
        max_doc_length: 文档最大token长度
        lowercase: 是否转为小写
    """

    def __init__(
        self,
        tokenizer: Any,
        max_query_length: int = 64,
        max_doc_length: int = 256,
        lowercase: bool = False
    ):
        self.tokenizer = tokenizer
        self.max_query_length = max_query_length
        self.max_doc_length = max_doc_length
        self.lowercase = lowercase

    def preprocess_text(self, text: str) -> str:
        """
        文本基础预处理（去空白、可选小写）

        参数:
            text: 输入文本

        返回:
            预处理后的文本
        """
        if text is None:
            return ""

        text = text.strip()

        if self.lowercase:
            text = text.lower()

        # 规范化空白字符
        text = ' '.join(text.split())

        return text

    def tokenize_query(
        self,
        query: Union[str, List[str]],
        return_tensors: str = "pt"
    ) -> Dict[str, torch.Tensor]:
        """
        对查询文本进行Tokenization

        参数:
            query: 单个查询字符串或查询列表
            return_tensors: 返回张量类型

        返回:
            包含 input_ids 和 attention_mask 的字典
        """
        if isinstance(query, str):
            query = [query]

        query = [self.preprocess_text(q) for q in query]

        encoded = self.tokenizer(
            query,
            max_length=self.max_query_length,
            padding='max_length',
            truncation=True,
            return_tensors=return_tensors
        )

        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
        }

    def tokenize_document(
        self,
        text: Union[str, List[str]],
        title: Optional[Union[str, List[str]]] = None,
        return_tensors: str = "pt"
    ) -> Dict[str, torch.Tensor]:
        """
        对文档进行Tokenization

        文档格式为 "title text" 拼接，若无标题则仅使用正文。

        参数:
            text: 文档正文（单个或列表）
            title: 文档标题（可选，单个或列表）
            return_tensors: 返回张量类型

        返回:
            包含 input_ids 和 attention_mask 的字典
        """
        if isinstance(text, str):
            text = [text]

        if title is not None:
            if isinstance(title, str):
                title = [title]
            # 拼接标题和正文
            documents = []
            for t, d in zip(title, text):
                t = self.preprocess_text(t)
                d = self.preprocess_text(d)
                doc_text = f"{t} {d}".strip() if t else d
                documents.append(doc_text)
        else:
            documents = [self.preprocess_text(d) for d in text]

        encoded = self.tokenizer(
            documents,
            max_length=self.max_doc_length,
            padding='max_length',
            truncation=True,
            return_tensors=return_tensors
        )

        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
        }

    def batch_tokenize(
        self,
        texts: List[str],
        max_length: int,
        return_tensors: str = "pt"
    ) -> Dict[str, torch.Tensor]:
        """
        批量Tokenization

        通用的批量tokenize方法，可用于任意文本列表。

        参数:
            texts: 文本列表
            max_length: 最大token长度
            return_tensors: 返回张量类型

        返回:
            包含 input_ids 和 attention_mask 的字典
        """
        texts = [self.preprocess_text(t) for t in texts]

        encoded = self.tokenizer(
            texts,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors=return_tensors
        )

        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
        }


class QueryPreprocessor(TextPreprocessor):
    """
    查询专用预处理器

    继承通用预处理器，默认max_length为64。

    参数:
        tokenizer: HuggingFace Tokenizer实例
        max_length: 查询最大token长度
    """

    def __init__(self, tokenizer: Any, max_length: int = 64):
        super().__init__(tokenizer, max_query_length=max_length)

    def tokenize(
        self,
        query: Union[str, List[str]],
        return_tensors: str = "pt"
    ) -> Dict[str, torch.Tensor]:
        """Tokenize查询的快捷方法"""
        return self.tokenize_query(query, return_tensors=return_tensors)


class DocumentPreprocessor(TextPreprocessor):
    """
    文档专用预处理器

    继承通用预处理器，默认max_length为256。

    参数:
        tokenizer: HuggingFace Tokenizer实例
        max_length: 文档最大token长度
    """

    def __init__(self, tokenizer: Any, max_length: int = 256):
        super().__init__(tokenizer, max_doc_length=max_length)

    def tokenize(
        self,
        text: Union[str, List[str]],
        title: Optional[Union[str, List[str]]] = None,
        return_tensors: str = "pt"
    ) -> Dict[str, torch.Tensor]:
        """Tokenize文档的快捷方法"""
        return self.tokenize_document(text, title=title, return_tensors=return_tensors)