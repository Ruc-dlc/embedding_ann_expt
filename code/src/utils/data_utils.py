"""
数据预处理工具函数

对齐 DPR 开源代码的预处理标准，确保训练和评估结果可复现。
参考：
  - DPR/dpr/utils/data_utils.py  → normalize_question
  - DPR/dpr/data/biencoder_data.py → normalize_passage
  - DPR/dpr/data/qa_validation.py  → has_answer
"""

import unicodedata
from typing import List

from .tokenizers import SimpleTokenizer


# ---------- 全局单例 ----------
# SimpleTokenizer 初始化编译正则，复用避免重复编译开销
_default_tokenizer = None


def get_default_tokenizer() -> SimpleTokenizer:
    """获取全局 SimpleTokenizer 单例。"""
    global _default_tokenizer
    if _default_tokenizer is None:
        _default_tokenizer = SimpleTokenizer()
    return _default_tokenizer


# ---------- 文本预处理 ----------

def normalize_question(question: str) -> str:
    """Query 预处理：仅替换 Unicode 右引号为 ASCII 引号。

    与 DPR 的 normalize_question 完全一致：
    只做 ' (U+2019) → ' 替换，不做大小写转换或其他处理。
    """
    question = question.replace("\u2019", "'")
    return question


def normalize_passage(text: str) -> str:
    """Passage text 预处理。

    与 DPR 的 normalize_passage 完全一致：
    1. 换行符 → 空格
    2. Unicode 右引号 → ASCII 引号
    3. 去除首尾双引号（若有）
    """
    text = text.replace("\n", " ").replace("\u2019", "'")
    if text.startswith('"'):
        text = text[1:]
    if text.endswith('"'):
        text = text[:-1]
    return text


# ---------- 答案匹配 ----------

def _normalize_unicode(text: str) -> str:
    """Unicode NFD 归一化（将组合字符分解为基字符+变音符号）。"""
    return unicodedata.normalize("NFD", text)


def has_answer(answers: List[str], text: str, tokenizer: SimpleTokenizer = None) -> bool:
    """检查 passage text 中是否包含任一答案（token 级连续子序列匹配）。

    与 DPR 的 has_answer(match_type="string") 完全一致：
    1. 对 text 和 answer 均做 NFD 归一化
    2. 用 SimpleTokenizer 分词
    3. 全部转小写 (uncased=True)
    4. 检查 answer token 序列是否作为连续子序列出现在 text token 序列中

    Args:
        answers: 答案字符串列表（一个 query 可能有多个合法答案）
        text: passage 原始文本
        tokenizer: SimpleTokenizer 实例，None 则使用全局单例

    Returns:
        True 如果 text 中包含至少一个答案的 token 子序列
    """
    if tokenizer is None:
        tokenizer = get_default_tokenizer()

    text = _normalize_unicode(text)
    text_tokens = tokenizer.tokenize(text).words(uncased=True)

    for single_answer in answers:
        single_answer = _normalize_unicode(single_answer)
        answer_tokens = tokenizer.tokenize(single_answer).words(uncased=True)

        ans_len = len(answer_tokens)
        if ans_len == 0:
            continue

        for i in range(len(text_tokens) - ans_len + 1):
            if answer_tokens == text_tokens[i: i + ans_len]:
                return True

    return False
