"""
SimpleTokenizer - 基于正则的规则分词器

移植自 DPR/DrQA 的 SimpleTokenizer，用于 has_answer 的 token 级匹配。
注意：这不是 BERT tokenizer，仅用于评估阶段的答案匹配判断。

依赖：regex 库（非标准库 re），支持 Unicode property escapes。
安装：pip install regex>=2023.0.0
"""

import regex


class Tokens:
    """分词结果的轻量封装。"""

    TEXT = 0
    TEXT_WS = 1
    SPAN = 2

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def words(self, uncased=False):
        """返回 token 文本列表。

        Args:
            uncased: 是否转小写
        """
        if uncased:
            return [t[self.TEXT].lower() for t in self.data]
        return [t[self.TEXT] for t in self.data]


class SimpleTokenizer:
    """基于 regex 的规则分词器（来源于 DrQA）。

    分词规则：
    - [\p{L}\p{N}\p{M}]+  匹配 Unicode 字母/数字/音标组合序列 → 作为一个 word token
    - [^\p{Z}\p{C}]        匹配单个非空白/非控制字符 → 作为一个标点 token
    - 空白字符被跳过（作为 token 间的分隔）
    """

    ALPHA_NUM = r"[\p{L}\p{N}\p{M}]+"
    NON_WS = r"[^\p{Z}\p{C}]"

    def __init__(self):
        self._regexp = regex.compile(
            "(%s)|(%s)" % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE,
        )

    def tokenize(self, text):
        data = []
        matches = [m for m in self._regexp.finditer(text)]
        for i in range(len(matches)):
            token = matches[i].group()
            span = matches[i].span()
            start_ws = span[0]
            if i + 1 < len(matches):
                end_ws = matches[i + 1].span()[0]
            else:
                end_ws = span[1]
            data.append((
                token,
                text[start_ws:end_ws],
                span,
            ))
        return Tokens(data)
