"""分词。**框架路径和手写路径共用**，所以单独放一个文件。

## 为什么这个不算"手写实现"而留在主路径

它不是某条链路的实现细节，是**语料本身的性质**决定的：

- 中文没有空格，任何检索器都得先决定怎么切
- LlamaIndex 的 `BM25Retriever` 默认按空白分词，中文会被切成一整坨，
  等于关键词检索完全失效

所以 `llamaindex_chain` 把这个函数当分词器传给 `BM25Retriever`。
换句话说：**框架接管了 BM25 算法，但没接管"中文怎么切"这个问题。**

## 切法：拉丁保留整体 + 拆分，中文走二元组

拉丁复合串（`MTG-20260617-02`）既整体入库、也按分隔符拆开入库——
只记得编号前几位时仍能命中。

中文切二元组（bigram）而不是用 jieba：**不用分词库的理由是测出来的**，
二元组会产生"偶然稀有词"（「三年年假」切出「年年」），污染 BM25 的 IDF；
所以真正进倒排索引的只有标识符类 token，见 `handwritten/keyword.py`
的 `identifier_tokens`。这个取舍在 eval/RESULTS.md 里有数据。
"""

from __future__ import annotations

import re

_LATIN = re.compile(r"[a-z0-9]+(?:[-_./][a-z0-9]+)*")
_SEPARATORS = re.compile(r"[-_./]")
# 连续的中日韩汉字
_CJK = re.compile(r"[一-鿿]+")


def tokenize(text: str) -> list[str]:
    """切成 token。拉丁复合串保留整体 + 拆分，中文走二元组。"""
    text = text.lower()
    tokens: list[str] = []

    for match in _LATIN.finditer(text):
        whole = match.group()
        tokens.append(whole)
        if _SEPARATORS.search(whole):
            # 拆开的部分也算 token：只记得编号前几位时仍能命中
            tokens.extend(part for part in _SEPARATORS.split(whole) if part)

    for match in _CJK.finditer(text):
        run = match.group()
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i:i + 2] for i in range(len(run) - 1))

    return tokens
