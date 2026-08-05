"""BM25 关键词检索。纯函数，不碰数据库，可以被单元测试钉死。

## 为什么要它

实测（eval/RESULTS.md 探针一之二/之三）：向量检索在「前缀相同、只差几位数字」
的编号上排序严重退化——

    会议编号 MTG-2026xxxx-xx   MRR 0.489
    工单编号 TICKET-88xxx      MRR 0.708
    项目编号 PRJ-2026-xxxx     MRR 0.750

根因不是召回（Recall@5 = 1.000，东西都捞回来了），是**排序**：
8 篇会议纪要的相似度全挤在 0.60~0.67 之间，跨度只有 0.0738，
而普通查询是 0.3950。数字在稠密向量里几乎不携带区分信息。

已经排除的更便宜方案：调切分（当前语料测不了）、升维到 4096（无效，仍排第 8）。
所以补一条精确匹配的路径。

## 为什么是自己写 BM25，而不是 PostgreSQL 全文检索

PG 的 `to_tsvector` 不认中文——需要 zhparser / pg_jieba 扩展，
要改部署、要改 CI、要在镜像里编译。而 BM25 是这个领域的标准算法，
面试也按名字问，自己实现一遍反而更清楚它在算什么。

代价说清楚：**本实现是内存版**，和向量侧的 numpy 余弦一样，
把可见片段全量拉进来算。几万条以上要换 PG 全文检索或专门的检索引擎。
这条已记进 SCALING.md。

## 分词策略

中文没有空格，标准做法是二元组（bigram）：「差旅住宿标准」→ 差旅/旅住/住宿/宿标/标准。
不需要词典，覆盖率高，是没有分词器时的通行兜底。

拉丁字母和数字的复合串**整体保留**，同时也拆开：

    MTG-20260617-02  →  ['mtg-20260617-02', 'mtg', '20260617', '02']
                          ↑ 整体             ↑ 拆开的各部分

整体那个 token 只在完全匹配时命中，文档频率极低 → IDF 极高 → 一击命中。
拆开的部分让部分匹配也有分数（比如只记得单号前几位）。
**这一条正是为编号型查询设计的**，也是向量检索做不到的事。
"""

from __future__ import annotations

import math
import re
from collections import Counter

# 拉丁字母 / 数字的复合串，允许中间有 - _ . / 连接
_LATIN = re.compile(r"[a-z0-9]+(?:[-_./][a-z0-9]+)*")
_SEPARATORS = re.compile(r"[-_./]")
# 连续的中日韩汉字
_CJK = re.compile(r"[一-鿿]+")

# Okapi BM25 的两个经典参数。k1 控制词频饱和速度，b 控制文档长度惩罚强度。
# 1.5 / 0.75 是文献里的通用默认值，本项目没有调参——调参需要独立的验证集，
# 拿测试集调出来的参数是自欺。
K1 = 1.5
B = 0.75


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


def identifier_tokens(text: str) -> list[str]:
    """只保留**标识符** token：拉丁字母 / 数字构成的串。

    ★ 这个函数是一次失败实验的产物，理由必须写清楚，否则以后有人会"顺手简化掉"。

    朴素 RRF（向量 + 全量 BM25）在盲区上很成功（会议编号 MRR 0.489 → 0.906），
    但把原来 10 道题的 Recall@5 从 0.803 打到 0.610。

    诊断：BM25 在没有区分性信号时**仍然会给出一个看起来很自信的排名**，
    而 RRF 只看名次不看分数，把噪声当成了权威。典型样本：

        问「哪个部门的人最多？」
          向量：dept-tech / dept-hr / dept-product / dept-finance / dept-market   全对
          BM25：metric-rpo / policy-team-building / policy-recruit / ...          全错
          融合：垃圾把对的挤掉了

    第一反应是"加 IDF 门槛：查询里没有稀有词就不启用 BM25"。**这条被数据否了**——
    「哪个部门的人最多」里最稀有的词是「最多」，语料里只出现 1 次，和精确编号一样稀有。
    「同一」「天年」也是。这些是**中文二元组的"偶然稀有"**：
    `天年` 是从「三年年假」切出来的伪词，根本不是一个词。

    真正的分界线是 token 的**类型**，不是稀有度：

      拉丁/数字 token（20260617、ticket-88823、sla、x7-pro）
        → 稀有性是真实语义信号，而且**正是向量的盲区**
      中文二元组（最多、同一、天年）
        → 稀有性是分词方式的副产品；中文语义匹配本来就是向量的强项

    所以关键词路**只负责标识符**，中文交给向量。查询里没有标识符时，
    BM25 一分不贡献，退化成纯向量检索——那本来就是它该做的。

    ## 试过三元组吗？试了，不行（实测，见 eval/RESULTS.md 探针八）

    「二元组会切出伪词，那用三元组呢？」——很自然的想法，跑了对比：

      BM25 用什么词        全局MRR  Recall@5  编号类MRR  普通题Recall@5
      二元组 + 中文          0.967    0.933     0.936      0.892
      三元组 + 中文          0.950    0.970     0.913      0.951
      四元组 + 中文          0.950    0.970     0.913      0.951   ← 和三元组完全一样
      只用标识符（当前）      0.971    0.970     0.947      0.951

    三元组确实把普通题的 Recall 修回来了（0.892 → 0.951），但 MRR 反而不如二元组。

    **关键证据是三元组和四元组的数字一字不差**——说明到三元组时中文已经
    基本匹配不上任何东西了。它不是"更精确的中文匹配"，它就是
    **"中文不参与"的一个不干净版本**：还残留少量中文命中，而那些恰好是噪声。

    而且它有个结构性缺陷：**中文的人名、组名大多是 2~3 字**，
    三元组对「刘洋」「陈刚」这类 2 字词完全失明——丢掉的是真实信号。

    所以结论不是"n 取大一点就好了"，是**中文语义匹配本来就该交给向量**。

    已知边界：这条规则依赖"中文正文 + 拉丁标识符"这个组合。
    纯英文语料里所有词都是拉丁的，规则会退化成"总是启用 BM25"，
    那时需要真正的停用词表。见 SCALING.md。
    """
    return [t for t in tokenize(text) if _LATIN.fullmatch(t)]


def bm25_scores(query: str, documents: list[str], *,
                identifiers_only: bool = True) -> list[float]:
    """对每篇文档算 BM25 得分。分数越高越相关，0 表示一个查询词都没命中。

    IDF 用的是当前**可见文档集合**算出来的——这一点是刻意的：
    权限过滤之后剩下什么，稀有度就按什么算。用全库统计反而会泄露
    "库里还有多少篇含这个词"的信息。
    """
    if not documents:
        return []

    doc_tokens = [tokenize(doc) for doc in documents]
    total = len(documents)
    avg_len = sum(len(t) for t in doc_tokens) / total or 1.0

    doc_freq: Counter[str] = Counter()
    for tokens in doc_tokens:
        doc_freq.update(set(tokens))

    term_freqs = [Counter(tokens) for tokens in doc_tokens]
    scores = [0.0] * total

    query_terms = identifier_tokens(query) if identifiers_only else tokenize(query)
    for term in set(query_terms):
        n_containing = doc_freq.get(term, 0)
        if n_containing == 0:
            continue
        # 带 +1 的 IDF 变体，保证非负——否则出现在超过半数文档里的词会得负分，
        # 融合时反而把命中的文档往下压。
        idf = math.log(1 + (total - n_containing + 0.5) / (n_containing + 0.5))

        for index, freqs in enumerate(term_freqs):
            freq = freqs.get(term, 0)
            if freq == 0:
                continue
            doc_len = len(doc_tokens[index])
            scores[index] += idf * (freq * (K1 + 1)) / (
                freq + K1 * (1 - B + B * doc_len / avg_len)
            )

    return scores


def rank_by_bm25(query: str, documents: list[str], *,
                 identifiers_only: bool = True) -> list[int]:
    """返回按 BM25 降序排好的文档下标。**得分为 0 的不返回**。

    这一点和向量检索不同：余弦相似度对任何一对文本都有值（哪怕毫不相关），
    所以向量排名总是覆盖全部文档；BM25 得 0 意味着一个查询词都没命中，
    把它排进来只是噪声。
    """
    scores = bm25_scores(query, documents, identifiers_only=identifiers_only)
    hits = [i for i, s in enumerate(scores) if s > 0]
    hits.sort(key=lambda i: (-scores[i], i))
    return hits


# ---------------------------------------------------------------- 倒排索引版
#
# 上面那个 bm25_scores 每次查询都把**全部文档重新分词、重算 IDF**。
# 实测（eval/RESULTS.md 探针十）：84 片段时 4.98ms，900 片段时 201.65ms——
# 增长 40.5 倍，而片段数只涨 10.7 倍。因为它是 O(总字符数) 不是 O(片段数)。
#
# 下面这个版本接受**预先建好的倒排索引**，只对命中的片段打分。
# 分词和 IDF 统计在入库时做一次，查询时不碰原文。
#
# ★ 两个版本必须算出**完全相同**的分数——test_keyword.py 里有等价性测试钉着。
#   这是纯性能优化，质量指标一位小数都不许变。


def bm25_from_postings(
    query: str,
    postings: dict[str, dict[str, int]],
    doc_freq: dict[str, int],
    doc_lengths: dict[str, int],
    total_docs: int,
    avg_len: float,
) -> dict[str, float]:
    """用预建索引给命中的片段打分。

    参数全部来自数据库，不需要原文：

      postings      {词: {片段id: 该词在这片段出现几次}}   —— 只含查询命中的词
      doc_freq      {词: 含这个词的片段数}                 —— 用于 IDF
      doc_lengths   {片段id: 该片段的总 token 数}          —— 用于长度归一化
      total_docs    可见片段总数                           —— 用于 IDF
      avg_len       可见片段的平均 token 数                —— 用于长度归一化

    ★ doc_lengths 必须是**全量分词**的计数（含中文），不能只数索引里的标识符——
      否则长度归一化的分母变了，分数就和内存版对不上。

    返回 {片段id: 分数}，只含分数 > 0 的。
    """
    if total_docs == 0:
        return {}
    avg_len = avg_len or 1.0

    scores: dict[str, float] = {}
    for term in set(identifier_tokens(query)):
        n_containing = doc_freq.get(term, 0)
        if n_containing == 0:
            continue
        idf = math.log(1 + (total_docs - n_containing + 0.5) / (n_containing + 0.5))

        for chunk_id, freq in postings.get(term, {}).items():
            doc_len = doc_lengths.get(chunk_id, 0)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + idf * (freq * (K1 + 1)) / (
                freq + K1 * (1 - B + B * doc_len / avg_len)
            )
    return scores


def index_terms(text: str) -> dict[str, int]:
    """入库时调用：算出这个片段要存进倒排索引的词及其出现次数。

    **只存标识符**——查询侧走 identifier_tokens()，中文 token 永远查不到。
    实测：全量索引约 112,000 行，只索引标识符 759 行，省 99% 且结果完全一致。
    """
    return dict(Counter(identifier_tokens(text)))


def total_term_count(text: str) -> int:
    """入库时调用：这个片段的**全量** token 数，存进 chunks.term_count。

    必须全量（含中文二元组）——它是 BM25 长度归一化的分母，
    只数标识符会让分数和内存版对不上。
    """
    return len(tokenize(text))
