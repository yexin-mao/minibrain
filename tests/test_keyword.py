"""BM25 关键词检索。纯函数，不碰数据库。

期望值尽量手推或用可验证的性质来断言，不去对具体分数——
分数依赖 k1/b/IDF 的具体形式，钉死数值只会让改参数变成改测试。
钉死的是**性质**：稀有词权重更高、精确编号能一击命中、没命中就是 0。
"""

from __future__ import annotations

import pytest

from minibrain.modules.vector_rag.keyword import (
    bm25_scores, identifier_tokens, rank_by_bm25, tokenize,
)


# ---------------------------------------------------------------- 分词

def test_latin_compound_is_kept_whole_and_split():
    """★ 编号必须整体保留：整体 token 文档频率极低 → IDF 极高 → 一击命中。

    同时也拆开，让"只记得编号前几位"的查询仍能命中。
    """
    tokens = tokenize("MTG-20260617-02")
    assert "mtg-20260617-02" in tokens        # 整体
    assert "mtg" in tokens                     # 拆开
    assert "20260617" in tokens
    assert "02" in tokens


def test_tokenize_is_case_insensitive():
    assert tokenize("X7-Pro") == tokenize("x7-pro")


def test_chinese_uses_bigrams():
    """中文没空格，用二元组兜底——不需要词典。"""
    assert tokenize("差旅住宿") == ["差旅", "旅住", "住宿"]


def test_single_chinese_char_is_kept():
    assert tokenize("的") == ["的"]


def test_mixed_text():
    tokens = tokenize("工单 TICKET-88231 是什么问题")
    assert "ticket-88231" in tokens
    assert "工单" in tokens


def test_punctuation_is_dropped():
    assert "？" not in tokenize("多少钱？")


# ---------------------------------------------------------------- BM25 性质

DOCS = [
    "会议编号 MTG-20260415-03，季度技术评审，主持人李伟",
    "会议编号 MTG-20260617-02，财务预算评审，主持人周涛",
    "会议编号 MTG-20260701-05，架构治理周会，主持人李伟",
]


def test_exact_id_hits_the_right_document():
    """★ 这就是引入 BM25 的全部理由：向量检索在这题上把正确答案排第 8。"""
    assert rank_by_bm25("MTG-20260617-02 这次会议的决议是什么？", DOCS)[0] == 1


@pytest.mark.parametrize("query, expected", [
    ("MTG-20260415-03", 0),
    ("MTG-20260617-02", 1),
    ("MTG-20260701-05", 2),
])
def test_every_id_hits_its_own_document(query, expected):
    """三个编号前缀完全相同，只差几位数字——向量检索分不清，BM25 必须分得清。"""
    assert rank_by_bm25(query, DOCS)[0] == expected


def test_rare_term_outweighs_common_term():
    """「会议编号」三篇都有（IDF 低），「周涛」只有一篇（IDF 高）。

    这里显式关掉标识符过滤——测的是 BM25 的 IDF 加权本身，不是过滤规则。
    """
    scores = bm25_scores("会议编号 周涛", DOCS, identifiers_only=False)
    assert scores[1] == max(scores)


def test_no_match_scores_zero():
    assert bm25_scores("完全无关的内容", DOCS) == [0.0, 0.0, 0.0]


def test_rank_excludes_zero_scores():
    """★ 和向量检索的关键区别：一个词都没命中就不进排名。

    余弦相似度对任何一对文本都有值，所以向量排名总覆盖全部文档；
    BM25 得 0 说明毫无关系，排进来只是噪声。
    """
    assert rank_by_bm25("完全无关的内容", DOCS) == []


def test_ranking_is_deterministic():
    """分数相同时按下标升序——评测要能复现，不能依赖遍历顺序。"""
    docs = ["同样的内容", "同样的内容"]
    assert rank_by_bm25("同样的内容", docs, identifiers_only=False) == [0, 1]


def test_empty_documents():
    assert bm25_scores("任何查询", []) == []
    assert rank_by_bm25("任何查询", []) == []


def test_long_document_is_penalised():
    """b=0.75 的作用：同样命中一次，长文档得分更低（信息密度低）。"""
    short = "TICKET-88231 报表导出超时"
    long = "TICKET-88231 报表导出超时。" + "无关的填充内容。" * 50
    scores = bm25_scores("TICKET-88231", [short, long])
    assert scores[0] > scores[1]


# ---------------------------------------------------------------- 标识符门槛
#
# 朴素 RRF（向量 + 全量 BM25）把盲区修好了，却把原来 10 道题的
# Recall@5 从 0.803 打到 0.610。根因是中文二元组的"偶然稀有"——
# 「天年」是从「三年年假」切出来的伪词，BM25 抓住它排出一篇无关文档，
# 而 RRF 只看名次不看分数，把噪声当成了权威。
#
# 所以关键词路只负责标识符。下面这几条钉死这个规则。

def test_identifier_tokens_keeps_latin_and_digits():
    assert identifier_tokens("MTG-20260617-02") == [
        "mtg-20260617-02", "mtg", "20260617", "02"]


def test_identifier_tokens_drops_chinese():
    """★ 中文二元组一律不算标识符——它们的稀有性是分词副产品。"""
    assert identifier_tokens("哪个部门的人最多") == []
    assert identifier_tokens("入职满三年有几天年假") == []


def test_identifier_tokens_keeps_only_latin_from_mixed():
    assert identifier_tokens("工单 TICKET-88231 是什么问题") == [
        "ticket-88231", "ticket", "88231"]


def test_pure_chinese_query_contributes_nothing():
    """★ 查询里没有标识符 → BM25 一分不贡献 → 退化成纯向量检索。

    这正是要的行为：中文语义匹配是向量的强项，BM25 掺和进来只会添乱。
    """
    docs = ["技术部共 32 人", "产品部共 14 人", "市场部共 18 人"]
    assert bm25_scores("哪个部门的人最多？", docs) == [0.0, 0.0, 0.0]
    assert rank_by_bm25("哪个部门的人最多？", docs) == []


def test_chinese_pseudo_word_no_longer_hijacks_ranking():
    """回归测试：「天年」这类伪词曾经把无关文档顶到第一。"""
    docs = ["入职满 1 年享有 5 天年假，满 3 年享有 10 天",
            "股权激励授予后分四年归属，第一年归属 25%"]
    assert rank_by_bm25("入职满三年有几天年假？", docs) == []


def test_identifier_query_still_works_with_chinese_around_it():
    docs = ["会议 MTG-20260415-03 季度技术评审", "会议 MTG-20260617-02 财务预算评审"]
    assert rank_by_bm25("MTG-20260617-02 这次会议的决议是什么？", docs)[0] == 1


# ---------------------------------------------------------------- 倒排索引等价性
#
# ★ 这一组是本次优化的验收标准：倒排索引版必须和内存版算出**完全相同**的分数。
#   这是纯性能优化，分数变了就说明实现有 bug。

from minibrain.modules.vector_rag.keyword import (      # noqa: E402
    bm25_from_postings, index_terms, total_term_count,
)


def _build_index(documents: list[str]):
    """把内存里的文档建成倒排索引，模拟数据库里存的东西。"""
    postings: dict[str, dict[str, int]] = {}
    doc_lengths, doc_freq = {}, {}
    for i, doc in enumerate(documents):
        cid = str(i)
        doc_lengths[cid] = total_term_count(doc)
        for term, freq in index_terms(doc).items():
            postings.setdefault(term, {})[cid] = freq
            doc_freq[term] = doc_freq.get(term, 0) + 1
    avg_len = sum(doc_lengths.values()) / len(documents) if documents else 1.0
    return postings, doc_freq, doc_lengths, avg_len


@pytest.mark.parametrize("query", [
    "MTG-20260617-02 这次会议的决议是什么？",
    "MTG-20260415-03",
    "工单 TICKET-88231 是什么问题？",
    "X7-Pro 的 SLA 是多少",
])
def test_index_version_matches_memory_version(query):
    """★ 验收标准：两个版本分数完全一致。"""
    docs = DOCS + ["产品 X7-Pro 的 SLA 承诺 99.95%", "工单 TICKET-88231 报表导出超时"]
    postings, doc_freq, doc_lengths, avg_len = _build_index(docs)

    memory = bm25_scores(query, docs)
    indexed = bm25_from_postings(query, postings, doc_freq, doc_lengths, len(docs), avg_len)

    for i, expected in enumerate(memory):
        actual = indexed.get(str(i), 0.0)
        assert actual == pytest.approx(expected), f"文档 {i} 分数对不上"


def test_index_only_stores_identifiers():
    """中文不进索引——查询侧永远查不到它们，存了是浪费。"""
    terms = index_terms("会议编号 MTG-20260415-03 主持人李伟")
    assert "mtg-20260415-03" in terms
    assert not any("会议" in t or "李伟" in t for t in terms)


def test_term_count_is_full_tokenization():
    """★ 长度归一化的分母必须是全量 token 数，不是索引里的标识符数。

    只数标识符会让分数和内存版对不上——这是最容易写错的地方。
    """
    text = "会议编号 MTG-20260415-03 主持人李伟"
    assert total_term_count(text) > len(index_terms(text))
    assert total_term_count(text) == len(tokenize(text))


def test_postings_miss_returns_empty():
    postings, doc_freq, doc_lengths, avg_len = _build_index(DOCS)
    assert bm25_from_postings("TICKET-99999", postings, doc_freq,
                              doc_lengths, len(DOCS), avg_len) == {}


def test_empty_corpus():
    assert bm25_from_postings("任何查询", {}, {}, {}, 0, 1.0) == {}
