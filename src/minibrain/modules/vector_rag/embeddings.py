"""Embedding 调用。openai-compatible 接口，同步 + 线程池并发。

维度处理分两种情况，都不允许静默漂移：
- 模型输出 > 配置维度：MRL 截断 + 重新归一化。qwen3-embedding 这类模型
  用 Matryoshka 训练，取前 N 维仍是有效表示，能省 4 倍内存和存储。
- 模型输出 < 配置维度：直接报错。维度是编不出来的。

query 和 chunk 必须走同一个函数、同一套截断口径，否则检索结果会悄悄变差。
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from ...config import get_config
from ...contracts import ModuleError

_BATCH = 16
_client: OpenAI | None = None


def _truncate_normalize(vector: list[float], target: int) -> list[float]:
    head = vector[:target]
    norm = math.sqrt(sum(v * v for v in head)) or 1.0
    return [v / norm for v in head]


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        cfg = get_config()
        if not cfg.embedding_configured:
            raise ModuleError(
                "EMBEDDING_API_KEY 未配置，请在 .env 中填入", code="embedding_not_configured", status=503
            )
        _client = OpenAI(base_url=cfg.embedding_base_url, api_key=cfg.embedding_api_key)
    return _client


def _embed_batch(texts: list[str]) -> list[list[float]]:
    cfg = get_config()

    # 取 client 放在 try 外面：配置类错误（比如没填 key）本身就是稳定错误，
    # 不能被下面的兜底吞掉再包成一句没信息量的「调用失败：ModuleError」。
    client = _get_client()

    try:
        response = client.embeddings.create(model=cfg.embedding_model, input=texts)
    except ModuleError:
        raise
    except Exception as exc:  # 外部调用错误映射成稳定错误，不把原始响应和密钥漏出去
        raise ModuleError(
            f"embedding 服务调用失败：{type(exc).__name__}: {exc}"[:300],
            code="embedding_failed",
            status=502,
        ) from exc

    vectors = [item.embedding for item in sorted(response.data, key=lambda d: d.index)]
    target = cfg.embedding_dimensions

    for vec in vectors:
        if len(vec) < target:
            raise ModuleError(
                f"embedding 维度不足：模型 {cfg.embedding_model} 只返回 {len(vec)} 维，"
                f"但 EMBEDDING_DIMENSIONS={target}。请调低配置或换模型。",
                code="embedding_dim_mismatch",
                status=500,
            )

    if len(vectors[0]) > target:
        vectors = [_truncate_normalize(vec, target) for vec in vectors]

    return vectors


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    batches = [texts[i : i + _BATCH] for i in range(0, len(texts), _BATCH)]
    if len(batches) == 1:
        return _embed_batch(batches[0])

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(_embed_batch, batches))

    return [vec for batch in results for vec in batch]


def embed_query(text: str) -> list[float]:
    return _embed_batch([text])[0]
