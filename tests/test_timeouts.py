"""出站 HTTP 超时。这个文件守的是一条**用 13 小时换来的**防线。

## 背景

三处构造 OpenAI 客户端（agent/loop、rerank、embeddings）原本都没设 timeout。
OpenAI SDK 默认 600 秒 + 2 次重试，单次最坏 30 分钟。
一次 194 调用的重排消融因此挂死 13 小时 17 分——
**进程活着、CPU 只用了 14.7 秒**，全程在等一个永远不返回的响应，
而且外部看不出它在挂：没有报错、没有退出、日志停在中间。

> `except Exception` 挡得住「调用失败」，挡不住「调用不返回」。
> 这是两种完全不同的故障，超时是后者唯一的防线。

## 为什么值得写测试

超时是那种「加上去之后什么都不会发生」的代码——正因为如此，
它**特别容易在重构时被顺手删掉或改成 None**，而且删掉之后
所有测试照样绿、所有功能照样能用，只在某次线上抖动时炸成十几个小时的挂起。

所以这里不测「超时会不会触发」（那要造一个假的慢服务，成本高、还容易 flaky），
只测**客户端确实带着一个有限的超时值被构造出来**。
这条断言便宜、稳定，而且正好覆盖那次事故的根因。
"""

from __future__ import annotations

import pytest

from minibrain.config import get_config


def _clients():
    """返回 (名字, 构造函数)。惰性 import，避免没配 key 时报错。

    ★ 换框架之后这张表**必须跟着扩**。原来只有三处手写客户端；
      现在主路径走 LangChain 的 ChatOpenAI 和 LlamaIndex 的 LLM/embedding，
      那几处同样是出站 HTTP 调用，同样会挂。
      **防线要覆盖新代码路径，不是覆盖旧代码路径。**
      框架客户端的超时在 graph.py / chain.py 里显式传，见 test_framework_clients。
    """
    from minibrain.handwritten import agent_loop, rerank
    from minibrain.modules.vector_rag import embeddings
    return [
        ("handwritten/agent_loop", agent_loop._get_client),
        ("handwritten/rerank", rerank._get_client),
        ("vector_rag/embeddings", embeddings._get_client),
    ]


@pytest.mark.parametrize("name", [n for n, _ in _clients()])
def test_every_openai_client_has_a_finite_timeout(name):
    """三处客户端都必须带有限超时。少一处，那一处就是下一次挂 13 小时的地方。"""
    builder = dict(_clients())[name]
    cfg = get_config()
    if not (cfg.agent_configured and cfg.embedding_configured):
        pytest.skip("没配 API key，构造不出客户端")

    client = builder()
    assert client.timeout is not None, f"{name} 没有设超时"
    assert 0 < float(client.timeout) < 600, (
        f"{name} 的超时是 {client.timeout}s。"
        "必须是正数，且要明显小于 SDK 默认的 600s——"
        "默认值正是那次挂死 13 小时的原因。")


@pytest.mark.parametrize("name", [n for n, _ in _clients()])
def test_every_openai_client_caps_retries(name):
    """重试次数要有上限。重试会把最坏耗时翻倍，是超时之外的第二个放大器。"""
    builder = dict(_clients())[name]
    cfg = get_config()
    if not (cfg.agent_configured and cfg.embedding_configured):
        pytest.skip("没配 API key，构造不出客户端")

    client = builder()
    assert client.max_retries <= 2, f"{name} 的重试次数 {client.max_retries} 太多"


@pytest.mark.parametrize("var", ["LLM_TIMEOUT_SECONDS", "EMBEDDING_TIMEOUT_SECONDS"])
@pytest.mark.parametrize("bad", ["0", "-1"])
def test_config_rejects_disabled_timeout(monkeypatch, var, bad):
    """★ 不允许把超时关掉。

    设成 0 或负数在 SDK 里等于「永不超时」，那正是事故时的状态。
    config.py 对此 fail-fast，这个测试钉住那条校验——
    免得有人为了「调试方便」改成 0 然后忘了改回来。

    ★★ 写法上的注意：这里必须**真的走一遍 get_config()**。
       第一版图省事，用 dataclasses.replace 造一个坏配置，然后断言
       `not (0 > 0)` —— 那是一句恒真的废话，什么都没测到。
       本项目在 CI 里出过三个这种空转通过的测试（其中一个是权限测试），
       教训是：**断言必须能因为源码改动而变红**，否则它只是装饰。
    """
    import minibrain.config as config_module

    monkeypatch.setenv(var, bad)
    monkeypatch.setattr(config_module, "_config", None)   # 绕过单例缓存
    with pytest.raises(RuntimeError, match="必须为正数"):
        config_module.get_config()


# ---------------------------------------------------------------- 框架路径
#
# ★ 主路径已经换成 LangGraph + LlamaIndex，它们各自持有自己的 HTTP 客户端。
#   超时这条防线**不会因为换框架而自动继承**——框架的默认值多半更宽松，
#   甚至没有。所以这两条单独钉住。

@pytest.mark.skipif(not get_config().agent_configured, reason="没配 AGENT_API_KEY")
def test_langchain_chat_model_has_a_finite_timeout():
    """LangGraph 版 agent 用的 ChatOpenAI 必须带超时。"""
    from langchain_openai import ChatOpenAI

    cfg = get_config()
    model = ChatOpenAI(
        base_url=cfg.agent_base_url, api_key=cfg.agent_api_key,
        model=cfg.agent_model, temperature=cfg.agent_temperature,
        timeout=cfg.llm_timeout_seconds, max_retries=cfg.llm_max_retries)
    assert model.request_timeout is not None
    assert 0 < float(model.request_timeout) < 600
    assert model.max_retries <= 2


@pytest.mark.skipif(not get_config().agent_configured, reason="没配 AGENT_API_KEY")
def test_llamaindex_llm_has_a_finite_timeout():
    """LlamaIndex 的 Settings.llm 同理。

    它是 QueryFusionRetriever 强制要求的（即使 num_queries=1），
    所以它一定会被构造出来，也一定要有超时。
    """
    from minibrain import llamaindex_setup

    llamaindex_setup.ensure_llm()
    from llama_index.core import Settings
    assert Settings.llm.timeout is not None
    assert 0 < float(Settings.llm.timeout) < 600
