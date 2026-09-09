"""被框架换下来的手写实现。**不在主路径上，但保留**。

主路径：`agent/graph.py`（LangGraph）+ `modules/vector_rag/chain.py`（LlamaIndex）。
这里是对照组，用途和取舍见同目录的 README.md。

★ 这个包**不应该被主路径 import**。反过来可以：手写版会用主路径的
  tokenizer / embeddings / core（权限和文档注册表），因为那些两版共用。
"""
