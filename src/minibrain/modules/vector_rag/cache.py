"""进程内短 TTL 检索缓存；键必须同时绑定权限范围和知识库版本。"""

from __future__ import annotations

import copy
import time
from collections import OrderedDict
from threading import Lock
from typing import Callable, Hashable, TypeVar


T = TypeVar("T")


class TTLCache:
    def __init__(self, *, max_entries: int = 256, ttl_seconds: float = 60.0,
                 clock: Callable[[], float] = time.monotonic):
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._clock = clock
        self._items: OrderedDict[Hashable, tuple[float, object]] = OrderedDict()
        self._lock = Lock()

    def get(self, key: Hashable) -> T | None:
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            expires_at, value = item
            if expires_at <= self._clock():
                del self._items[key]
                return None
            self._items.move_to_end(key)
            # Evidence 会在 Agent 层补 evidence_id，不能把这次修改泄漏回缓存。
            return copy.deepcopy(value)  # type: ignore[return-value]

    def put(self, key: Hashable, value: T) -> None:
        with self._lock:
            self._items[key] = (self._clock() + self.ttl_seconds, copy.deepcopy(value))
            self._items.move_to_end(key)
            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


retrieval_cache = TTLCache()


def retrieval_cache_key(*, user_id: str, is_admin: bool, knowledge_version: str,
                        query: str, top_k: int, options: tuple[object, ...]) -> tuple:
    """显式列出安全边界，防止未来简化键时误删用户或版本。"""
    return (
        "vector-search-v1", user_id, is_admin, knowledge_version,
        " ".join(query.split()).casefold(), top_k, *options,
    )
