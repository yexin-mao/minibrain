"""PostgreSQL 持久化入库 worker。

Web 只把实体登记为 uploaded；worker 用 FOR UPDATE SKIP LOCKED + lease 领取。
进程在处理途中退出时，lease 过期后另一个 worker 会重新领取，不丢任务。
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
from contextlib import contextmanager

from . import gateway
from .config import get_config
from .contracts import MODULE_IDS
from .db import close_all
from .observability import core as observability


@contextmanager
def _lease_heartbeat(module_id: str, entity_id: str):
    """处理期间后台续租；停止事件让短任务无需等待完整心跳周期。"""
    stop = threading.Event()
    interval = max(1.0, get_config().ingest_lease_seconds / 3)

    def beat() -> None:
        while not stop.wait(interval):
            try:
                if not gateway.renew_lease(module_id, entity_id):
                    return
            except Exception as exc:  # noqa: BLE001 - 主任务仍可能成功
                print(f"[{module_id}] heartbeat error: {type(exc).__name__}: {exc}",
                      file=sys.stderr, flush=True)

    thread = threading.Thread(target=beat, name=f"lease-{module_id}", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1)


def run_once() -> int:
    """每个模块至多处理一个任务，避免某一条链路长期饿死另一条。"""
    processed = 0
    for module_id in MODULE_IDS:
        entity_id = gateway.claim_next(module_id)
        if entity_id is None:
            continue
        print(f"[{module_id}] processing {entity_id}", flush=True)
        with _lease_heartbeat(module_id, entity_id):
            gateway.process(module_id, entity_id, claimed=True)
        print(f"[{module_id}] finished {entity_id}", flush=True)
        processed += 1
    return processed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true",
                        help="每个模块至多处理一个任务后退出（CI/排查用）")
    args = parser.parse_args()
    stop = threading.Event()

    def request_stop(_signum, _frame) -> None:
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        if args.once:
            observability.archive_stale_runs(get_config().agent_run_stale_seconds)
            run_once()
            return 0

        poll = get_config().ingest_poll_seconds
        print("minibrain worker started", flush=True)
        while not stop.is_set():
            try:
                observability.archive_stale_runs(get_config().agent_run_stale_seconds)
                processed = run_once()
            except Exception as exc:  # noqa: BLE001 - DB 短暂不可用时 worker 不应退出
                print(f"worker loop error: {type(exc).__name__}: {exc}", file=sys.stderr,
                      flush=True)
                processed = 0
            if processed == 0:
                stop.wait(poll)
        return 0
    finally:
        close_all()


if __name__ == "__main__":
    raise SystemExit(main())
