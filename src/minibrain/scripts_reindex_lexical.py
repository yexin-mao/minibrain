"""为已有 LlamaIndex 节点重建全局 BM25 索引。"""

from __future__ import annotations

import json
from collections import defaultdict

from psycopg.types.json import Jsonb

from .db import close_all, vector_db, vector_index_db
from .modules.vector_rag.chain import rebuild_lexical_index


def _attach_document_id(metadata: dict, document_id: str) -> dict:
    """同步更新 PGVectorStore 顶层 metadata 与其序列化节点副本。"""
    updated = dict(metadata)
    updated["registry_document_id"] = document_id
    raw_node = updated.get("_node_content")
    if isinstance(raw_node, str):
        node = json.loads(raw_node)
        node.setdefault("metadata", {})["registry_document_id"] = document_id
        excluded = node.setdefault("excluded_embed_metadata_keys", [])
        if "registry_document_id" not in excluded:
            excluded.append("registry_document_id")
        updated["_node_content"] = json.dumps(node, ensure_ascii=False)
    return updated


def backfill_document_ids() -> tuple[int, int, int]:
    """用 owner/source/filename 唯一匹配历史节点；歧义或无主节点绝不猜。"""
    documents: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    with vector_db() as cur:
        cur.execute(
            """SELECT d.id, d.filename, s.name AS source_name, s.owner_id
               FROM documents d JOIN sources s ON s.id = d.source_id""")
        for row in cur.fetchall():
            key = (str(row["owner_id"]), str(row["source_name"]), str(row["filename"]))
            documents[key].append(str(row["id"]))

    updates: list[tuple[Jsonb, str]] = []
    ambiguous = unmatched = 0
    known_document_ids = {document_id for ids in documents.values() for document_id in ids}
    with vector_index_db() as cur:
        cur.execute("SELECT node_id, metadata_ FROM data_nodes ORDER BY id")
        for row in cur.fetchall():
            metadata = dict(row["metadata_"] or {})
            existing_id = metadata.get("registry_document_id")
            if existing_id:
                if str(existing_id) not in known_document_ids:
                    unmatched += 1
                continue
            key = (
                str(metadata.get("owner_id", "")),
                str(metadata.get("source_name", "")),
                str(metadata.get("filename", "")),
            )
            matches = documents.get(key, [])
            if len(matches) == 1:
                updates.append((Jsonb(_attach_document_id(metadata, matches[0])),
                                str(row["node_id"])))
            elif len(matches) > 1:
                ambiguous += 1
            else:
                unmatched += 1
        if updates:
            cur.executemany(
                "UPDATE data_nodes SET metadata_ = %s WHERE node_id = %s", updates)
    return len(updates), ambiguous, unmatched


def main() -> int:
    try:
        linked, ambiguous, unmatched = backfill_document_ids()
        count = rebuild_lexical_index()
    finally:
        close_all()
    print(f"已为 {count} 个节点重建全局 sparse 索引（未调用 embedding）")
    print(f"registry_document_id 回填：{linked}；歧义跳过：{ambiguous}；"
          f"无注册文档：{unmatched}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
