"""审计 LlamaIndex dense、全局 sparse 与文档注册表的一致性。"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

from .db import close_all, vector_db, vector_index_db
from .modules.vector_rag import chain


def _registry() -> tuple[set[str], dict[tuple[str, str, str], list[str]]]:
    ids: set[str] = set()
    legacy: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    with vector_db() as cur:
        cur.execute(
            """SELECT d.id, d.filename, s.name AS source_name, s.owner_id
               FROM documents d JOIN sources s ON s.id = d.source_id""")
        for row in cur.fetchall():
            document_id = str(row["id"])
            ids.add(document_id)
            key = (str(row["owner_id"]), str(row["source_name"]), str(row["filename"]))
            legacy[key].append(document_id)
    return ids, legacy


def _dense_state() -> list[tuple[str, dict]]:
    with vector_index_db() as cur:
        cur.execute("SELECT node_id, metadata_ FROM data_nodes ORDER BY id")
        return [(str(row["node_id"]), dict(row["metadata_"] or {}))
                for row in cur.fetchall()]


def audit_index() -> dict:
    registry_ids, legacy = _registry()
    dense = _dense_state()
    dense_ids = {node_id for node_id, _ in dense}
    with vector_index_db() as cur:
        cur.execute("SELECT node_id FROM node_lexical_stats")
        sparse_ids = {str(row["node_id"]) for row in cur.fetchall()}
        cur.execute("SELECT id, document_id FROM parent_contexts")
        parents = [(str(row["id"]), (str(row["document_id"])
                    if row["document_id"] is not None else None))
                   for row in cur.fetchall()]

    referenced_parents = {
        str(metadata["parent_context_id"])
        for _, metadata in dense if metadata.get("parent_context_id")
    }
    parent_ids = {parent_id for parent_id, _ in parents}
    orphan_parents = sorted(
        parent_id for parent_id, document_id in parents
        if parent_id not in referenced_parents
        or (document_id is not None and document_id not in registry_ids)
    )

    orphan_dense: list[str] = []
    legacy_backfillable: list[str] = []
    ambiguous_legacy: list[str] = []
    for node_id, metadata in dense:
        registered = metadata.get("registry_document_id")
        if registered:
            if str(registered) not in registry_ids:
                orphan_dense.append(node_id)
            continue
        key = (
            str(metadata.get("owner_id", "")),
            str(metadata.get("source_name", "")),
            str(metadata.get("filename", "")),
        )
        matches = legacy.get(key, [])
        if len(matches) == 1:
            legacy_backfillable.append(node_id)
        elif len(matches) > 1:
            ambiguous_legacy.append(node_id)
        else:
            orphan_dense.append(node_id)

    return {
        "registry_documents": len(registry_ids),
        "dense_nodes": len(dense_ids),
        "sparse_nodes": len(sparse_ids),
        "orphan_dense": sorted(orphan_dense),
        "sparse_without_dense": sorted(sparse_ids - dense_ids),
        "dense_without_sparse": sorted(dense_ids - sparse_ids),
        "legacy_backfillable": sorted(legacy_backfillable),
        "ambiguous_legacy": sorted(ambiguous_legacy),
        "parent_contexts": len(parent_ids),
        "orphan_parents": orphan_parents,
        "children_without_parent": sorted(referenced_parents - parent_ids),
    }


def prune_orphans(report: dict) -> dict[str, int]:
    """只删除审计已证明无注册文档的 dense 与无 dense 的 sparse。"""
    dense_ids = list(report["orphan_dense"])
    sparse_only = list(report["sparse_without_dense"])
    if dense_ids:
        chain.get_index().vector_store.delete_nodes(node_ids=dense_ids)
    with vector_index_db() as cur:
        all_sparse = sorted(set(dense_ids) | set(sparse_only))
        if all_sparse:
            cur.execute("DELETE FROM node_lexical_stats WHERE node_id = ANY(%s)",
                        (all_sparse,))
        orphan_parents = list(report["orphan_parents"])
        if orphan_parents:
            cur.execute("DELETE FROM parent_contexts WHERE id = ANY(%s)",
                        (orphan_parents,))
    return {"dense_deleted": len(dense_ids), "sparse_deleted": len(all_sparse),
            "parents_deleted": len(orphan_parents)}


def _summary(report: dict) -> str:
    return (
        f"registry={report['registry_documents']} dense={report['dense_nodes']} "
        f"sparse={report['sparse_nodes']} orphan_dense={len(report['orphan_dense'])} "
        f"sparse_without_dense={len(report['sparse_without_dense'])} "
        f"dense_without_sparse={len(report['dense_without_sparse'])} "
        f"legacy_backfillable={len(report['legacy_backfillable'])} "
        f"ambiguous_legacy={len(report['ambiguous_legacy'])}"
        f" parents={report['parent_contexts']}"
        f" orphan_parents={len(report['orphan_parents'])}"
        f" children_without_parent={len(report['children_without_parent'])}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--prune-orphans", action="store_true",
                        help="删除已证明无注册文档的 dense/sparse 节点")
    args = parser.parse_args()
    try:
        report = audit_index()
        deleted = prune_orphans(report) if args.prune_orphans else None
        after = audit_index() if deleted else None
    finally:
        close_all()

    if args.json:
        print(json.dumps({"before": report, "deleted": deleted, "after": after},
                         ensure_ascii=False, indent=2))
    else:
        print("before:", _summary(report))
        if deleted:
            print("deleted:", deleted)
            print("after:", _summary(after))
        elif (report["orphan_dense"] or report["sparse_without_dense"]
              or report["orphan_parents"] or report["children_without_parent"]):
            print("只读审计完成；如确认删除，请显式加 --prune-orphans")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
