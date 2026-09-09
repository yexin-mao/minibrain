"""NanoBEIR 两个公开检索任务的数据适配层；不连接数据库、不调用模型。"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    repo_id: str
    revision: str
    split: str = "train"


@dataclass(frozen=True)
class CorpusDocument:
    doc_id: str
    text: str
    title: str = ""

    @property
    def content(self) -> str:
        return f"# {self.title}\n\n{self.text}" if self.title else self.text


@dataclass(frozen=True)
class NanoBeirDataset:
    spec: DatasetSpec
    corpus: tuple[CorpusDocument, ...]
    queries: Mapping[str, str]
    qrels: Mapping[str, frozenset[str]]


SPECS = {
    "NanoNQRetrieval": DatasetSpec(
        name="NanoNQRetrieval",
        repo_id="mteb/NanoNQRetrieval",
        revision="e3973b405feb9e94d07fd971d690d24d7f45264a",
    ),
    "NanoHotpotQARetrieval": DatasetSpec(
        name="NanoHotpotQARetrieval",
        repo_id="mteb/NanoHotpotQARetrieval",
        revision="a38a2615f29adba8b47a42905031aa9c137d8fae",
    ),
}


def document_filename(task: str, doc_id: str) -> str:
    """把任意 corpus id 变成可逆、安全且无碰撞的虚拟文件名。"""
    encoded = base64.urlsafe_b64encode(doc_id.encode("utf-8")).decode("ascii").rstrip("=")
    return f"{task.lower()}--{encoded}.md"


def _rows_by_id(rows: Iterable[Mapping], *, id_key: str, text_key: str,
                label: str) -> dict[str, Mapping]:
    indexed: dict[str, Mapping] = {}
    for row in rows:
        if id_key not in row or text_key not in row:
            raise ValueError(f"{label} 缺少列 {id_key!r} 或 {text_key!r}")
        row_id, value = str(row[id_key]), str(row[text_key] or "").strip()
        if not row_id or not value:
            raise ValueError(f"{label} 含空 id 或空文本")
        if row_id in indexed:
            raise ValueError(f"{label} 含重复 id: {row_id}")
        indexed[row_id] = row
    return indexed


def build_dataset(spec: DatasetSpec, corpus_rows: Iterable[Mapping],
                  query_rows: Iterable[Mapping], qrel_rows: Iterable[Mapping]) -> NanoBeirDataset:
    """从普通字典构建并校验数据集，便于不安装 pyarrow 也能单测。"""
    corpus_index = _rows_by_id(corpus_rows, id_key="_id", text_key="text", label="corpus")
    query_index = _rows_by_id(query_rows, id_key="_id", text_key="text", label="queries")
    relevant: dict[str, set[str]] = {query_id: set() for query_id in query_index}
    for row in qrel_rows:
        required = {"query-id", "corpus-id", "score"}
        if not required <= row.keys():
            raise ValueError(f"qrels 缺少列: {sorted(required - row.keys())}")
        if float(row["score"]) <= 0:
            continue
        query_id, doc_id = str(row["query-id"]), str(row["corpus-id"])
        if query_id not in query_index:
            raise ValueError(f"qrels 引用了不存在的 query: {query_id}")
        if doc_id not in corpus_index:
            raise ValueError(f"qrels 引用了不存在的 corpus document: {doc_id}")
        relevant[query_id].add(doc_id)
    empty = [query_id for query_id, doc_ids in relevant.items() if not doc_ids]
    if empty:
        raise ValueError(f"有 query 没有正相关文档: {empty[:3]}")

    corpus = tuple(CorpusDocument(
        doc_id=doc_id,
        text=str(row["text"]).strip(),
        title=str(row.get("title") or "").strip(),
    ) for doc_id, row in corpus_index.items())
    queries = {query_id: str(row["text"]).strip() for query_id, row in query_index.items()}
    return NanoBeirDataset(
        spec=spec, corpus=corpus, queries=queries,
        qrels={query_id: frozenset(doc_ids) for query_id, doc_ids in relevant.items()},
    )


def _read_parquet_rows(root: Path, section: str, split: str) -> list[dict]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "公开评测依赖未安装，请先运行 `uv sync --extra public-eval`") from exc
    files = sorted((root / section).glob(f"{split}*.parquet"))
    if not files:
        raise FileNotFoundError(f"{root / section} 下找不到 {split} parquet")
    rows: list[dict] = []
    for path in files:
        rows.extend(pq.read_table(path).to_pylist())
    return rows


def download_and_load(spec: DatasetSpec, cache_dir: Path) -> NanoBeirDataset:
    """按固定 commit 下载官方数据，并读取 corpus / queries / qrels。"""
    local_dir = cache_dir / spec.name
    required = ("corpus", "queries", "qrels")
    revision_record = local_dir / ".cache" / "huggingface" / "trees" / f"{spec.revision}.json"
    has_local_snapshot = revision_record.is_file() and all(
        any((local_dir / section).glob(f"{spec.split}*.parquet"))
        for section in required)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "公开评测依赖未安装，请先运行 `uv sync --extra public-eval`") from exc
    snapshot = local_dir if has_local_snapshot else Path(snapshot_download(
        repo_id=spec.repo_id, repo_type="dataset", revision=spec.revision,
        local_dir=str(local_dir),
        allow_patterns=["corpus/*.parquet", "queries/*.parquet", "qrels/*.parquet", "README.md"],
    ))
    return build_dataset(
        spec,
        _read_parquet_rows(snapshot, "corpus", spec.split),
        _read_parquet_rows(snapshot, "queries", spec.split),
        _read_parquet_rows(snapshot, "qrels", spec.split),
    )


def aggregate_document_ranking(locations: Iterable[str],
                               filename_to_doc_id: Mapping[str, str]) -> list[str]:
    """chunk 排名聚合成 document 排名：每篇原文只保留最高排名的 chunk。"""
    ranking: list[str] = []
    seen: set[str] = set()
    for location in locations:
        filename = location.rsplit(" #", 1)[0]
        doc_id = filename_to_doc_id.get(filename)
        if doc_id is not None and doc_id not in seen:
            seen.add(doc_id)
            ranking.append(doc_id)
    return ranking
