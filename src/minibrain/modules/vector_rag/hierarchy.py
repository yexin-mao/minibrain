"""结构感知的 small-to-big 切分。

子块用于 dense/sparse 召回，父块只用于命中后的上下文展开。父块不做 embedding，
因此检索粒度和生成粒度可以解耦，而不会把向量数量翻倍。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, TextNode


_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_PAGE_HEADING = re.compile(r"(?:^| > )第\s*(\d+)\s*页$")
_SLIDE_HEADING = re.compile(r"(?:^| > )幻灯片\s*(\d+)$")


@dataclass(frozen=True)
class ParentContext:
    id: str
    document_id: str | None
    source_name: str
    owner_id: str
    visibility: str
    filename: str
    ordinal: int
    heading_path: str
    content: str
    content_hash: str


def markdown_sections(text: str) -> list[tuple[str, str]]:
    """按 ATX 标题切章节并保留标题层级；代码围栏里的 ``#`` 不算标题。"""
    sections: list[tuple[str, str]] = []
    path: list[str] = []
    body: list[str] = []
    in_fence = False

    def flush() -> None:
        content = "\n".join(body).strip()
        # 连续标题（例如 H1 文档名后立刻 H2）没有独立正文。把这种标题单独
        # 建成 chunk 会在长文档中制造大量“只含标题”的高分噪声；标题信息已经
        # 包含在后续正文块的 heading_path 中，无需重复建空块。
        if content:
            sections.append((" > ".join(path), content))

    for line in text.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
            body.append(line)
            continue
        match = None if in_fence else _HEADING.match(line)
        if not match:
            body.append(line)
            continue
        flush()
        body = []
        level = len(match.group(1))
        title = match.group(2).strip().rstrip("#").strip()
        path = path[:level - 1]
        path.append(title)

    flush()
    # 只有标题、没有正文的极端文档仍保留原文，避免上传成功后零 chunk。
    return sections or [("", text.strip())]


def _split(text: str, size: int, overlap: int = 0) -> list[str]:
    if not text.strip():
        return []
    splitter = SentenceSplitter(chunk_size=size, chunk_overlap=overlap, tokenizer=list)
    return [node.get_content().strip() for node in
            splitter.get_nodes_from_documents([Document(text=text)])
            if node.get_content().strip()]


def _location_metadata(heading_path: str) -> dict[str, int]:
    """Turn parser-created page/slide headings into typed, filterable metadata."""
    page = _PAGE_HEADING.search(heading_path)
    if page:
        return {"page_number": int(page.group(1))}
    slide = _SLIDE_HEADING.search(heading_path)
    if slide:
        return {"slide_number": int(slide.group(1))}
    return {}


def build_hierarchy(*, filename: str, text: str, source_name: str,
                    visibility: str, owner_id: str, document_id: str | None,
                    child_size: int, child_overlap: int,
                    parent_size: int,
                    document_metadata: dict[str, object]
                    ) -> tuple[list[TextNode], list[ParentContext]]:
    """生成只参与召回的子节点，以及只参与上下文展开的父块。"""
    children: list[TextNode] = []
    parents: list[ParentContext] = []
    child_ordinal = 0

    for heading_path, section_body in markdown_sections(text):
        # 标题本身也是信息；空章节仍保留标题作为可检索文本。
        section_body = section_body or heading_path
        for parent_text in _split(section_body, parent_size):
            display = (f"标题路径：{heading_path}\n\n{parent_text}"
                       if heading_path else parent_text)
            normalized = " ".join(display.split())
            content_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            parent_ordinal = len(parents)
            identity = (f"{document_id or ''}\0{owner_id}\0{source_name}\0{filename}"
                        f"\0{parent_ordinal}\0{content_hash}")
            parent_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
            parents.append(ParentContext(
                id=parent_id, document_id=document_id, source_name=source_name,
                owner_id=owner_id, visibility=visibility, filename=filename,
                ordinal=parent_ordinal, heading_path=heading_path,
                content=display, content_hash=content_hash,
            ))

            # 子块也携带标题路径，属于零 LLM 成本的 contextual retrieval。
            for child_text in _split(parent_text, child_size, child_overlap):
                searchable = (f"标题路径：{heading_path}\n\n{child_text}"
                              if heading_path else child_text)
                metadata = {
                    "filename": filename,
                    "source_name": source_name,
                    "owner_id": owner_id,
                    "visibility": visibility,
                    **document_metadata,
                    "ordinal": child_ordinal,
                    "node_level": "child",
                    "parent_context_id": parent_id,
                    "parent_ordinal": parent_ordinal,
                    "heading_path": heading_path,
                    **_location_metadata(heading_path),
                    "chunk_content_hash": hashlib.sha256(
                        " ".join(searchable.split()).encode("utf-8")).hexdigest(),
                }
                if document_id is not None:
                    metadata["registry_document_id"] = document_id
                node = TextNode(text=searchable, metadata=metadata)
                node.excluded_embed_metadata_keys = list(metadata.keys())
                children.append(node)
                child_ordinal += 1

    return children, parents
