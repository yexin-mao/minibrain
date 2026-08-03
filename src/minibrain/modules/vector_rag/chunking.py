"""按字符切分，尽量在段落边界断开。

先不做 token 化切分：它要拖进一个 tokenizer 依赖和一份离线词表，
而字符切分够简单、够可预测，适合当基线。

注意这是**假设，不是结论**：本项目还没有做过「字符 vs token 切分」的对照实验
（见 eval/RESULTS.md 的"还没测的"一节）。要推翻或确认它，得先有评测数据。
"""

from __future__ import annotations


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buffer = ""

    for para in paragraphs:
        # 单段就超长：先冲掉缓冲区，再把这段硬切。
        if len(para) > chunk_size:
            if buffer:
                chunks.append(buffer)
                buffer = ""
            step = chunk_size - overlap
            for start in range(0, len(para), step):
                piece = para[start : start + chunk_size]
                if piece.strip():
                    chunks.append(piece.strip())
            continue

        candidate = f"{buffer}\n\n{para}" if buffer else para
        if len(candidate) <= chunk_size:
            buffer = candidate
        else:
            chunks.append(buffer)
            # 带 overlap 起新块：从上一块尾部截一段接上，避免边界处语义被切断。
            tail = buffer[-overlap:] if overlap else ""
            buffer = f"{tail}\n\n{para}".strip() if tail else para

    if buffer:
        chunks.append(buffer)

    return chunks
