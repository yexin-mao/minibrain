"""把上传的字节变成纯文本。**知识库的第一道门，也是最容易静默出错的一道。**

## 为什么单独有这个模块

原来的上传路径是这么一行：

    raw.decode("utf-8", "replace")

传个 PDF 进去，二进制被强行解码成一大片 `\\ufffd` 替换字符，然后**照常切分、
照常向量化、状态标成 `ready`**。用户以为传成功了，检索时永远命中不了。

**这是静默腐化，比报错糟糕得多**：报错至少你知道要换个文件，
而这种失败要等到有人问了才发现，而且会怪到检索头上。

本项目一贯的规矩是「任何失败都必须落 failed 状态，绝不伪装成 ready」
（见 `process_document` 的 docstring）。上传这一步之前漏掉了。

## 怎么判断类型：看内容，不看扩展名

扩展名是**用户可以随便改的**。`report.md` 里装着 PDF、`data.txt` 其实是 docx，
在真实系统里天天发生。所以先看**魔数**（文件头几个字节），
扩展名只在魔数认不出来时当参考。

## 支持到什么程度，以及为什么就到这

| 类型 | 怎么处理 | 取舍 |
|---|---|---|
| 纯文本 / Markdown | 直接解码 | 主力格式 |
| PDF | `pypdf` 抽文字层 | **只抽文字层**，见下 |
| 其他二进制 | **拒绝，落 failed** | 宁可拒收，不可静默腐化 |

★ PDF 只抽文字层，**不做 OCR、不做版面还原、不抽表格**。这是个明确的取舍：

- 扫描件（图片型 PDF）抽不出文字 → **判定为空并拒收**，而不是入库一个空文档
- 双栏排版会串行 → 已知缺陷，写在这里
- 表格会塌成一行文字 → 表格类数据应该走 `table-rag` 那条链路，不是硬塞进向量库

真做到「版面还原 + 表格抽取」要上 MinerU / PP-Structure 那一级的工具，
带 GPU 和模型权重。**那不是这一步该做的事**——先把「能收、收不了会说」做对，
再谈收得多好。
"""

from __future__ import annotations

import io

# 魔数 → 类型。只列真的会遇到的。
_MAGIC: list[tuple[bytes, str]] = [
    (b"%PDF-", "pdf"),
    (b"PK\x03\x04", "zip"),        # docx / xlsx / pptx 本质都是 zip
    (b"\xd0\xcf\x11\xe0", "ole"),  # 老的 .doc / .xls
    (b"\x89PNG", "image"),
    (b"\xff\xd8\xff", "image"),
    (b"GIF8", "image"),
]

# 二进制判定阈值：前 4KB 里有多少比例是"不该出现在文本里"的字节。
# 0.10 是折中值——UTF-8 中文本身有大量高位字节，阈值太低会误伤中文。
_BINARY_RATIO = 0.10
_SNIFF_BYTES = 4096


class UnsupportedFile(Exception):
    """无法可靠地抽出文本。**必须让调用方看见**，不能吞掉。"""


def detect_kind(raw: bytes, filename: str = "") -> str:
    """返回 'text' / 'pdf' / 'zip' / 'ole' / 'image' / 'binary'。

    ★ 先看魔数再看扩展名。扩展名是用户可以随便改的，
      而 `%PDF-` 这五个字节不会说谎。
    """
    head = raw[:8]
    for magic, kind in _MAGIC:
        if head.startswith(magic):
            return kind

    # 魔数认不出来：看内容像不像文本
    sample = raw[:_SNIFF_BYTES]
    if not sample:
        return "text"                      # 空文件交给下游判空
    if b"\x00" in sample:
        return "binary"                    # NUL 字节基本排除文本

    # ★ 判定用的编码集合必须和 load_text 里实际会尝试的**完全一致**。
    #   第一版这里只试 utf-8，而 load_text 里还有 gbk 回退 ——
    #   结果 GBK 文件在判定阶段就被打成 binary，压根走不到那个回退分支。
    #   **判定和解码用两套标准，是这类模块最典型的失效方式。**
    if not _decodable(sample):
        return "binary"

    control = sum(1 for b in sample if b < 9 or 13 < b < 32)
    return "binary" if control / len(sample) > _BINARY_RATIO else "text"


# 尝试顺序 = load_text 里的尝试顺序。改一处必须改另一处。
TEXT_ENCODINGS = ("utf-8", "gbk")


def _decodable(sample: bytes) -> bool:
    """样本能不能被任一受支持的文本编码解开。

    ★ 末尾截断在多字节字符中间是正常的（我们只取了前 4KB），
      所以解不开时再退 3 个字节试一次，避免把正常文本误判成二进制。
    """
    for encoding in TEXT_ENCODINGS:
        for candidate in (sample, sample[:-3]):
            try:
                candidate.decode(encoding)
                return True
            except UnicodeDecodeError:
                continue
    return False


def _extract_pdf(raw: bytes, filename: str) -> str:
    try:
        from pypdf import PdfReader                       # noqa: PLC0415
    except ImportError as exc:                            # pragma: no cover
        raise UnsupportedFile(
            "解析 PDF 需要 pypdf，请先 `uv sync`") from exc

    try:
        reader = PdfReader(io.BytesIO(raw))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:                              # noqa: BLE001
        raise UnsupportedFile(f"PDF 解析失败：{type(exc).__name__}") from exc

    text = "\n\n".join(p.strip() for p in pages if p.strip())
    if not text.strip():
        # ★ 抽不出文字的 PDF 绝大多数是扫描件（整页是图）。
        #   入库一个空文档等于制造一条永远命中不了的"幽灵记录"，
        #   还会让文档总数看起来是对的 —— 这种假象比报错难查得多。
        raise UnsupportedFile(
            f"{filename} 抽不出任何文字。多半是扫描件（图片型 PDF），"
            "当前不支持 OCR。")
    return text


def load_text(raw: bytes, filename: str = "") -> tuple[str, str]:
    """字节 → (纯文本, 用了哪种解析方式)。解析不了就抛 UnsupportedFile。

    ★ 返回值带上"用了哪种方式"，是为了让它能一路存进 documents 表 ——
      排查"这篇怎么检索不到"时，第一个要看的就是它当初是怎么被解析的。
    """
    kind = detect_kind(raw, filename)

    if kind == "pdf":
        return _extract_pdf(raw, filename), "pdf"

    if kind == "text":
        # errors="strict"：到这一步已经确认是文本了，再出错说明判定有问题，
        # 应该暴露出来而不是用替换字符掩盖 —— 掩盖正是原来那个 bug 的根源。
        try:
            return raw.decode("utf-8"), "text"
        except UnicodeDecodeError:
            # 少数遗留文件是 GBK。试一次，再不行就拒收。
            try:
                return raw.decode("gbk"), "text(gbk)"
            except UnicodeDecodeError as exc:
                raise UnsupportedFile(
                    f"{filename} 不是 UTF-8 也不是 GBK 编码") from exc

    hint = {
        "zip": "Office 文档（docx/xlsx/pptx）暂不支持，请导出为 PDF 或纯文本",
        "ole": "老版 Office 文档（doc/xls）暂不支持，请另存为 PDF 或纯文本",
        "image": "图片暂不支持，当前没有 OCR",
    }.get(kind, "无法识别的二进制文件")
    raise UnsupportedFile(f"{filename}：{hint}")
