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
| DOCX | 按块级 XML 顺序转 Markdown | 保留标题、列表、表格 |
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
import re
import zipfile
from html.parser import HTMLParser
from pathlib import PurePosixPath
from xml.etree import ElementTree as ET

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
_DOCX_MAX_ENTRIES = 5000
_DOCX_MAX_UNCOMPRESSED_BYTES = 100 * 1024 * 1024
_OFFICE_MAX_ENTRIES = 5000
_OFFICE_MAX_UNCOMPRESSED_BYTES = 100 * 1024 * 1024


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
            if kind == "zip":
                return _detect_zip_kind(raw)
            return kind

    # HTML 没有可靠魔数，但开头标签比扩展名可靠；允许 BOM、空白和 XML 声明。
    html_head = raw[:2048].decode("utf-8", "ignore").lstrip("\ufeff \t\r\n").casefold()
    if (html_head.startswith("<!doctype html") or html_head.startswith("<html")
            or (html_head.startswith("<?xml") and "<html" in html_head)):
        return "html"

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


def _detect_zip_kind(raw: bytes) -> str:
    """识别 OOXML 容器；损坏或未知 ZIP 留给调用方明确拒绝。"""
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            names = set(archive.namelist())
    except (zipfile.BadZipFile, OSError):
        return "zip"
    if "word/document.xml" in names:
        return "docx"
    if "xl/workbook.xml" in names:
        return "xlsx"
    if "ppt/presentation.xml" in names:
        return "pptx"
    return "zip"


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

    # 页码是 PDF 最稳定、最容易由面试官和用户复核的出处。转成 Markdown 标题后，
    # 现有 heading-aware splitter 会自然保证跨页边界，并把页码带入 chunk metadata。
    text = "\n\n".join(
        f"# 第 {index} 页\n\n{page.strip()}"
        for index, page in enumerate(pages, start=1) if page.strip()
    )
    if not text.strip():
        # ★ 抽不出文字的 PDF 绝大多数是扫描件（整页是图）。
        #   入库一个空文档等于制造一条永远命中不了的"幽灵记录"，
        #   还会让文档总数看起来是对的 —— 这种假象比报错难查得多。
        raise UnsupportedFile(
            f"{filename} 抽不出任何文字。多半是扫描件（图片型 PDF），"
            "当前不支持 OCR。")
    return text


def _validate_docx_archive(raw: bytes, filename: str) -> None:
    """在 python-docx 解压前限制条目数和声明的解压体积。"""
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            infos = archive.infolist()
    except (zipfile.BadZipFile, OSError) as exc:
        raise UnsupportedFile(f"{filename}：DOCX 压缩包损坏") from exc
    if len(infos) > _DOCX_MAX_ENTRIES:
        raise UnsupportedFile(f"{filename}：DOCX 内部文件数量异常")
    total = sum(item.file_size for item in infos)
    if total > _DOCX_MAX_UNCOMPRESSED_BYTES:
        raise UnsupportedFile(f"{filename}：DOCX 解压后超过 100 MB 安全上限")


def _validate_office_archive(raw: bytes, filename: str, kind: str) -> None:
    """限制 OOXML ZIP 数量和声明体积，避免 zip bomb。"""
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            infos = archive.infolist()
    except (zipfile.BadZipFile, OSError) as exc:
        raise UnsupportedFile(f"{filename}：{kind} 压缩包损坏") from exc
    if len(infos) > _OFFICE_MAX_ENTRIES:
        raise UnsupportedFile(f"{filename}：{kind} 内部文件数量异常")
    if sum(item.file_size for item in infos) > _OFFICE_MAX_UNCOMPRESSED_BYTES:
        raise UnsupportedFile(f"{filename}：{kind} 解压后超过 100 MB 安全上限")


def _heading_level(paragraph) -> int | None:
    """同时识别内置 Heading 样式和带 outlineLvl 的自定义标题样式。"""
    style = paragraph.style
    name = (style.name or "") if style is not None else ""
    match = re.match(r"^(?:Heading|标题)\s*([1-6])$", name, re.IGNORECASE)
    if match:
        return int(match.group(1))
    if name.casefold() == "title":
        return 1
    if style is not None and style.element.pPr is not None:
        outline = style.element.pPr.find(
            "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}outlineLvl")
        if outline is not None:
            value = outline.get(
                "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val")
            if value is not None and value.isdigit() and int(value) < 6:
                return int(value) + 1
    return None


def _list_prefix(paragraph) -> str | None:
    """返回 Markdown 列表前缀；保留层级，不伪造连续序号。"""
    style_name = (paragraph.style.name or "") if paragraph.style else ""
    ppr = paragraph._p.pPr
    numpr = ppr.numPr if ppr is not None else None
    level = 0
    if numpr is not None and numpr.ilvl is not None:
        level = int(numpr.ilvl.val)
    indent = "  " * min(level, 6)
    lowered = style_name.casefold()
    if "bullet" in lowered or "项目符号" in style_name:
        return f"{indent}- "
    if "number" in lowered or "编号" in style_name or numpr is not None:
        return f"{indent}1. "
    return None


def _escape_table_cell(text: str) -> str:
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>").strip()


def _table_to_markdown(table) -> str:
    rows = [[_escape_table_cell(cell.text) for cell in row.cells] for row in table.rows]
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    if width == 0:
        return ""
    header = rows[0]
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join(["---"] * width) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows[1:])
    return "\n".join(lines)


def _extract_docx(raw: bytes, filename: str) -> str:
    _validate_docx_archive(raw, filename)
    try:
        from docx import Document as DocxDocument                # noqa: PLC0415
        from docx.table import Table                             # noqa: PLC0415
        from docx.text.paragraph import Paragraph                # noqa: PLC0415
    except ImportError as exc:                                   # pragma: no cover
        raise UnsupportedFile("解析 DOCX 需要 python-docx，请先 `uv sync`") from exc

    try:
        document = DocxDocument(io.BytesIO(raw))
        blocks: list[str] = []
        # iter_inner_content 保证段落与表格仍处于用户在 Word 里看到的顺序。
        for block in document.iter_inner_content():
            if isinstance(block, Paragraph):
                text = block.text.strip()
                if not text:
                    continue
                level = _heading_level(block)
                prefix = _list_prefix(block)
                if level:
                    blocks.append(f"{'#' * level} {text}")
                elif prefix:
                    blocks.append(f"{prefix}{text}")
                else:
                    blocks.append(text)
            elif isinstance(block, Table):
                markdown = _table_to_markdown(block)
                if markdown:
                    blocks.append(markdown)
    except Exception as exc:                                     # noqa: BLE001
        raise UnsupportedFile(f"DOCX 解析失败：{type(exc).__name__}") from exc

    text = "\n\n".join(blocks).strip()
    if not text:
        raise UnsupportedFile(
            f"{filename} 没有可提取的正文文字；图片型 Word 当前不支持 OCR。")
    return text


class _MarkdownHTMLParser(HTMLParser):
    """只保留用户可见正文；脚本、样式和导航噪声不进入知识库。"""

    _SKIP = {"script", "style", "noscript", "svg", "canvas", "template"}
    _BLOCK = {"p", "div", "section", "article", "main", "header", "footer",
              "blockquote", "pre", "tr", "table"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip_depth = 0
        self.list_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.casefold()
        if tag in self._SKIP:
            self.skip_depth += 1
            return
        if self.skip_depth:
            return
        if re.fullmatch(r"h[1-6]", tag):
            self.parts.append(f"\n\n{'#' * int(tag[1])} ")
        elif tag in {"ul", "ol"}:
            self.list_depth += 1
            self.parts.append("\n")
        elif tag == "li":
            self.parts.append(f"\n{'  ' * max(self.list_depth - 1, 0)}- ")
        elif tag == "br":
            self.parts.append("\n")
        elif tag in self._BLOCK:
            self.parts.append("\n\n")
        elif tag in {"td", "th"}:
            self.parts.append(" | ")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if tag in self._SKIP:
            self.skip_depth = max(0, self.skip_depth - 1)
            return
        if self.skip_depth:
            return
        if tag in {"ul", "ol"}:
            self.list_depth = max(0, self.list_depth - 1)
        if re.fullmatch(r"h[1-6]", tag) or tag in self._BLOCK:
            self.parts.append("\n\n")

    def handle_data(self, data: str) -> None:
        if not self.skip_depth:
            clean = re.sub(r"\s+", " ", data)
            if clean.strip():
                self.parts.append(clean)


def _extract_html(raw: bytes, filename: str) -> str:
    decoded = None
    for encoding in TEXT_ENCODINGS:
        try:
            decoded = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if decoded is None:
        raise UnsupportedFile(f"{filename}：HTML 不是 UTF-8 也不是 GBK 编码")
    parser = _MarkdownHTMLParser()
    try:
        parser.feed(decoded)
        parser.close()
    except Exception as exc:  # noqa: BLE001
        raise UnsupportedFile(f"HTML 解析失败：{type(exc).__name__}") from exc
    text = "".join(parser.parts)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        raise UnsupportedFile(f"{filename} 没有可提取的 HTML 正文")
    return text


def _slide_number(name: str) -> int:
    match = re.search(r"slide(\d+)\.xml$", name)
    return int(match.group(1)) if match else 10**9


def _extract_pptx(raw: bytes, filename: str) -> str:
    """按幻灯片顺序抽取文字层；图片型演示文稿仍明确拒绝。"""
    _validate_office_archive(raw, filename, "PPTX")
    namespace = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
    slides: list[str] = []
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            names = sorted(
                (name for name in archive.namelist()
                 if PurePosixPath(name).parent.as_posix() == "ppt/slides"
                 and re.fullmatch(r"slide\d+\.xml", PurePosixPath(name).name)),
                key=_slide_number,
            )
            for index, name in enumerate(names, start=1):
                root = ET.fromstring(archive.read(name))
                paragraphs = []
                for paragraph in root.findall(".//a:p", namespace):
                    text = "".join(
                        node.text or "" for node in paragraph.findall(".//a:t", namespace)
                    ).strip()
                    if text:
                        paragraphs.append(text)
                if paragraphs:
                    slides.append(f"# 幻灯片 {index}\n\n" + "\n\n".join(paragraphs))
    except (zipfile.BadZipFile, ET.ParseError, KeyError, OSError) as exc:
        raise UnsupportedFile(f"PPTX 解析失败：{type(exc).__name__}") from exc
    if not slides:
        raise UnsupportedFile(
            f"{filename} 没有可提取的文字；图片型 PPTX 当前不支持 OCR。")
    return "\n\n".join(slides)


def load_text(raw: bytes, filename: str = "") -> tuple[str, str]:
    """字节 → (纯文本, 用了哪种解析方式)。解析不了就抛 UnsupportedFile。

    ★ 返回值带上"用了哪种方式"，是为了让它能一路存进 documents 表 ——
      排查"这篇怎么检索不到"时，第一个要看的就是它当初是怎么被解析的。
    """
    kind = detect_kind(raw, filename)

    if kind == "pdf":
        return _extract_pdf(raw, filename), "pdf→markdown(page-aware)"

    if kind == "docx":
        return _extract_docx(raw, filename), "docx→markdown"

    if kind == "pptx":
        return _extract_pptx(raw, filename), "pptx→markdown"

    if kind == "html":
        return _extract_html(raw, filename), "html→markdown"

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
        "zip": "无法识别的 ZIP/Office 文档；当前支持 DOCX、PPTX",
        "xlsx": "XLSX 请上传到表格链路，不要作为文档切分",
        "ole": "老版 Office 文档（doc/xls）暂不支持；请将 .doc 另存为 .docx",
        "image": "图片暂不支持，当前没有 OCR",
    }.get(kind, "无法识别的二进制文件")
    raise UnsupportedFile(f"{filename}：{hint}")
