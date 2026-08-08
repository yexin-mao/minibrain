"""上传解析。守的是「**不能静默腐化**」这条线。

## 修的是什么 bug

Web 上传原来是这么一行：

    raw.decode("utf-8", "replace")

传个 PDF 进去，二进制被强行解码成一大片 `\\ufffd` 替换字符，
然后**照常切分、照常向量化、状态标成 `ready`**。
用户以为传成功了，检索时永远命中不了，而且会怪到检索头上。

本项目最核心的规矩是「任何失败都必须落 failed，绝不伪装成 ready」，
它写在 `process_document` 上——**但上传发生在它之前，漏掉了**。

## 这些测试为什么这么写

不测「PDF 能解析出多好的文字」（那取决于 pypdf，测它等于测别人的库），
只测**边界行为**：

- 认得出来的，要走对分支
- 认不出来的，要**明确拒绝**，不能悄悄放行
- 拒绝时要**留下一条 failed 记录**，用户在列表里看得见原因
"""

from __future__ import annotations

import pytest

from minibrain.modules.vector_rag.loaders import (
    UnsupportedFile, detect_kind, load_text,
)


# ---------------------------------------------------------------- 类型识别

def test_plain_text_is_text():
    assert detect_kind("# 标题\n\n正文内容".encode()) == "text"


def test_pdf_detected_by_magic_not_extension():
    """★ 扩展名是用户可以随便改的，魔数不会说谎。

    真实系统里「report.md 其实是 PDF」天天发生。
    只看扩展名的话，这种文件会走文本分支被腐化掉。
    """
    assert detect_kind(b"%PDF-1.7\n...", filename="report.md") == "pdf"


def test_text_extension_on_binary_content_is_still_binary():
    """反过来也要成立：叫 .txt 不代表它是文本。"""
    assert detect_kind(b"\x00\x01\x02\x03" * 100, filename="notes.txt") == "binary"


@pytest.mark.parametrize("head,kind", [
    (b"PK\x03\x04", "zip"),          # docx/xlsx/pptx
    (b"\xd0\xcf\x11\xe0", "ole"),    # 老 doc/xls
    (b"\x89PNG\r\n\x1a\n", "image"),
    (b"\xff\xd8\xff\xe0", "image"),
])
def test_common_binary_formats_are_recognized(head, kind):
    assert detect_kind(head + b"\x00" * 200) == kind


def test_chinese_text_is_not_mistaken_for_binary():
    """★ 这条是防误伤的。

    UTF-8 中文本身有大量高位字节，二进制判定阈值定得太激进就会把
    中文文档判成二进制——那会让主力语料全部传不上去。
    """
    assert detect_kind("公司差旅报销标准：住宿每晚不超过 400 元。".encode() * 50) == "text"


def test_empty_file_is_text_not_binary():
    """空文件交给下游判空，不在这里当成二进制拒掉。"""
    assert detect_kind(b"") == "text"


# ---------------------------------------------------------------- 解析

def test_load_text_returns_how_it_was_parsed():
    """★ 返回「用了哪种解析方式」不是多余的。

    排查「这篇怎么检索不到」时，第一个要看的就是它当初被怎么解析的：
    传了 PDF 却显示 text，说明走错了分支。这个值会一路存进 documents 表。
    """
    text, how = load_text("正文".encode(), "a.md")
    assert text == "正文"
    assert how == "text"


def test_gbk_file_is_recovered_not_rejected():
    """少数遗留文件是 GBK。试一次再放弃，比直接拒收友好。"""
    text, how = load_text("公司制度".encode("gbk"), "old.txt")
    assert text == "公司制度"
    assert how == "text(gbk)"


@pytest.mark.parametrize("raw,hint", [
    (b"PK\x03\x04" + b"\x00" * 100, "Office"),
    (b"\xd0\xcf\x11\xe0" + b"\x00" * 100, "老版 Office"),
    (b"\x89PNG\r\n\x1a\n" + b"\x00" * 100, "图片"),
])
def test_unsupported_binary_is_rejected_with_a_useful_hint(raw, hint):
    """★ 拒绝时要说人话：告诉用户**能怎么办**，不是只说「不支持」。"""
    with pytest.raises(UnsupportedFile, match=hint):
        load_text(raw, "x.bin")


def test_corrupt_pdf_is_rejected_not_silently_emptied():
    """假装成 PDF 的垃圾要报错，不能入库一个空文档。"""
    with pytest.raises(UnsupportedFile):
        load_text(b"%PDF-1.7\n" + b"\x00" * 500, "broken.pdf")


def test_the_original_bug_would_have_passed_silently():
    """★★ 这个测试是用来记录 bug 本身的，不是测新代码。

    原来的做法是 raw.decode("utf-8", "replace")：**任何字节都能"成功"**，
    产出一堆替换字符，然后被当成正常文档入库。
    这里把那个行为复现一遍，说明「为什么必须换掉它」。
    """
    png = b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 4

    # 老做法：永远成功，产出垃圾
    corrupted = png.decode("utf-8", "replace")
    assert corrupted                       # 有内容
    assert "�" in corrupted           # 而且全是替换字符
    assert len(corrupted) > 100            # 长度还不小，看起来"像"一篇文档

    # 新做法：明确拒绝
    with pytest.raises(UnsupportedFile):
        load_text(png, "screenshot.png")
