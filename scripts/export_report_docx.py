from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MARKDOWN = ROOT / "docs" / "STAD多组学分型实验报告.md"
DEFAULT_OUTPUT = ROOT / "docs" / "STAD多组学分型实验报告.docx"

LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
INLINE_CODE_RE = re.compile(r"`([^`]+)`")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def set_run_font(run, font_name: str = "宋体", size: int | None = None, bold: bool | None = None) -> None:
    run.font.name = font_name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), font_name)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold


def set_document_defaults(doc: Document) -> None:
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "宋体"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    normal.font.size = Pt(11)

    for name, size in [("Heading 1", 16), ("Heading 2", 14), ("Heading 3", 12)]:
        style = styles[name]
        style.font.name = "黑体"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor(0, 0, 0)

    for section in doc.sections:
        section.top_margin = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin = Cm(2.8)
        section.right_margin = Cm(2.8)


def add_page_number(paragraph) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    fld_char_begin = OxmlElement("w:fldChar")
    fld_char_begin.set(qn("w:fldCharType"), "begin")
    instr_text = OxmlElement("w:instrText")
    instr_text.set(qn("xml:space"), "preserve")
    instr_text.text = "PAGE"
    fld_char_end = OxmlElement("w:fldChar")
    fld_char_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_char_begin)
    run._r.append(instr_text)
    run._r.append(fld_char_end)


def add_toc(paragraph) -> None:
    run = paragraph.add_run()
    fld_char_begin = OxmlElement("w:fldChar")
    fld_char_begin.set(qn("w:fldCharType"), "begin")
    instr_text = OxmlElement("w:instrText")
    instr_text.set(qn("xml:space"), "preserve")
    instr_text.text = r'TOC \o "1-3" \h \z \u'
    fld_char_separate = OxmlElement("w:fldChar")
    fld_char_separate.set(qn("w:fldCharType"), "separate")
    placeholder = OxmlElement("w:t")
    placeholder.text = "右键更新域以生成目录"
    fld_char_end = OxmlElement("w:fldChar")
    fld_char_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_char_begin)
    run._r.append(instr_text)
    run._r.append(fld_char_separate)
    run._r.append(placeholder)
    run._r.append(fld_char_end)


def split_markdown_table(lines: list[str], start: int) -> tuple[list[list[str]], int] | None:
    table_lines: list[str] = []
    i = start
    while i < len(lines) and lines[i].strip().startswith("|") and lines[i].strip().endswith("|"):
        table_lines.append(lines[i].strip())
        i += 1
    if len(table_lines) < 2:
        return None
    rows = []
    for line in table_lines:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if all(set(cell.replace(" ", "")) <= {"-", ":"} for cell in cells):
            continue
        rows.append(cells)
    return rows, i


def add_inline_runs(paragraph, text: str) -> None:
    text = LINK_RE.sub(lambda match: f"{match.group(1)}（{match.group(2)}）", text)
    pos = 0
    for match in INLINE_CODE_RE.finditer(text):
        if match.start() > pos:
            run = paragraph.add_run(text[pos : match.start()])
            set_run_font(run)
        code_run = paragraph.add_run(match.group(1))
        set_run_font(code_run, font_name="Consolas")
        code_run.font.size = Pt(10.5)
        pos = match.end()
    if pos < len(text):
        run = paragraph.add_run(text[pos:])
        set_run_font(run)


def resolve_image_path(path_text: str) -> Path | None:
    path_text = path_text.strip()
    if not path_text.lower().endswith(".png"):
        return None
    path = Path(path_text)
    if not path.is_absolute():
        path = ROOT / path
    return path if path.exists() else None


def maybe_add_image(doc: Document, text: str) -> bool:
    added = False
    for label, path_text in LINK_RE.findall(text):
        image_path = resolve_image_path(path_text)
        if image_path is None:
            continue
        caption = doc.add_paragraph()
        caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = caption.add_run(label)
        set_run_font(run, font_name="宋体", size=10, bold=True)
        image_para = doc.add_paragraph()
        image_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        image_para.add_run().add_picture(str(image_path), width=Cm(14.5))
        added = True
    return added


def add_cover(doc: Document, title: str, metadata_lines: list[str]) -> None:
    for _ in range(3):
        doc.add_paragraph()
    title_para = doc.add_paragraph()
    title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title_para.add_run(title)
    set_run_font(title_run, font_name="黑体", size=20, bold=True)

    doc.add_paragraph()
    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle_run = subtitle.add_run("生物信息学实验课正式报告")
    set_run_font(subtitle_run, font_name="黑体", size=15, bold=True)

    for _ in range(2):
        doc.add_paragraph()
    table = doc.add_table(rows=0, cols=2)
    table.style = "Table Grid"
    for line in metadata_lines:
        if "：" not in line:
            continue
        key, value = line.split("：", 1)
        row = table.add_row().cells
        row[0].text = key.strip()
        row[1].text = INLINE_CODE_RE.sub(r"\1", value.strip())
    placeholders = [
        ("抽签序号", "待填写"),
        ("课题组长", "待填写"),
        ("成员人数", "待填写"),
        ("完成日期", "2026 年 4 月"),
    ]
    for key, value in placeholders:
        row = table.add_row().cells
        row[0].text = key
        row[1].text = value
    for row in table.rows:
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                for run in paragraph.runs:
                    set_run_font(run)

    doc.add_paragraph()
    note = doc.add_paragraph()
    note.alignment = WD_ALIGN_PARAGRAPH.CENTER
    note_run = note.add_run("提交前请将封面与第 9 节成员信息表中的占位内容替换为真实信息。")
    set_run_font(note_run, size=10)
    doc.add_page_break()


def add_member_table_reminder(doc: Document) -> None:
    paragraph = doc.add_paragraph()
    run = paragraph.add_run("成员信息表提示：当前 Word 已保留作者、分工、贡献度三列表格模板。提交前请补齐真实小组成员信息。")
    set_run_font(run, size=10, bold=True)


def markdown_to_docx(markdown_path: Path, output_path: Path) -> None:
    text = markdown_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    title = "基于 TCGA-STAD 多组学数据的胃腺癌分型研究实验报告"
    metadata_lines: list[str] = []
    body_start = 0
    if lines and lines[0].startswith("# "):
        title = lines[0].removeprefix("# ").strip()
        body_start = 1
    for line in lines[body_start:]:
        if line.startswith("## "):
            break
        if line.strip() and not line.startswith(">") and not line.startswith("报告状态"):
            metadata_lines.append(line.strip())
    content_start = next((idx for idx, line in enumerate(lines) if line.startswith("## ")), body_start)

    doc = Document()
    set_document_defaults(doc)
    add_cover(doc, title, metadata_lines)

    toc_heading = doc.add_paragraph(style="Heading 1")
    add_inline_runs(toc_heading, "目录")
    toc_para = doc.add_paragraph()
    add_toc(toc_para)
    doc.add_page_break()

    i = content_start
    in_quote = False
    while i < len(lines):
        raw = lines[i]
        line = raw.rstrip()
        stripped = line.strip()

        if not stripped:
            i += 1
            continue
        if stripped.startswith(">"):
            clean = stripped.lstrip(">").strip()
            if clean:
                para = doc.add_paragraph()
                para.paragraph_format.left_indent = Cm(0.6)
                add_inline_runs(para, clean)
            in_quote = True
            i += 1
            continue
        if in_quote:
            in_quote = False

        table = split_markdown_table(lines, i)
        if table is not None:
            rows, next_i = table
            word_table = doc.add_table(rows=0, cols=len(rows[0]))
            word_table.style = "Table Grid"
            for row_idx, cells in enumerate(rows):
                row = word_table.add_row().cells
                for col_idx, cell_text in enumerate(cells):
                    row[col_idx].text = INLINE_CODE_RE.sub(r"\1", LINK_RE.sub(lambda m: f"{m.group(1)}（{m.group(2)}）", cell_text))
                    for paragraph in row[col_idx].paragraphs:
                        for run in paragraph.runs:
                            set_run_font(run, bold=(row_idx == 0))
            i = next_i
            continue

        heading_match = HEADING_RE.match(stripped)
        if heading_match:
            level = min(len(heading_match.group(1)), 3)
            para = doc.add_paragraph(style=f"Heading {level}")
            add_inline_runs(para, heading_match.group(2).strip())
            i += 1
            continue

        if stripped.startswith("- "):
            content = stripped[2:].strip()
            para = doc.add_paragraph(style="List Bullet")
            add_inline_runs(para, content)
            maybe_add_image(doc, content)
            i += 1
            continue

        if re.match(r"^\d+\.\s+", stripped):
            content = re.sub(r"^\d+\.\s+", "", stripped)
            para = doc.add_paragraph(style="List Number")
            add_inline_runs(para, content)
            i += 1
            continue

        para = doc.add_paragraph()
        add_inline_runs(para, stripped)
        maybe_add_image(doc, stripped)
        i += 1

    add_member_table_reminder(doc)

    footer = doc.sections[-1].footer.paragraphs[0]
    add_page_number(footer)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output_path)


if __name__ == "__main__":
    markdown_to_docx(DEFAULT_MARKDOWN, DEFAULT_OUTPUT)
    print(DEFAULT_OUTPUT)
