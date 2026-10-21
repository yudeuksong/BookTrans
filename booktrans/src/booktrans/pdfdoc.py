"""PDF text extraction, paragraph reconstruction and translated-output rendering (PyMuPDF).

PDF text comes as visual lines. A line break in a PDF is layout, not a sentence boundary,
so lines are regrouped into paragraphs before translation:

* lines join only within the same column (horizontal overlap), at a normal line gap,
  with matching font size and weight;
* headings (short, all caps, larger or bold), table-of-contents entries and running
  heads/page numbers stay on their own;
* an indented line or a short line ending a sentence starts/ends a paragraph;
* "exam-\\nple" is de-hyphenated when the next line starts in lower case;
* a body paragraph that runs off the bottom of a page without terminal punctuation is
  joined with the first body paragraph of the next page into one translation unit, and
  the translation is split back over both places.
"""

from __future__ import annotations

import collections
import hashlib
import html
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from booktrans.segment import normalize

_TERMINAL = re.compile(r"[.!?:;…][\"'”’)\]]*$")
_TOC = re.compile(r"(\.{3,}|…{2,}|\s{4,}|\t)\s*\d{1,4}$|^\s*(chapter|part|section|appendix)\b.*\d{1,4}$", re.I)
MIN_READABLE_PT = 6.0
CONTINUATION_NOTE = "(앞 페이지 문단에서 이어짐 — 번역은 앞 페이지에 있습니다)"
FIRST_SCALE_LOW = 0.7


@dataclass
class Line:
    rect: pymupdf.Rect
    text: str
    size: float
    bold: bool
    color: int


@dataclass
class Paragraph:
    page: int
    lines: list[Line]
    kind: str  # body | heading | toc | furniture
    text: str = ""
    unit: "Unit | None" = None
    part_index: int = 0
    translation: str = ""  # text rendered in this paragraph's place
    suffix: str = ""  # kept verbatim after the translation (TOC leader dots and page number)
    fit: str = ""  # "", "expanded", "shrunk", "overflow"
    rendered_pt: float = 0.0

    @property
    def rect(self) -> pymupdf.Rect:
        r = pymupdf.Rect(self.lines[0].rect)
        for ln in self.lines[1:]:
            r |= ln.rect
        return r

    @property
    def size(self) -> float:
        return statistics.median(ln.size for ln in self.lines)

    @property
    def bold(self) -> bool:
        return self.lines[0].bold

    @property
    def color(self) -> int:
        return self.lines[0].color


@dataclass
class Unit:
    """One translation unit: a paragraph, or a paragraph continued across a page break."""

    parts: list[Paragraph]
    text: str
    sentences: list[str] = field(default_factory=list)
    results: list = field(default_factory=list)  # optimizer.SegmentResult per sentence

    @property
    def status(self) -> str:
        if not self.results:
            return "pending"
        states = {r.status for r in self.results}
        if "unreliable" in states:
            return "partial" if "ok" in states else "unreliable"
        return "ok" if "ok" in states else "skipped"


@dataclass
class PageInfo:
    number: int
    rect: pymupdf.Rect
    paragraphs: list[Paragraph]
    needs_ocr: bool


def _line_from_spans(spans: list[dict], bbox) -> Line | None:
    text = "".join(s["text"] for s in spans)
    if not text.strip():
        return None
    main = max(spans, key=lambda s: len(s["text"].strip()))
    bold = bool(main["flags"] & 16) or "bold" in main.get("font", "").lower()
    return Line(pymupdf.Rect(bbox), re.sub(r"\s+", " ", text).strip(), main["size"], bold, main["color"])


def _is_heading(line: Line, body_size: float) -> bool:
    letters = [c for c in line.text if c.isalpha()]
    if not letters or len(line.text) > 90 or _TERMINAL.search(line.text.rstrip(":")):
        return False
    upper = sum(c.isupper() for c in letters) / len(letters)
    return (upper > 0.85 and len(letters) >= 2) or line.size >= body_size * 1.18 or (line.bold and len(line.text) <= 70)


def _kind(line: Line, page_rect: pymupdf.Rect, body_size: float) -> str:
    band = page_rect.height * 0.065
    if (line.rect.y1 <= page_rect.y0 + band or line.rect.y0 >= page_rect.y1 - band) and len(line.text) <= 70:
        return "furniture"
    if _TOC.search(line.text):
        return "toc"
    if _is_heading(line, body_size):
        return "heading"
    return "body"


def join_lines(texts: list[str]) -> str:
    out = ""
    for t in texts:
        t = t.strip()
        if not out:
            out = t
        elif re.search(r"[A-Za-z]-$", out) and not out.endswith("--") and t[:1].islower():
            out = out[:-1] + t  # de-hyphenate "trans-" + "lation"
        else:
            out = f"{out} {t}"
    return normalize(out)


@dataclass
class _PageMetrics:
    line_gap: float  # typical gap between consecutive lines of one paragraph
    right_edge: float  # where full body lines end


def _page_metrics(lines: list[Line]) -> _PageMetrics:
    gaps = []
    for a, b in zip(lines, lines[1:]):
        h = max(a.rect.height, 1.0)
        overlap = min(a.rect.x1, b.rect.x1) - max(a.rect.x0, b.rect.x0)
        d = b.rect.y0 - a.rect.y1
        if overlap > 0 and abs(a.size - b.size) < 0.5 and -0.5 * h < d < 1.5 * h:
            gaps.append(d)
    long_ends = sorted(ln.rect.x1 for ln in lines if len(ln.text) > 40)
    right = long_ends[int(len(long_ends) * 0.9)] if long_ends else max((ln.rect.x1 for ln in lines), default=0)
    return _PageMetrics(statistics.median(gaps) if gaps else 0.0, right)


def _continues(par: Paragraph, line: Line, m: _PageMetrics) -> bool:
    last = par.lines[-1]
    if abs(last.size - line.size) > 0.15 * max(last.size, line.size) or last.bold != line.bold:
        return False
    h = max(last.rect.height, 1.0)
    gap = line.rect.y0 - last.rect.y1
    # Paragraph spacing shows up as a gap clearly larger than this page's usual line gap.
    if not -0.5 * h <= gap <= m.line_gap + 0.35 * h:
        return False
    pr = par.rect
    overlap = min(pr.x1, line.rect.x1) - max(pr.x0, line.rect.x0)
    if overlap < 0.3 * min(pr.width, line.rect.width) or line.rect.x0 < pr.x0 - 2 * line.size:
        return False
    ends = bool(_TERMINAL.search(last.text))
    if ends and line.rect.x0 > pr.x0 + 1.2 * line.size:
        return False  # indented first line of a new paragraph
    if ends and last.rect.x1 < min(pr.x1, m.right_edge) - 4 * line.size:
        return False  # short last line closes the paragraph
    if ends and last.rect.x1 < m.right_edge - 4 * line.size and len(par.lines) == 1:
        return False  # a short single line ending a sentence (list item, caption)
    return True


def extract(doc: pymupdf.Document, pages: list[int]) -> tuple[list[PageInfo], list[Unit]]:
    raw: list[tuple[int, pymupdf.Rect, list[Line], bool]] = []
    sizes: list[float] = []
    for pno in pages:
        page = doc[pno]
        data = page.get_text("dict", flags=pymupdf.TEXT_PRESERVE_LIGATURES)
        lines: list[Line] = []
        for b in data["blocks"]:
            if b.get("type") != 0:
                continue
            for ln in b["lines"]:
                if abs(ln["dir"][1]) > 0.1:  # vertical text stays as it is
                    continue
                line = _line_from_spans(ln["spans"], ln["bbox"])
                if line:
                    lines.append(line)
                    sizes += [line.size] * max(1, len(line.text) // 10)
        has_images = bool(page.get_images(full=False))
        raw.append((pno, page.rect, lines, has_images))
    body_size = statistics.median(sizes) if sizes else 10.0

    infos: list[PageInfo] = []
    for pno, prect, lines, has_images in raw:
        paras: list[Paragraph] = []
        metrics = _page_metrics(lines)
        for line in lines:
            kind = _kind(line, prect, body_size)
            cur = paras[-1] if paras else None
            if cur and cur.kind == "body" and kind == "body" and _continues(cur, line, metrics):
                cur.lines.append(line)
            else:
                paras.append(Paragraph(pno, [line], kind))
        for p in paras:
            p.text = join_lines([ln.text for ln in p.lines])
            if p.kind == "toc":
                m = re.search(r"\s*(?:[.…·]{2,}\s*|\s{2,}|\t)?\d{1,4}$", p.text)
                if m and m.start() > 0:
                    # A short leader keeps the page number readable next to a longer translation.
                    number = re.search(r"\d{1,4}$", p.text).group()
                    p.text, p.suffix = p.text[: m.start()].strip(), f" … {number}"
        letters = sum(c.isalpha() for p in paras for c in p.text)
        infos.append(PageInfo(pno, prect, paras, needs_ocr=letters < 20 and has_images))

    units: list[Unit] = []
    prev_tail: Paragraph | None = None
    prev_page = None
    for info in infos:
        bodies = [p for p in info.paragraphs if p.kind != "furniture"]
        for i, p in enumerate(info.paragraphs):
            joinable = (
                i == info.paragraphs.index(bodies[0]) if bodies else False
            ) and prev_tail is not None and prev_page is not None and info.number == prev_page + 1
            if (
                joinable
                and p.kind == "body"
                and prev_tail.kind == "body"
                and not _TERMINAL.search(prev_tail.text)
                and p.text[:1].islower()
                and abs(p.size - prev_tail.size) <= 0.15 * max(p.size, prev_tail.size)
            ):
                unit = prev_tail.unit
                unit.parts.append(p)
                unit.text = join_lines([unit.text, p.text])
                p.unit, p.part_index = unit, len(unit.parts) - 1
            else:
                unit = Unit([p], p.text)
                p.unit = unit
                units.append(unit)
        prev_tail = bodies[-1] if bodies else None
        prev_page = info.number
    return infos, units


def distribute(translation: str, parts: list[Paragraph]) -> list[str]:
    """Split one translation over the places its source came from, proportionally, at spaces."""
    if len(parts) == 1:
        return [translation]
    total = sum(len(p.text) for p in parts) or 1
    pieces, rest = [], translation
    remaining_weight = total
    for p in parts[:-1]:
        share = len(p.text) / remaining_weight
        cut = round(len(rest) * share)
        space = rest.rfind(" ", 0, cut + 1)
        nxt = rest.find(" ", cut)
        if space <= 0 or (nxt != -1 and nxt - cut < cut - space):
            space = nxt if nxt != -1 else len(rest)
        pieces.append(rest[:space].strip())
        rest = rest[space:].strip()
        remaining_weight -= len(p.text)
    pieces.append(rest)
    return pieces


_CSS = "* {{ font-family: sans-serif; font-size: {size:.2f}px; line-height: 1.2; color: #{color:06x}; margin: 0; {weight} }}"


def _room_below(par: Paragraph, info: PageInfo) -> float:
    """Lowest y the paragraph may grow to without running into the next thing in its column."""
    r = par.rect
    limit = info.rect.y1 - info.rect.height * 0.04
    for other in info.paragraphs:
        o = other.rect
        if other is par or o.y0 < r.y1 - 1:
            continue
        if min(o.x1, r.x1) - max(o.x0, r.x0) > 0:
            limit = min(limit, o.y0 - 1)
    return limit


def _room_right(par: Paragraph, info: PageInfo) -> float:
    """Rightmost x a single-line item may grow to without touching its right-hand neighbour."""
    r = par.rect
    limit = info.rect.x1 - info.rect.width * 0.08
    for other in info.paragraphs:
        o = other.rect
        if other is par or o.x0 < r.x1 - 1:
            continue
        if min(o.y1, r.y1) - max(o.y0, r.y0) > 0:
            limit = min(limit, o.x0 - 6)
    return max(limit, r.x1)


def _initial_rect(p: Paragraph, info: PageInfo) -> pymupdf.Rect:
    """The original text box, padded for Korean line height; single lines may also grow right."""
    r = p.rect
    rect = pymupdf.Rect(r.x0, r.y0 - 0.15 * p.size, r.x1, r.y1 + 0.3 * p.size)
    if len(p.lines) == 1:
        rect.x1 = _room_right(p, info)
    return rect


def render_overlay(src_path: Path, pages: list[PageInfo]) -> pymupdf.Document:
    """Layout-preserving output. Paragraphs with unreliable sentences keep those sentences in
    English and get a highlight explaining why; nothing unchecked is shown as a translation."""
    doc = pymupdf.open(src_path)
    for info in pages:
        page = doc[info.number]
        todo = [p for p in info.paragraphs if p.translation and p.translation != p.text]
        for p in todo:
            for ln in p.lines:
                page.add_redact_annot(ln.rect)
        if todo:
            page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE, graphics=pymupdf.PDF_REDACT_LINE_ART_NONE)
        for p in todo:
            css = _CSS.format(size=p.size, color=p.color & 0xFFFFFF, weight="font-weight: bold;" if p.bold else "")
            body = f"<p>{html.escape(p.translation + p.suffix)}</p>"
            rect = _initial_rect(p, info)
            spare, scale = page.insert_htmlbox(rect, body, css=css, scale_low=FIRST_SCALE_LOW)
            p.fit = ""
            if spare < 0:
                rect = pymupdf.Rect(rect.x0, rect.y0, rect.x1, max(rect.y1, _room_below(p, info)))
                spare, scale = page.insert_htmlbox(rect, body, css=css, scale_low=FIRST_SCALE_LOW)
                p.fit = "expanded"
            if spare < 0:
                spare, scale = page.insert_htmlbox(rect, body, css=css, scale_low=0)
                p.fit = "shrunk" if spare >= 0 else "overflow"
            p.rendered_pt = p.size * scale
        for p in info.paragraphs:
            unit = p.unit
            if unit is None or unit.status not in ("partial", "unreliable"):
                continue
            bad = [r for r in unit.results if r.status == "unreliable"]
            note = "자동 검사를 통과하지 못한 문장은 원문(영어)으로 남겼습니다.\n" + "\n".join(
                f"- {r.source}\n  사유: {', '.join(r.issues)}\n  기계 번역 후보: {r.candidate}" for r in bad
            )
            annot = page.add_highlight_annot([ln.rect for ln in p.lines] if not p.translation else [p.rect])
            annot.set_info(title="BookTrans 검토 필요", content=note)
            annot.set_colors(stroke=(1, 0.85, 0.2))
            annot.update()
    return doc


def render_side_by_side(src_path: Path, translated: pymupdf.Document, page_numbers: list[int]) -> pymupdf.Document:
    """Bilingual output: original page on the left, translated page on the right."""
    src = pymupdf.open(src_path)
    # show_pdf_page drops annotations; bake the review highlights into the page content.
    translated = pymupdf.open("pdf", translated.tobytes())
    translated.bake(annots=True, widgets=False)
    out = pymupdf.open()
    for i, pno in enumerate(page_numbers):
        w, h = src[pno].rect.width, src[pno].rect.height
        page = out.new_page(width=2 * w, height=h)
        page.show_pdf_page(pymupdf.Rect(0, 0, w, h), src, pno)
        page.show_pdf_page(pymupdf.Rect(w, 0, 2 * w, h), translated, i)
        page.draw_line((w, 0), (w, h), color=(0.75, 0.75, 0.75), width=0.5)
    src.close()
    return out


def finalize(doc: pymupdf.Document, path: Path, *, subset: bool) -> None:
    """Write the document once: merge identical objects/streams first (every inserted text box
    carries its own copy of the fonts), then subset each remaining font a single time."""
    merged = pymupdf.open("pdf", doc.tobytes(garbage=4, deflate=True))
    if subset:
        merged.subset_fonts()
    merged.save(path, garbage=4, deflate=True, deflate_fonts=True, deflate_images=True, use_objstms=1)
    merged.close()


def font_streams(path: Path) -> dict:
    """Embedded font program statistics for verification."""
    doc = pymupdf.open(path)
    digests: collections.Counter = collections.Counter()
    total = 0
    for x in range(1, doc.xref_length()):
        for key in ("FontFile", "FontFile2", "FontFile3"):
            kind, val = doc.xref_get_key(x, key)
            if kind == "xref":
                ref = int(val.split()[0])
                data = doc.xref_stream_raw(ref)
                digests[hashlib.sha256(data).hexdigest()] += 1
                total += len(data)
    doc.close()
    return {
        "font_programs": sum(digests.values()),
        "distinct": len(digests),
        "duplicates": sum(c - 1 for c in digests.values()),
        "font_bytes": total,
    }


def write_bilingual_text(pages: list[PageInfo], path: Path) -> None:
    """Paragraph-by-paragraph bilingual Markdown for reading or proofing."""
    with path.open("w", encoding="utf-8") as fh:
        fh.write("> BookTrans 기계 번역입니다. 자동 검사는 흔한 오류만 찾을 뿐 뜻이 맞는지는 보장하지 않습니다.\n")
        for info in pages:
            fh.write(f"\n## p.{info.number + 1}\n\n")
            if info.needs_ocr:
                fh.write("> ⚠ 이 페이지는 이미지로만 되어 있어 번역하지 않았습니다(OCR 미지원).\n\n")
            for p in info.paragraphs:
                fh.write(f"{p.text}{p.suffix}\n\n")
                unit = p.unit
                if unit is None or p.part_index > 0:
                    if unit is not None:
                        fh.write(f"> {CONTINUATION_NOTE}\n\n")
                    continue
                if unit.status == "skipped":
                    continue
                for r in unit.results:
                    if r.status == "unreliable":
                        fh.write(f"> ⚠ 검토 필요({', '.join(r.issues)}): {r.candidate}\n")
                    elif r.status == "ok":
                        fh.write(f"> {r.text}\n")
                fh.write("\n")
