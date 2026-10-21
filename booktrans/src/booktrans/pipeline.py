"""End-to-end PDF translation job."""

from __future__ import annotations

import collections
import hashlib
import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import pymupdf

from booktrans import __version__, pdfdoc
from booktrans.engine import TranslationEngine
from booktrans.optimizer import CACHE_POLICY, OptimizerStats, OptimizingTranslator, TranslationMemory
from booktrans.segment import split_sentences

OUTPUT_MODES = ("overlay", "side-by-side", "text")

Progress = Callable[[str, int, int], None]


@dataclass
class JobOptions:
    input_pdf: Path
    output_dir: Path
    modes: tuple[str, ...] = ("overlay", "side-by-side")
    pages: str = ""  # "1-10,15" (1-based); empty = all
    batch_size: int = 16


@dataclass
class JobReport:
    """What happened, in numbers. `status` is "complete" only when nothing needs attention."""

    status: str = "complete"
    booktrans: str = __version__
    cache_policy: str = CACHE_POLICY
    pages: int = 0
    pages_without_text: list[int] = field(default_factory=list)  # 1-based
    scanned_pages: list[int] = field(default_factory=list)  # 1-based, need OCR
    paragraphs: int = 0
    units: int = 0
    units_ok: int = 0
    units_partial: int = 0
    units_unreliable: int = 0
    sentences: int = 0  # sentence occurrences in the document (a repeated sentence counts each time)
    sentences_unreliable: int = 0  # occurrences left in English after review
    unique_sentences_unreliable: int = 0  # distinct sentences left in English
    issue_counts: dict = field(default_factory=dict)  # issue code -> distinct sentences with that code
    unreliable_sentences: list = field(default_factory=list)  # [{id, sentence, occurrences, pages, issues}]
    fit_expanded: int = 0
    fit_shrunk: int = 0
    fit_overflow: int = 0
    smallest_rendered_pt: float = 0.0
    small_text_paragraphs: int = 0
    outputs: dict = field(default_factory=dict)  # name -> {"bytes": int, **font stats}


@dataclass
class JobResult:
    outputs: list[Path]
    stats: OptimizerStats
    report: JobReport
    warnings: list[str] = field(default_factory=list)

    @property
    def ocr_pages(self) -> list[int]:
        return self.report.scanned_pages


def parse_pages(spec: str, page_count: int) -> list[int]:
    if not spec.strip():
        return list(range(page_count))
    selected: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            start, end = int(a or 1), int(b or page_count)
        else:
            start = end = int(part)
        if start < 1 or end > page_count or start > end:
            raise ValueError(f"페이지 범위가 잘못되었습니다: {part} (문서는 1-{page_count}쪽)")
        selected.update(range(start - 1, end))
    return sorted(selected)


def run_job(
    opts: JobOptions,
    engine: TranslationEngine,
    memory: TranslationMemory,
    *,
    progress: Progress | None = None,
    cancel: threading.Event | None = None,
) -> JobResult:
    def report_progress(stage: str, done: int, total: int) -> None:
        if progress:
            progress(stage, done, total)

    with pymupdf.open(opts.input_pdf) as doc:
        if doc.needs_pass:
            raise ValueError("암호로 보호된 PDF는 지원하지 않습니다.")
        page_numbers = parse_pages(opts.pages, doc.page_count)
        report_progress("extract", 0, len(page_numbers))
        pages, units = pdfdoc.extract(doc, page_numbers)
        report_progress("extract", len(page_numbers), len(page_numbers))

    for u in units:
        u.sentences = split_sentences(u.text)
    sentences = [s for u in units for s in u.sentences]
    translator = OptimizingTranslator(engine, memory, batch_size=opts.batch_size, cancel=cancel)
    results = translator.translate(sentences, progress=lambda d, t: report_progress("translate", d, t))

    it = iter(results)
    for u in units:
        u.results = [next(it) for _ in u.sentences]
        if u.status in ("ok", "partial"):
            shown = " ".join(r.text for r in u.results)
            for part, piece in zip(u.parts, pdfdoc.distribute(shown, u.parts)):
                part.translation = piece

    opts.output_dir.mkdir(parents=True, exist_ok=True)
    stem = opts.input_pdf.stem
    outputs: list[Path] = []
    rep = JobReport(pages=len(page_numbers))
    report_progress("render", 0, len(opts.modes))
    if "overlay" in opts.modes or "side-by-side" in opts.modes:
        overlay = pdfdoc.render_overlay(opts.input_pdf, pages)
        # Keep only the selected pages, so a partial job doesn't ship untranslated pages.
        overlay.select(page_numbers)
        ko_path = opts.output_dir / f"{stem}.ko.pdf"
        pdfdoc.finalize(overlay, ko_path, subset=True)
        overlay.close()
        if "overlay" in opts.modes:
            outputs.append(ko_path)
        if "side-by-side" in opts.modes:
            with pymupdf.open(ko_path) as finished:
                dual = pdfdoc.render_side_by_side(opts.input_pdf, finished, page_numbers)
                path = opts.output_dir / f"{stem}.en-ko.pdf"
                # Fonts were subset once already; only merge duplicates here.
                pdfdoc.finalize(dual, path, subset=False)
                dual.close()
            outputs.append(path)
        if "overlay" not in opts.modes:
            ko_path.unlink()
    if "text" in opts.modes:
        path = opts.output_dir / f"{stem}.en-ko.md"
        pdfdoc.write_bilingual_text(pages, path)
        outputs.append(path)
    report_progress("render", len(opts.modes), len(opts.modes))

    paragraphs = [p for info in pages for p in info.paragraphs]
    rep.paragraphs = len(paragraphs)
    rep.units = len(units)
    rep.units_ok = sum(u.status == "ok" for u in units)
    rep.units_partial = sum(u.status == "partial" for u in units)
    rep.units_unreliable = sum(u.status == "unreliable" for u in units)
    rep.sentences = len(sentences)
    rep.sentences_unreliable = sum(r.status == "unreliable" for r in results)
    by_sentence: dict[str, dict] = {}
    for u in units:
        for r in u.results:
            if r.status != "unreliable":
                continue
            entry = by_sentence.setdefault(r.source, {
                "id": hashlib.sha1(r.source.encode("utf-8")).hexdigest()[:10],
                "sentence": r.source, "occurrences": 0, "pages": [], "issues": list(r.issues),
            })
            entry["occurrences"] += 1
            for page in sorted({part.page + 1 for part in u.parts}):
                if page not in entry["pages"]:
                    entry["pages"].append(page)
    rep.unreliable_sentences = sorted(by_sentence.values(), key=lambda e: (e["pages"], e["id"]))
    rep.unique_sentences_unreliable = len(by_sentence)
    rep.issue_counts = dict(sorted(collections.Counter(c for e in by_sentence.values() for c in e["issues"]).items()))
    rep.scanned_pages = [i.number + 1 for i in pages if i.needs_ocr]
    rep.pages_without_text = [i.number + 1 for i in pages if not i.paragraphs and not i.needs_ocr]
    rendered = [p for p in paragraphs if p.translation and p.translation != p.text]
    rep.fit_expanded = sum(p.fit == "expanded" for p in rendered)
    rep.fit_shrunk = sum(p.fit == "shrunk" for p in rendered)
    rep.fit_overflow = sum(p.fit == "overflow" for p in rendered)
    sizes = [p.rendered_pt for p in rendered if p.rendered_pt]
    rep.smallest_rendered_pt = round(min(sizes), 2) if sizes else 0.0
    rep.small_text_paragraphs = sum(s < pdfdoc.MIN_READABLE_PT for s in sizes)
    for path in outputs:
        info = {"bytes": path.stat().st_size}
        if path.suffix == ".pdf":
            info.update(pdfdoc.font_streams(path))
        rep.outputs[path.name] = info

    warnings: list[str] = []
    if rep.scanned_pages:
        warnings.append(f"텍스트가 없는 스캔 페이지(OCR 필요, 번역 안 함): {rep.scanned_pages}")
    if rep.pages_without_text:
        warnings.append(f"번역할 글자가 없는 페이지: {rep.pages_without_text}")
    if rep.sentences_unreliable:
        warnings.append(
            f"자동 검사를 통과하지 못해 원문(영어)으로 남긴 문장: 서로 다른 문장 {rep.unique_sentences_unreliable}개, "
            f"문서 안 등장 {rep.sentences_unreliable}/{rep.sentences}회 "
            f"(PDF에 노란 표시와 메모, .md에 ⚠ 표시)"
        )
    if rep.fit_overflow or rep.small_text_paragraphs:
        warnings.append(
            f"자리가 부족한 문단: 넘침 {rep.fit_overflow}개, {pdfdoc.MIN_READABLE_PT:g}pt 미만으로 작아진 문단 {rep.small_text_paragraphs}개"
        )
    if warnings:
        rep.status = "complete-with-warnings"
    if not any(u.status in ("ok", "partial") for u in units):
        rep.status = "nothing-translated"
        warnings.append("번역된 문단이 하나도 없습니다.")

    report_path = opts.output_dir / f"{stem}.report.json"
    report_path.write_text(json.dumps(asdict(rep), ensure_ascii=False, indent=2), encoding="utf-8")
    outputs.append(report_path)
    return JobResult(outputs=outputs, stats=translator.stats, report=rep, warnings=warnings)
