"""Paragraph reconstruction and PDF output size on the synthetic book fixture."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pymupdf
import pytest

from booktrans import pdfdoc
from booktrans.optimizer import TranslationMemory
from booktrans.pipeline import JobOptions, run_job

from conftest import FakeEngine

_spec = importlib.util.spec_from_file_location("make_fixture", Path(__file__).parents[1] / "packaging" / "make_fixture.py")
make_fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(make_fixture)

BUDGET = 5 * 1024 * 1024


@pytest.fixture(scope="module")
def book(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("fx") / "book.pdf"
    make_fixture.build(path)
    return path


@pytest.fixture(scope="module")
def extracted(book):
    with pymupdf.open(book) as doc:
        return pdfdoc.extract(doc, list(range(doc.page_count)))


def test_fixture_has_book_like_line_count(book):
    with pymupdf.open(book) as doc:
        lines = sum(len(b["lines"]) for p in doc for b in p.get_text("dict")["blocks"] if b["type"] == 0)
    assert 75 <= lines <= 95


def test_headings_and_toc_stay_separate(extracted):
    pages, units = extracted
    first = pages[0].paragraphs
    kinds = {p.text: p.kind for p in first}
    assert kinds["YOU MUST BE BORN AGAIN"] == "heading"
    assert kinds["CONTENTS"] == "heading"
    assert kinds["PROBLEM"] == "heading"
    toc = [p for p in first if p.kind == "toc"]
    assert [p.text for p in toc][:2] == ["Preface", "The Problem"]
    assert all(p.suffix.endswith(("1", "3", "17", "29", "41", "55")) for p in toc)


def test_lines_join_into_paragraphs_and_dehyphenate(extracted):
    _, units = extracted
    intro = next(u for u in units if u.text.startswith("Every reader"))
    assert "transformed by new ideas" in intro.text and "trans-" not in intro.text
    assert "faster than ever before." in intro.text  # continued across the page break
    assert len(intro.parts) == 2 and intro.parts[0].page == 0 and intro.parts[1].page == 1


def test_columns_and_list_items_are_not_merged(extracted):
    _, units = extracted
    left = next(u for u in units if u.text.startswith("In the left column"))
    assert "traveller" not in left.text
    questions = [u.text for u in units if u.text.endswith("?") and len(u.text) < 80]
    assert "What does the author mean by a new beginning?" in questions
    assert "Can a person change without help from others?" in questions


def test_running_heads_are_not_joined(extracted):
    pages, _ = extracted
    for info in pages:
        furniture = [p for p in info.paragraphs if p.kind == "furniture"]
        assert {p.text for p in furniture} >= {str(info.number + 1)}


def test_distribute_splits_at_spaces(extracted):
    _, units = extracted
    intro = next(u for u in units if len(u.parts) == 2)
    pieces = pdfdoc.distribute("가나다 라마바 사아자 차카타 파하", intro.parts)
    assert len(pieces) == 2 and " ".join(pieces) == "가나다 라마바 사아자 차카타 파하"


def test_outputs_within_budget_without_duplicate_fonts(book, tmp_path):
    result = run_job(JobOptions(book, tmp_path, modes=("overlay", "side-by-side", "text")), FakeEngine(), TranslationMemory(None))
    report = json.loads((tmp_path / "book.report.json").read_text(encoding="utf-8"))
    for name in ("book.ko.pdf", "book.en-ko.pdf"):
        info = report["outputs"][name]
        assert info["bytes"] <= BUDGET, f"{name} is {info['bytes']} bytes"
        assert info["duplicates"] == 0
        with pymupdf.open(tmp_path / name) as doc:
            assert doc.page_count == 3
            assert "가" in doc[0].get_text()
    assert result.report.fit_overflow == 0
    assert result.report.smallest_rendered_pt >= pdfdoc.MIN_READABLE_PT


def test_unreliable_sentences_stay_english_and_are_flagged(book, tmp_path):
    bad = "Readers did not accept every argument they found, but they argued about them in homes, in shops and in the streets of the growing towns, where news travelled faster than ever before."
    engine = FakeEngine(always_bad={bad})
    result = run_job(JobOptions(book, tmp_path, modes=("overlay", "text")), engine, TranslationMemory(None))
    assert result.report.sentences_unreliable >= 1 and result.report.status == "complete-with-warnings"
    md = (tmp_path / "book.en-ko.md").read_text(encoding="utf-8")
    assert "⚠ 검토 필요" in md
    with pymupdf.open(tmp_path / "book.ko.pdf") as doc:
        notes = [a.info["content"] for p in doc for a in p.annots()]
    assert any("자동 검사를 통과하지 못한" in n for n in notes)


@pytest.mark.parametrize(
    "modes",
    [("overlay",), ("side-by-side",), ("text",), ("overlay", "side-by-side"), ("overlay", "text"),
     ("side-by-side", "text"), ("overlay", "side-by-side", "text")],
)
def test_output_contract_for_every_mode_combination(book, tmp_path, modes):
    from conftest import assert_output_contract

    result = run_job(JobOptions(book, tmp_path, modes=modes), FakeEngine(), TranslationMemory(None))
    assert_output_contract(result, book, tmp_path, modes)


def test_contract_rejects_extra_or_missing_files(book, tmp_path):
    from conftest import assert_output_contract

    result = run_job(JobOptions(book, tmp_path, modes=("overlay",)), FakeEngine(), TranslationMemory(None))
    (tmp_path / "stray.tmp").write_text("x")
    with pytest.raises(AssertionError):
        assert_output_contract(result, book, tmp_path, ("overlay",))
    (tmp_path / "stray.tmp").unlink()
    with pytest.raises(AssertionError):
        assert_output_contract(result, book, tmp_path, ("overlay", "side-by-side"))


def test_same_folder_as_source_keeps_user_files(tmp_path):
    """Default product behaviour: outputs go next to the source PDF, among the user's files."""
    import hashlib
    import shutil

    from conftest import assert_output_contract, snapshot

    folder = tmp_path / "library"
    folder.mkdir()
    book = folder / "book.pdf"
    make_fixture.build(book)
    (folder / "notes.txt").write_text("my reading notes", encoding="utf-8")
    shutil.copy(book, folder / "other-book.pdf")
    before = snapshot(folder)

    modes = ("overlay", "side-by-side", "text")
    result = run_job(JobOptions(book, folder, modes=modes), FakeEngine(), TranslationMemory(None))
    assert_output_contract(result, book, folder, modes, before=before)
    assert hashlib.sha256(book.read_bytes()).hexdigest() == before["book.pdf"]

    # A second run over the same folder replaces its own outputs and nothing else.
    before = snapshot(folder)
    result = run_job(JobOptions(book, folder, modes=modes), FakeEngine(), TranslationMemory(None))
    assert_output_contract(result, book, folder, modes, before=before)


def test_contract_detects_a_modified_user_file(tmp_path):
    from conftest import assert_output_contract, snapshot

    folder = tmp_path / "f"
    folder.mkdir()
    book = folder / "book.pdf"
    make_fixture.build(book)
    (folder / "notes.txt").write_text("before", encoding="utf-8")
    before = snapshot(folder)
    result = run_job(JobOptions(book, folder, modes=("overlay",)), FakeEngine(), TranslationMemory(None))
    (folder / "notes.txt").write_text("changed", encoding="utf-8")
    with pytest.raises(AssertionError, match="notes.txt"):
        assert_output_contract(result, book, folder, ("overlay",), before=before)


def test_md_translation_starting_with_parenthesis_is_parsed(book, tmp_path):
    """A translation line that begins with '(' is a translation, not the continuation note."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("verify_output", Path(__file__).parents[1] / "packaging" / "verify_output.py")
    vo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vo)
    md = "> header\n\n## p.1\n\nCONTENTS\n\n> (목차)\n\nnext page part\n\n> " + pdfdoc.CONTINUATION_NOTE + "\n\n"
    parsed = vo._md_sentences(md)
    assert parsed["CONTENTS"] == [("ok", "(목차)")]
    assert parsed["next page part"] == []
