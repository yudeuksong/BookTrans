"""Runs only when a real model is available (BOOKTRANS_MODEL); CI installs the m2m100 package."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from booktrans.optimizer import TranslationMemory
from booktrans.pipeline import JobOptions, run_job
from booktrans.quality import check_translation

pytestmark = pytest.mark.skipif(not os.environ.get("BOOKTRANS_MODEL"), reason="BOOKTRANS_MODEL not set")


@pytest.fixture(scope="module")
def engine():
    from booktrans.engine import load_engine
    from booktrans.models import locate_model

    return load_engine(locate_model())


def test_real_translation_is_korean(engine):
    src = ["The book is on the table.", "She opened the door and walked into the garden."]
    out = engine.translate_batch(src)
    for s, t in zip(src, out):
        print(s, "->", t)
        assert check_translation(s, t) is None


def test_real_pipeline(engine, sample_pdf, tmp_path: Path):
    from conftest import assert_output_contract

    out = tmp_path / "out"  # the input PDF lives in tmp_path; the output folder must hold outputs only
    result = run_job(JobOptions(sample_pdf, out), engine, TranslationMemory(None))
    print(result.stats.summary())
    report = assert_output_contract(result, sample_pdf, out, ("overlay", "side-by-side"), min_hangul_per_page=5)
    assert result.stats.translated > 0
    assert report["units_ok"] + report["units_partial"] > 0


def test_real_fixture_all_modes(engine, tmp_path: Path):
    """The synthetic book through the real model: contract, size, fonts, and what was withheld."""
    import importlib.util

    from conftest import assert_output_contract

    spec = importlib.util.spec_from_file_location("make_fixture", Path(__file__).parents[1] / "packaging" / "make_fixture.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    book = tmp_path / "book.pdf"
    mod.build(book)
    out = tmp_path / "out"
    modes = ("overlay", "side-by-side", "text")
    result = run_job(JobOptions(book, out, modes=modes), engine, TranslationMemory(None))
    print(result.stats.summary())
    print(result.warnings)
    report = assert_output_contract(result, book, out, modes, min_hangul_per_page=20)
    assert report["fit_overflow"] == 0 and report["small_text_paragraphs"] == 0
