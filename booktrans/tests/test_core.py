from __future__ import annotations

import pymupdf
import pytest

from booktrans.engine import ModelNotFoundError, find_model_dir
from booktrans.optimizer import OptimizingTranslator, TranslationMemory, should_skip
from booktrans.pipeline import JobOptions, parse_pages, run_job
from booktrans.quality import check_translation
from booktrans.segment import normalize, split_sentences

from conftest import FakeEngine


def test_split_sentences_respects_abbreviations():
    text = "Dr. Smith met Mr. Jones at 5 p.m. on Monday. They talked. See Fig. 3 for details!"
    assert split_sentences(text) == [
        "Dr. Smith met Mr. Jones at 5 p.m. on Monday.",
        "They talked.",
        "See Fig. 3 for details!",
    ]


def test_split_sentences_initials_and_quotes():
    assert split_sentences('J. R. R. Tolkien wrote it. "Really?" She asked.') == [
        "J. R. R. Tolkien wrote it.",
        '"Really?"',
        "She asked.",
    ]


def test_long_sentence_is_chunked():
    long = ", ".join(["this clause is fairly long and has many words"] * 20) + "."
    parts = split_sentences(long)
    assert len(parts) > 1 and all(len(p) <= 400 for p in parts)
    assert " ".join(parts) == normalize(long)


def test_normalize_dehyphenates():
    assert normalize("trans-\nlation  of\nbooks") == "translation of books"


@pytest.mark.parametrize("text", ["12", "iv", "https://example.com/x", "x = {a; b; c}", "∑ 3.14 + 2", ""])
def test_should_skip(text):
    assert should_skip(text)


@pytest.mark.parametrize("text", ["Hello there.", "Chapter 1", "I am."])
def test_should_not_skip(text):
    assert not should_skip(text)


def test_quality_checks():
    src = "The quick brown fox jumps over the lazy dog."
    assert check_translation(src, "빠른 갈색 여우가 게으른 개를 뛰어넘는다.") is None
    assert check_translation(src, "").code == "empty"
    assert check_translation(src, src).code == "untranslated"
    assert check_translation(src, "여우 여우 여우 여우 여우 여우").code == "repetition"
    assert check_translation("Hi.", "") is None  # too short to judge


def test_optimizer_dedupes_skips_and_caches(fake_engine):
    memory = TranslationMemory(None)
    tr = OptimizingTranslator(fake_engine, memory)
    segs = ["Hello world again.", "42", "Hello world again.", "Another sentence here."]
    out = tr.translate(segs)
    assert out[1].status == "skipped" and out[1].text == "42"
    assert out[0].text == out[2].text and "가" in out[3].text and out[3].status == "ok"
    assert tr.stats.skipped == 1 and tr.stats.deduped == 1 and tr.stats.translated == 2
    sent_first = sum(len(c[0]) for c in fake_engine.calls)

    tr2 = OptimizingTranslator(fake_engine, memory)
    assert [r.text for r in tr2.translate(segs)] == [r.text for r in out]
    assert tr2.stats.cache_hits == 2 and tr2.stats.source_tokens_sent == 0
    assert sum(len(c[0]) for c in fake_engine.calls) == sent_first  # nothing new sent


def test_cache_policy_isolates_old_entries(fake_engine, monkeypatch):
    from booktrans import optimizer

    memory = TranslationMemory(None)
    memory.put_many(f"{fake_engine.engine_id}|old-policy", {"Hello world again.": "나쁜 옛 번역"})
    out = OptimizingTranslator(fake_engine, memory).translate(["Hello world again."])
    assert out[0].text != "나쁜 옛 번역"  # old-policy cache is not reused
    assert memory.get_many(f"{fake_engine.engine_id}|old-policy", ["Hello world again."])  # ...nor deleted


def test_optimizer_retries_with_careful_decoding():
    bad = "This sentence makes the model loop badly."
    engine = FakeEngine(bad_first={bad})
    tr = OptimizingTranslator(engine, TranslationMemory(None))
    out = tr.translate([bad])
    assert out[0].status == "ok" and "그리고 그리고" not in out[0].text
    assert tr.stats.retried == 1 and tr.stats.unresolved == 0
    assert any(careful for _, careful in engine.calls)


def test_unreliable_output_is_not_shown_or_cached():
    whole = "First clause is here, second clause is there."
    engine = FakeEngine(always_bad={whole})
    tr = OptimizingTranslator(engine, TranslationMemory(None))
    assert tr.translate([whole])[0].status == "ok"  # rescued clause by clause

    hopeless = "No commas so nothing to split here at all."
    engine = FakeEngine(always_bad={hopeless})
    memory = TranslationMemory(None)
    tr = OptimizingTranslator(engine, memory)
    r = tr.translate([hopeless])[0]
    assert r.status == "unreliable" and r.text == hopeless and r.candidate and r.issues
    assert tr.stats.unresolved == 1
    assert memory.get_many(tr.cache_id, [hopeless]) == {}  # failures are not cached


def test_model_input_sentence_cases_all_caps():
    from booktrans.optimizer import model_input

    assert model_input("YOU MUST BE BORN AGAIN") == "You must be born again"
    assert model_input("CONTENTS") == "Contents"
    assert model_input("WHAT I SAW. THEN I LEFT.") == "What I saw. Then I left."
    assert model_input("NASA was founded in 1958.") == "NASA was founded in 1958."


def test_parse_pages():
    assert parse_pages("", 3) == [0, 1, 2]
    assert parse_pages("1-2, 3", 5) == [0, 1, 2]
    assert parse_pages("4-", 5) == [3, 4]
    with pytest.raises(ValueError):
        parse_pages("0-2", 5)
    with pytest.raises(ValueError):
        parse_pages("6", 5)


def test_pipeline_end_to_end(sample_pdf, tmp_path, fake_engine):
    out_dir = tmp_path / "out"
    opts = JobOptions(sample_pdf, out_dir, modes=("overlay", "side-by-side", "text"), pages="1-2")
    result = run_job(opts, fake_engine, TranslationMemory(None))
    names = sorted(p.name for p in result.outputs)
    assert names == ["book.en-ko.md", "book.en-ko.pdf", "book.ko.pdf", "book.report.json"]

    with pymupdf.open(out_dir / "book.ko.pdf") as ko:
        assert ko.page_count == 2
        text = ko[0].get_text()
        assert "가" in text and "quick brown fox" not in text
    with pymupdf.open(out_dir / "book.en-ko.pdf") as dual:
        assert dual.page_count == 2
        assert dual[0].rect.width == pytest.approx(2 * pymupdf.paper_rect("a4").width, rel=0.01)
        text = dual[0].get_text()
        assert "quick brown fox" in text and "가" in text
    assert result.stats.deduped >= 1  # running head repeats on every page
    assert result.report.status == "complete"
    assert "> " in (out_dir / "book.en-ko.md").read_text(encoding="utf-8")


def test_find_model_dir(tmp_path):
    pkg = tmp_path / "translate-en_ko-1_1"
    (pkg / "model").mkdir(parents=True)
    (pkg / "model" / "model.bin").write_bytes(b"x")
    (pkg / "sentencepiece.model").write_bytes(b"x")
    assert find_model_dir(tmp_path) == pkg
    with pytest.raises(ModelNotFoundError):
        find_model_dir(tmp_path / "model")


def test_cache_hits_are_re_reviewed_under_current_rules(fake_engine):
    """A translation accepted by older rules but rejected by the current ones must not be reused."""
    memory = TranslationMemory(None)
    tr = OptimizingTranslator(fake_engine, memory)
    src = "She never forgot the promise she had made."
    memory.put_many(tr.cache_id, {src: "그녀는 결코 약속을 잊었다."})  # reversed meaning, in the current namespace
    out = tr.translate([src])[0]
    assert out.text != "그녀는 결코 약속을 잊었다."
    assert tr.stats.cache_rejected == 1 and tr.stats.cache_hits == 0


def test_old_policy_entries_are_ignored_and_kept(fake_engine):
    memory = TranslationMemory(None)
    src = "In 1848 the city had nearly forty thousand inhabitants."
    old_id = f"{fake_engine.engine_id}|bt0.1.2:casefold1:decode1:review2"
    memory.put_many(old_id, {src: "1848년에 도시에는 거의 4천 명의 주민이 있었다."})
    tr = OptimizingTranslator(fake_engine, memory)
    assert tr.cache_id != old_id
    assert tr.translate([src])[0].text != "1848년에 도시에는 거의 4천 명의 주민이 있었다."
    assert memory.get_many(old_id, [src])  # the user's old entry is not deleted


def test_policy_fingerprint_tracks_rule_changes(monkeypatch):
    import re

    from booktrans import optimizer, quality

    before = optimizer._policy_fingerprint()
    assert before == optimizer._policy_fingerprint()
    monkeypatch.setattr(quality, "_TGT_NEG", re.compile("않|못"))
    assert optimizer._policy_fingerprint() != before
