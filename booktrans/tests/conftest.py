from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest


def _fake_word(w: str) -> str:
    """Hangul of half the word's length; digits and negation survive, like a decent model."""
    core = w.strip(".,;:!?\"'()")
    if any(c.isdigit() for c in core):
        return core
    if core.lower() in ("not", "never", "no", "nothing", "cannot") or core.lower().endswith("n't"):
        return "않"
    return "가" * max(1, len(core) // 2)


class FakeEngine:
    """Deterministic stand-in for the NMT model: every word becomes Hangul of half its length."""

    engine_id = "fake:en-ko:1"

    def __init__(self, bad_first: set[str] | None = None, always_bad: set[str] | None = None):
        self.calls: list[tuple[list[str], bool]] = []
        self.bad_first = bad_first or set()
        self.always_bad = always_bad or set()

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def translate_batch(self, texts, *, careful=False):
        self.calls.append((list(texts), careful))
        out = []
        for t in texts:
            if t in self.always_bad or (t in self.bad_first and not careful):
                out.append("그리고 그리고 그리고 그리고 그리고")
            else:
                out.append(" ".join(_fake_word(w) for w in t.split()))
        return out


@pytest.fixture
def fake_engine():
    return FakeEngine()


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    doc = pymupdf.open()
    for i in range(3):
        page = doc.new_page()
        page.insert_text((72, 40), "THE RUNNING HEAD OF THE BOOK", fontsize=9)
        page.insert_textbox(
            pymupdf.Rect(72, 80, 520, 300),
            f"Chapter {i + 1}. The quick brown fox jumps over the lazy dog. "
            "Dr. Smith said it was fine. It was a long and dark night in the valley.",
            fontsize=11,
        )
        page.insert_text((300, 800), str(i + 1), fontsize=9)
    path = tmp_path / "book.pdf"
    doc.save(path)
    return path


# ---------------------------------------------------------------- output contract
MODE_FILES = {"overlay": "{stem}.ko.pdf", "side-by-side": "{stem}.en-ko.pdf", "text": "{stem}.en-ko.md"}
REPORT_KEYS = {
    "status", "booktrans", "cache_policy", "pages", "pages_without_text", "scanned_pages", "paragraphs", "units",
    "units_ok", "units_partial", "units_unreliable", "sentences", "sentences_unreliable", "unique_sentences_unreliable",
    "issue_counts", "unreliable_sentences",
    "fit_expanded", "fit_shrunk", "fit_overflow", "smallest_rendered_pt", "small_text_paragraphs", "outputs",
}
SIZE_BUDGET = 5 * 1024 * 1024


def snapshot(folder: Path) -> dict[str, str]:
    """File name -> SHA-256 of everything already in a folder (taken before a run)."""
    import hashlib

    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.is_file()}


def assert_output_contract(result, src: Path, outdir: Path, modes: tuple[str, ...], *, min_hangul_per_page: int = 1,
                           before: dict[str, str] | None = None):
    """Exactly one file per requested mode plus <stem>.report.json, nothing else; every
    PDF opens with the source's page count, real Korean text on each page, no duplicated
    font program and stays under the size budget; the report matches the files."""
    import json

    import booktrans
    from booktrans.optimizer import CACHE_POLICY
    from booktrans.pdfdoc import font_streams

    stem = src.stem
    expected = {MODE_FILES[m].format(stem=stem) for m in modes} | {f"{stem}.report.json"}
    assert {p.name for p in result.outputs} == expected
    assert all(p.parent == outdir for p in result.outputs)
    # The folder may hold the user's own files (the source PDF, notes, earlier runs): the run
    # must add exactly the expected outputs, leave no temporary files, and not touch the rest.
    before = before or {}
    after = snapshot(outdir)
    assert set(after) == expected | set(before), set(after) ^ (expected | set(before))
    for name, digest in before.items():
        if name not in expected:
            assert after[name] == digest, f"pre-existing {name} was modified"
    assert result.outputs[-1].name == f"{stem}.report.json"

    report = json.loads((outdir / f"{stem}.report.json").read_text(encoding="utf-8"))
    assert set(report) == REPORT_KEYS
    assert report["booktrans"] == booktrans.__version__
    assert report["cache_policy"] == CACHE_POLICY
    assert report["status"] in ("complete", "complete-with-warnings", "nothing-translated")
    assert report["status"] == result.report.status
    assert set(report["outputs"]) == expected - {f"{stem}.report.json"}
    # Unreliable sentences: occurrences vs distinct sentences vs issue codes, all consistent.
    listed = report["unreliable_sentences"]
    assert len(listed) == report["unique_sentences_unreliable"] == result.stats.unresolved
    assert sum(e["occurrences"] for e in listed) == report["sentences_unreliable"]
    assert len({e["id"] for e in listed}) == len(listed)
    codes = {}
    for e in listed:
        assert e["issues"] and e["pages"]
        for code in e["issues"]:
            codes[code] = codes.get(code, 0) + 1
    assert codes == report["issue_counts"]

    with pymupdf.open(src) as s:
        n_pages = s.page_count
    for name, info in report["outputs"].items():
        path = outdir / name
        assert info["bytes"] == path.stat().st_size
        if name.endswith(".pdf"):
            assert info["bytes"] <= SIZE_BUDGET
            fonts = font_streams(path)
            assert fonts["duplicates"] == 0 and {k: info[k] for k in fonts} == fonts
            with pymupdf.open(path) as doc:
                assert doc.page_count == n_pages
                for page in doc:
                    text = page.get_text()
                    assert sum("가" <= c <= "힣" for c in text) >= min_hangul_per_page, f"{name} p.{page.number + 1}"
        else:
            md = path.read_text(encoding="utf-8")
            assert md.startswith("> BookTrans 기계 번역") and "## p.1" in md
    return report
