"""M2M100Engine framing with stand-ins for SentencePiece and CTranslate2 (no model needed)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from booktrans.engine import ArgosEngine, M2M100Engine, ModelNotFoundError, QuickMTEngine, find_model_dir, load_engine

META = {
    "format": 1, "engine": "m2m100", "name": "m2m100_418M", "revision": "55c2e61bbf05dfb8d7abccdc3fae6fc8512fd636",
    "quantization": "int8", "tokenizer": "sentencepiece.bpe.model",
    "source_lang_token": "__en__", "target_lang_token": "__ko__",
}


QMETA = {
    "format": 1, "engine": "quickmt", "name": "quickmt-en-ko", "revision": "0123456789abcdef0123456789abcdef01234567",
    "quantization": "int8", "source_tokenizer": "src.spm.model", "target_tokenizer": "tgt.spm.model",
}


class FakeSP:
    def encode(self, text, out_type=str):
        return ["▁" + w for w in text.split()]

    def decode(self, pieces):
        assert all(not p.startswith("__") and p not in ("</s>", "<unk>") for p in pieces), pieces
        return "".join(pieces).replace("▁", " ")


class FakeTranslator:
    def __init__(self):
        self.calls = []

    def translate_batch(self, tokens, **kw):
        self.calls.append((tokens, kw))
        return [SimpleNamespace(hypotheses=[["__ko__", "▁안녕", "<unk>", "▁세상", "</s>"]]) for _ in tokens]


def package(tmp_path: Path, meta: dict = META) -> Path:
    (tmp_path / "model").mkdir(parents=True)
    (tmp_path / "model" / "model.bin").write_bytes(b"x")
    (tmp_path / "booktrans-model.json").write_text(json.dumps(meta), encoding="utf-8")
    return tmp_path


def test_source_and_target_framing(tmp_path):
    tr = FakeTranslator()
    eng = M2M100Engine(package(tmp_path), translator=tr, tokenizer=FakeSP())
    assert eng.source_tokens("Hello world") == ["__en__", "▁Hello", "▁world", "</s>"]
    assert eng.translate_batch(["Hello world", "Hi"]) == ["안녕 세상", "안녕 세상"]
    tokens, kw = tr.calls[0]
    assert tokens[1] == ["__en__", "▁Hi", "</s>"]
    assert kw["target_prefix"] == [["__ko__"], ["__ko__"]]
    assert kw["beam_size"] == 4 and "replace_unknowns" not in kw
    eng.translate_batch(["Hi"], careful=True)
    assert tr.calls[-1][1]["beam_size"] == 6 and tr.calls[-1][1]["no_repeat_ngram_size"] == 3
    assert eng.translate_batch([]) == []


def test_engine_id_names_model_revision_and_quantization(tmp_path):
    eng = M2M100Engine(package(tmp_path), translator=FakeTranslator(), tokenizer=FakeSP())
    assert eng.engine_id == "m2m100:m2m100_418M:55c2e61bbf:int8:en-ko"


def test_find_model_dir_accepts_both_layouts(tmp_path):
    m2m = package(tmp_path / "a" / "inner")
    assert find_model_dir(tmp_path / "a") == m2m
    argos = tmp_path / "b" / "pkg"
    (argos / "model").mkdir(parents=True)
    (argos / "model" / "model.bin").write_bytes(b"x")
    (argos / "sentencepiece.model").write_bytes(b"x")
    assert find_model_dir(tmp_path / "b") == argos
    (tmp_path / "c" / "model").mkdir(parents=True)
    (tmp_path / "c" / "model" / "model.bin").write_bytes(b"x")
    with pytest.raises(ModelNotFoundError):
        find_model_dir(tmp_path / "c")


def test_load_engine_dispatch(tmp_path, monkeypatch):
    built = []
    monkeypatch.setattr(M2M100Engine, "__init__", lambda self, d, threads=None: built.append(("m2m", d)))
    monkeypatch.setattr(ArgosEngine, "__init__", lambda self, d, threads=None: built.append(("argos", d)))
    m2m = package(tmp_path / "m")
    assert isinstance(load_engine(m2m), M2M100Engine)
    argos = tmp_path / "argos"
    argos.mkdir()
    assert isinstance(load_engine(argos), ArgosEngine)
    monkeypatch.setattr(QuickMTEngine, "__init__", lambda self, d, threads=None: built.append(("quickmt", d)))
    assert isinstance(load_engine(package(tmp_path / "q", QMETA)), QuickMTEngine)
    assert [k for k, _ in built] == ["m2m", "argos", "quickmt"]
    other = package(tmp_path / "o", {**META, "engine": "nllb"})
    with pytest.raises(ModelNotFoundError):
        load_engine(other)


def test_pieces_outside_the_model_vocabulary_become_unk(tmp_path):
    """M2M100Tokenizer maps a SentencePiece piece missing from vocab.json to <unk> (CI saw
    '▁nearly'); the engine must send CTranslate2 the same tokens."""
    d = package(tmp_path)
    (d / "model" / "shared_vocabulary.json").write_text(json.dumps(["<s>", "<pad>", "</s>", "<unk>", "▁In", "▁1848", "__en__", "__ko__"]))
    eng = M2M100Engine(d, translator=FakeTranslator(), tokenizer=FakeSP())
    assert eng.source_tokens("In nearly 1848") == ["__en__", "▁In", "<unk>", "▁1848", "</s>"]


class Plain:
    """Target-side stand-in: decodes any piece, special tokens must already be gone."""

    def decode(self, pieces):
        assert "</s>" not in pieces and "<unk>" not in pieces, pieces
        return "".join(pieces).replace("▁", " ")


def test_quickmt_framing_matches_the_quickmt_library(tmp_path):
    """quickmt/translator.py: pieces + ["</s>"], no target prefix, disable_unk=True."""
    d = package(tmp_path, QMETA)
    (d / "model" / "source_vocabulary.json").write_text(json.dumps(["<unk>", "<s>", "</s>", "▁Hello", "▁world"]))
    tr = FakeTranslator()
    eng = QuickMTEngine(d, translator=tr, source_tokenizer=FakeSP(), target_tokenizer=Plain())
    assert eng.engine_id == "quickmt:quickmt-en-ko:0123456789:int8:en-ko"
    assert eng.source_tokens("Hello big world") == ["▁Hello", "<unk>", "▁world", "</s>"]
    out = eng.translate_batch(["Hello world"])
    tokens, kw = tr.calls[0]
    assert tokens == [["▁Hello", "▁world", "</s>"]] and "target_prefix" not in kw
    assert kw["disable_unk"] is True
    assert out == ["안녕 세상"]  # __ko__ from the fake hypothesis is dropped as a special token
