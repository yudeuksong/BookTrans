"""Translation engines.

Both engines run a CTranslate2 model with a SentencePiece tokenizer fully offline, without
transformers/torch (which would make the Windows installer several hundred MB larger).

* M2M100Engine (default): Meta's m2m100_418M converted to CTranslate2 int8 and packed by
  packaging/build_m2m100_model.py. The package carries booktrans-model.json.
* ArgosEngine: an Argos Translate package (kept for comparisons and existing installs).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence


class TranslationEngine(Protocol):
    engine_id: str

    def translate_batch(self, texts: Sequence[str], *, careful: bool = False) -> list[str]:
        """Translate texts in order. `careful` asks for slower, repetition-resistant decoding."""

    def count_tokens(self, text: str) -> int: ...


@dataclass(frozen=True)
class DecodeOptions:
    beam_size: int
    repetition_penalty: float
    no_repeat_ngram_size: int


FAST = DecodeOptions(beam_size=4, repetition_penalty=1.0, no_repeat_ngram_size=0)
CAREFUL = DecodeOptions(beam_size=6, repetition_penalty=1.2, no_repeat_ngram_size=3)


class ModelNotFoundError(RuntimeError):
    pass


MODEL_META = "booktrans-model.json"


def _is_model_dir(d: Path) -> bool:
    if not (d / "model" / "model.bin").is_file():
        return False
    return (d / MODEL_META).is_file() or (d / "sentencepiece.model").is_file()


def find_model_dir(root: Path) -> Path:
    """Return the directory under root holding `model/model.bin` plus its tokenizer description
    (booktrans-model.json for a BookTrans package, sentencepiece.model for an Argos package)."""
    for candidate in [root, *sorted(p for p in root.rglob("*") if p.is_dir())]:
        if _is_model_dir(candidate):
            return candidate
    raise ModelNotFoundError(f"No model (model/model.bin + {MODEL_META} or sentencepiece.model) under {root}")


def load_engine(model_dir: Path, *, threads: int | None = None) -> TranslationEngine:
    """The engine matching the package found in model_dir."""
    meta_path = model_dir / MODEL_META
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("engine") != "m2m100":
            raise ModelNotFoundError(f"Unsupported model engine in {meta_path}: {meta.get('engine')!r}")
        return M2M100Engine(model_dir, threads=threads)
    return ArgosEngine(model_dir, threads=threads)


def _threads(threads: int | None) -> int:
    return threads or max(1, (os.cpu_count() or 2) - 1)


class ArgosEngine:
    def __init__(self, model_dir: Path, *, threads: int | None = None):
        import ctranslate2
        import sentencepiece

        self.model_dir = model_dir
        metadata_path = model_dir / "metadata.json"
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        self.engine_id = "argos:{}-{}:{}".format(
            self.metadata.get("from_code", "en"),
            self.metadata.get("to_code", "ko"),
            self.metadata.get("package_version", "?"),
        )
        self.target_prefix = self.metadata.get("target_prefix", "")
        self._sp = sentencepiece.SentencePieceProcessor(model_file=str(model_dir / "sentencepiece.model"))
        self._translator = ctranslate2.Translator(
            str(model_dir / "model"),
            device="cpu",
            compute_type="auto",
            intra_threads=_threads(threads),
        )

    def count_tokens(self, text: str) -> int:
        return len(self._sp.encode(text, out_type=str))

    def translate_batch(self, texts: Sequence[str], *, careful: bool = False) -> list[str]:
        if not texts:
            return []
        opts = CAREFUL if careful else FAST
        tokens = [self._sp.encode(t, out_type=str) for t in texts]
        prefix = [[self.target_prefix]] * len(tokens) if self.target_prefix else None
        results = self._translator.translate_batch(
            tokens,
            target_prefix=prefix,
            beam_size=opts.beam_size,
            repetition_penalty=opts.repetition_penalty,
            no_repeat_ngram_size=opts.no_repeat_ngram_size,
            replace_unknowns=True,
            max_batch_size=32,
            # Cap runaway generations; Korean rarely needs more subwords than 2x the source.
            max_decoding_length=256,
        )
        out = []
        for res in results:
            hyp = res.hypotheses[0]
            if self.target_prefix and hyp and hyp[0] == self.target_prefix:
                hyp = hyp[1:]
            out.append(self._sp.decode(hyp).strip())
        return out


class M2M100Engine:
    """m2m100 through CTranslate2, tokenized exactly like transformers' M2M100Tokenizer:
    source = [__en__] + SentencePiece pieces + [</s>], target prefix = [__ko__]; a piece the
    SentencePiece model knows but the model vocabulary lacks becomes <unk>, as it does there.
    packaging/check_m2m100_engine.py verifies both against the official tokenizer and model."""

    def __init__(self, model_dir: Path, *, threads: int | None = None, translator=None, tokenizer=None):
        self.model_dir = model_dir
        self.metadata = json.loads((model_dir / MODEL_META).read_text(encoding="utf-8"))
        m = self.metadata
        self.engine_id = f"m2m100:{m['name']}:{m['revision'][:10]}:{m['quantization']}:en-ko"
        self.source_token = m["source_lang_token"]
        self.target_token = m["target_lang_token"]
        if tokenizer is None:
            import sentencepiece

            tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(model_dir / m["tokenizer"]))
        if translator is None:
            import ctranslate2

            translator = ctranslate2.Translator(
                str(model_dir / "model"), device="cpu", compute_type="auto", intra_threads=_threads(threads)
            )
        self._sp = tokenizer
        self._translator = translator
        self._vocab = _read_vocabulary(model_dir / "model")

    def count_tokens(self, text: str) -> int:
        return len(self._sp.encode(text, out_type=str))

    def source_tokens(self, text: str) -> list[str]:
        pieces = self._sp.encode(text, out_type=str)
        if self._vocab is not None:
            pieces = [p if p in self._vocab else "<unk>" for p in pieces]
        return [self.source_token, *pieces, "</s>"]

    def translate_batch(self, texts: Sequence[str], *, careful: bool = False) -> list[str]:
        if not texts:
            return []
        opts = CAREFUL if careful else FAST
        results = self._translator.translate_batch(
            [self.source_tokens(t) for t in texts],
            target_prefix=[[self.target_token]] * len(texts),
            beam_size=opts.beam_size,
            repetition_penalty=opts.repetition_penalty,
            no_repeat_ngram_size=opts.no_repeat_ngram_size,
            max_batch_size=32,
            max_decoding_length=256,
        )
        out = []
        for res in results:
            # Drop the language tag and any special token, as skip_special_tokens does.
            hyp = [t for t in res.hypotheses[0] if not _SPECIAL.fullmatch(t)]
            out.append(self._sp.decode(hyp).strip())
        return out


def _read_vocabulary(model_dir: Path) -> frozenset[str] | None:
    js, txt = model_dir / "shared_vocabulary.json", model_dir / "shared_vocabulary.txt"
    if js.is_file():
        return frozenset(json.loads(js.read_text(encoding="utf-8")))
    if txt.is_file():
        return frozenset(txt.read_text(encoding="utf-8").splitlines())
    return None


_SPECIAL = re.compile(r"__[a-z]{2,3}__|<s>|</s>|<pad>|<unk>")
