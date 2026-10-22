"""Token optimization and self-review: send each distinct, translatable sentence to the model
at most once, check every output, and never cache or present an output that failed review.

Savings come from four places, all counted in OptimizerStats:
  * skip    - segments with nothing to translate (page numbers, URLs, formulas, code)
  * dedupe  - repeated sentences within a document (running heads, captions, refrains)
  * cache   - sentences translated and accepted in an earlier run (persistent SQLite memory)
  * batching - remaining sentences are sorted by length so batches carry little padding
"""

from __future__ import annotations

import collections
import hashlib
import re
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from booktrans.engine import TranslationEngine
from booktrans.quality import review

CACHE_POLICY_NAME = "bt0.1.2"

_URL = re.compile(r"^(https?://|www\.)\S+$|^\S+@\S+\.\w+$", re.I)
_NO_WORDS = re.compile(r"^[\W\d_]*$")
_CODEISH = re.compile(r"[{};=<>]{3,}|^\s*(def|class|import|return|for|while)\b.*[:;{]$")
_ROMAN = re.compile(r"^(?=[mdclxvi])m*(c[md]|d?c{0,3})(x[cl]|l?x{0,3})(i[xv]|v?i{0,3})\.?$", re.I)


def should_skip(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    if _NO_WORDS.match(t) or _URL.match(t) or _ROMAN.match(t):
        return True
    letters = sum(c.isalpha() for c in t)
    if letters < 2:
        return True
    # Mostly symbols/digits (equations, tables of numbers).
    if letters / len(t) < 0.35:
        return True
    return bool(_CODEISH.search(t))


def _canon(value) -> str:
    """repr() that does not depend on hash randomization (sets are sorted)."""
    if isinstance(value, (set, frozenset)):
        return "{" + ",".join(sorted(_canon(v) for v in value)) + "}"
    if isinstance(value, (list, tuple)):
        return "(" + ",".join(_canon(v) for v in value) + ")"
    if isinstance(value, dict):
        return "{" + ",".join(f"{_canon(k)}:{_canon(v)}" for k, v in value.items()) + "}"
    if hasattr(value, "co_code"):
        return _code_canon(value)
    return repr(value)


def _code_canon(code) -> str:
    return code.co_code.hex() + _canon(code.co_consts) + _canon(code.co_names)


def _policy_fingerprint() -> str:
    """Hash of the code and settings that decide what a cached translation means: the
    review rules, the preprocessing and the decoding options. Any change to them gives a
    new cache namespace, so translations accepted by older rules are not looked up (the
    user's cache file itself is left alone)."""
    from booktrans import engine, quality

    h = hashlib.sha256()
    for name in sorted(vars(quality)):
        value = getattr(quality, name)
        if isinstance(value, re.Pattern):
            h.update(f"{name}={value.pattern}".encode())
        elif callable(value) and getattr(value, "__module__", "") == quality.__name__ and hasattr(value, "__code__"):
            h.update(f"{name}={_code_canon(value.__code__)}".encode())
        elif isinstance(value, (str, int, float, dict, list, tuple, set, frozenset)) and not name.startswith("__"):
            h.update(f"{name}={_canon(value)}".encode())
    h.update(_code_canon(model_input.__code__).encode())
    h.update(repr((engine.FAST, engine.CAREFUL)).encode())
    return h.hexdigest()[:16]


def model_input(text: str) -> str:
    """Text as the model should see it. All-caps headings ("YOU MUST BE BORN AGAIN") are
    rare in training data and get mistranslated, so they are sent in sentence case."""
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 2 or sum(c.isupper() for c in letters) / len(letters) < 0.85:
        return text
    lowered = text.lower()
    out = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), lowered)
    return re.sub(r"\bi\b", "I", out)


@dataclass
class SegmentResult:
    source: str
    text: str  # what to show: the translation, or the source when skipped/unreliable
    status: str  # ok | skipped | unreliable
    issues: list[str] = field(default_factory=list)
    candidate: str = ""  # best machine output for an unreliable segment (shown only as such)


@dataclass
class OptimizerStats:
    segments: int = 0
    skipped: int = 0
    deduped: int = 0
    cache_hits: int = 0
    cache_rejected: int = 0
    translated: int = 0
    source_tokens_total: int = 0
    source_tokens_sent: int = 0
    retried: int = 0
    unresolved: int = 0
    issue_counts: collections.Counter = field(default_factory=collections.Counter)

    @property
    def tokens_saved(self) -> int:
        return self.source_tokens_total - self.source_tokens_sent

    def summary(self) -> str:
        pct = 100 * self.tokens_saved / self.source_tokens_total if self.source_tokens_total else 0.0
        issues = ", ".join(f"{k}={v}" for k, v in sorted(self.issue_counts.items())) or "none"
        return (
            f"segments={self.segments} skipped={self.skipped} deduped={self.deduped} "
            f"cache_hits={self.cache_hits} cache_rejected={self.cache_rejected} translated={self.translated} retried={self.retried} "
            f"unreliable={self.unresolved} tokens_sent={self.source_tokens_sent}/"
            f"{self.source_tokens_total} ({pct:.1f}% saved) issues: {issues}"
        )


class TranslationMemory:
    """Persistent sentence cache keyed by engine, cache policy and source text."""

    def __init__(self, path: Path | None):
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path) if path else ":memory:", check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS tm (key TEXT PRIMARY KEY, target TEXT NOT NULL)")
        self._db.commit()

    @staticmethod
    def key(engine_id: str, source: str) -> str:
        return hashlib.sha256(f"{engine_id}\x00{source}".encode("utf-8")).hexdigest()

    def get_many(self, engine_id: str, sources: Sequence[str]) -> dict[str, str]:
        found: dict[str, str] = {}
        with self._lock:
            for src in sources:
                row = self._db.execute("SELECT target FROM tm WHERE key=?", (self.key(engine_id, src),)).fetchone()
                if row:
                    found[src] = row[0]
        return found

    def put_many(self, engine_id: str, pairs: dict[str, str]) -> None:
        with self._lock:
            self._db.executemany(
                "INSERT OR REPLACE INTO tm (key, target) VALUES (?, ?)",
                [(self.key(engine_id, s), t) for s, t in pairs.items()],
            )
            self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()


ProgressFn = Callable[[int, int], None]


class OptimizingTranslator:
    def __init__(
        self,
        engine: TranslationEngine,
        memory: TranslationMemory,
        *,
        batch_size: int = 16,
        cancel: threading.Event | None = None,
    ):
        self.engine = engine
        self.memory = memory
        self.batch_size = batch_size
        self.cancel = cancel or threading.Event()
        self.stats = OptimizerStats()
        self.cache_id = f"{engine.engine_id}|{CACHE_POLICY}"

    def translate(self, segments: Sequence[str], progress: ProgressFn | None = None) -> list[SegmentResult]:
        """One result per input segment, in order."""
        stats = self.stats
        stats.segments += len(segments)
        token_counts: dict[str, int] = {}
        unique: list[str] = []
        seen: set[str] = set()
        for seg in segments:
            if seg not in token_counts:
                token_counts[seg] = self.engine.count_tokens(model_input(seg))
            stats.source_tokens_total += token_counts[seg]
            if should_skip(seg):
                stats.skipped += 1
            elif seg in seen:
                stats.deduped += 1
            else:
                seen.add(seg)
                unique.append(seg)

        # Cached translations are re-reviewed on every use; one that the current rules reject
        # is translated again (and overwritten only if the new one passes).
        results: dict[str, SegmentResult] = {}
        for s, t in self.memory.get_many(self.cache_id, unique).items():
            if review(model_input(s), t):
                stats.cache_rejected += 1
            else:
                results[s] = SegmentResult(s, t, "ok")
        stats.cache_hits += len(results)
        pending = sorted((s for s in unique if s not in results), key=lambda s: token_counts[s])

        done = 0
        for i in range(0, len(pending), self.batch_size):
            if self.cancel.is_set():
                raise CancelledError()
            batch = pending[i : i + self.batch_size]
            stats.source_tokens_sent += sum(token_counts[s] for s in batch)
            outputs = self.engine.translate_batch([model_input(s) for s in batch])
            fresh = self._review(batch, outputs)
            results.update(fresh)
            # Only reviewed, accepted translations enter the persistent memory.
            self.memory.put_many(self.cache_id, {s: r.text for s, r in fresh.items() if r.status == "ok"})
            stats.translated += len(batch)
            done += len(batch)
            if progress:
                progress(done, len(pending))

        return [results.get(seg) or SegmentResult(seg, seg, "skipped") for seg in segments]

    def _review(self, sources: list[str], outputs: list[str]) -> dict[str, SegmentResult]:
        """Check each output; retry failures with careful decoding, then clause by clause."""
        accepted: dict[str, SegmentResult] = {}
        failed: list[tuple[str, str]] = []
        for src, out in zip(sources, outputs):
            if not review(model_input(src), out):
                accepted[src] = SegmentResult(src, out, "ok")
            else:
                failed.append((src, out))
        if not failed:
            return accepted

        self.stats.retried += len(failed)
        retry = self.engine.translate_batch([model_input(s) for s, _ in failed], careful=True)
        for (src, first), out in zip(failed, retry):
            issues = review(model_input(src), out)
            if not issues:
                accepted[src] = SegmentResult(src, out, "ok")
                continue
            pieced = self._translate_in_clauses(src)
            if pieced is not None:
                accepted[src] = SegmentResult(src, pieced, "ok")
                continue
            self.stats.unresolved += 1
            for issue in issues:
                self.stats.issue_counts[issue.code] += 1
            accepted[src] = SegmentResult(src, src, "unreliable", [i.code for i in issues], candidate=out)
        return accepted

    def _translate_in_clauses(self, src: str) -> str | None:
        clauses = [c for c in re.split(r"(?<=[,;:])\s+", src) if c]
        if len(clauses) < 2:
            return None
        outs = self.engine.translate_batch([model_input(c) for c in clauses], careful=True)
        if any(review(model_input(c), o) for c, o in zip(clauses, outs)):
            return None
        joined = " ".join(outs)
        return joined if not review(model_input(src), joined) else None


class CancelledError(RuntimeError):
    pass


CACHE_POLICY = f"{CACHE_POLICY_NAME}:{_policy_fingerprint()}"
