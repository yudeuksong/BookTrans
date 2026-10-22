"""Check BookTrans's quickmt path against the official quickmt library (CI only).

    python check_quickmt_engine.py <package.zip> <out.json>

BookTrans does not ship the quickmt library; QuickMTEngine re-implements its framing. This
fails unless, on every probe sentence,
* QuickMTEngine.source_tokens() equals the library's Translator.tokenize(), and
* translations agree with the library loaded from the same package files (int8, beam 4,
  one sentence at a time vs BookTrans's batching): mean similarity >= 0.95, and BookTrans
  never lacks Korean where the library produced Korean.
A sentence the model itself leaves in English (the library too) is a model-quality matter,
not a path error: it is listed here and BookTrans's review withholds it (untranslated).
It also prints BookTrans's review verdict per sentence (diagnostic).
"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from booktrans.engine import QuickMTEngine, find_model_dir  # noqa: E402
from booktrans.models import install_model_file  # noqa: E402
from booktrans.optimizer import model_input  # noqa: E402
from booktrans.quality import review  # noqa: E402
from evaluate_models import DEV, HELD_OUT  # noqa: E402

MIN_MEAN_SIMILARITY = 0.95
HANGUL = re.compile(r"[가-힣]")
EXTRA = [
    "Can a person change without help from others?",
    "The second edition appeared in 1762 with three new chapters.",
    "“Well,” she said — and paused; the room (full of people) fell silent…",
    "Dr. Smith's 3rd volume costs $12.50, i.e. less than £10!",
]


def main(archive: str, out: str) -> int:
    from quickmt import Translator

    sentences = [model_input(s) for s, *_ in DEV + HELD_OUT] + [model_input(s) for s in EXTRA]
    problems: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        digest = hashlib.sha256(Path(archive).read_bytes()).hexdigest()
        pkg = find_model_dir(install_model_file(Path(archive), sha256=digest, dest_root=Path(tmp)))
        engine = QuickMTEngine(pkg)
        # The library expects model files and tokenizers in one folder.
        lib_dir = Path(tmp) / "library-layout"
        lib_dir.mkdir()
        for f in (pkg / "model").iterdir():
            (lib_dir / f.name).symlink_to(f.resolve())
        for f in pkg.glob("*.spm.model"):
            (lib_dir / f.name).symlink_to(f.resolve())
        lib = Translator(str(lib_dir), compute_type="int8")

        mismatched = 0
        for s in sentences:
            mine, theirs = engine.source_tokens(s), lib.tokenize([s])[0]
            if mine != theirs:
                mismatched += 1
                print(f"TOKENS DIFFER: {s}\n  booktrans: {mine}\n  library:   {theirs}")
        print(f"tokenization: {len(sentences) - mismatched}/{len(sentences)} identical")
        if mismatched:
            problems.append(f"{mismatched} sentence(s) tokenized differently from the quickmt library")

        mine = engine.translate_batch(sentences)
        theirs = [lib(s, beam_size=4) for s in sentences]

    rows = []
    for s, a, b in zip(sentences, mine, theirs):
        sim = difflib.SequenceMatcher(None, a, b).ratio()
        issues = [i.code for i in review(s, a)]
        rows.append({"source": s, "booktrans": a, "library": b, "similarity": round(sim, 2), "review": issues})
        print(f"[{sim:.2f}] {s}\n  booktrans: {a}{'  (' + ', '.join(issues) + ')' if issues else ''}\n  library:   {b}")
    mean = sum(r["similarity"] for r in rows) / len(rows)
    not_korean = [r["source"] for r in rows if not HANGUL.search(r["booktrans"]) and HANGUL.search(r["library"])]
    model_english = [r["source"] for r in rows if not HANGUL.search(r["booktrans"]) and not HANGUL.search(r["library"])]
    print(f"left in English by the model itself (library too): {model_english}")
    print(f"mean similarity to the quickmt library: {mean:.3f} (minimum {MIN_MEAN_SIMILARITY})")
    if mean < MIN_MEAN_SIMILARITY:
        problems.append(f"translations disagree with the quickmt library (mean similarity {mean:.2f})")
    if not_korean:
        problems.append(f"no Korean in BookTrans's output where the library has Korean: {not_korean}")
    Path(out).write_text(json.dumps({"mean_similarity": round(mean, 3), "problems": problems, "rows": rows},
                                    ensure_ascii=False, indent=2), encoding="utf-8")
    print("RESULT:", "PASS" if not problems else "FAIL")
    for p in problems:
        print(f"  - {p}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
