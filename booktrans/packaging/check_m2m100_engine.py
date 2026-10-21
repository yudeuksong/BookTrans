"""Check BookTrans's own m2m100 path against the official transformers model (CI only).

    python check_m2m100_engine.py <package.zip> <checkpoint-dir> <out.json>

BookTrans ships no transformers: M2M100Engine tokenizes with SentencePiece alone and runs
the CTranslate2 package. This fails unless
* the converted vocabulary is the official one, id for id;
* M2M100Engine.source_tokens() equals M2M100Tokenizer(src_lang="en") on every probe sentence;
* translations of the probe sentences (batched, as BookTrans runs them) agree with the
  official transformers model (beam 4, forced __ko__) on average, and all are Korean.
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

from booktrans.engine import M2M100Engine, find_model_dir  # noqa: E402
from booktrans.models import install_model_file  # noqa: E402
from booktrans.optimizer import model_input  # noqa: E402
from booktrans.quality import review  # noqa: E402
from evaluate_models import DEV, HELD_OUT  # noqa: E402

MIN_MEAN_SIMILARITY = 0.85
HANGUL = re.compile(r"[가-힣]")
EXTRA = [
    "Can a person change without help from others?",
    "The second edition appeared in 1762 with three new chapters.",
    "“Well,” she said — and paused; the room (full of people) fell silent…",
    "Dr. Smith's 3rd volume costs $12.50, i.e. less than £10!",
]


def main(archive: str, checkpoint: str, out: str) -> int:
    import torch
    import transformers

    sentences = [model_input(s) for s, *_ in DEV + HELD_OUT] + [model_input(s) for s in EXTRA]
    problems: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        digest = hashlib.sha256(Path(archive).read_bytes()).hexdigest()
        engine = M2M100Engine(find_model_dir(install_model_file(Path(archive), sha256=digest, dest_root=Path(tmp))))
        tok = transformers.AutoTokenizer.from_pretrained(checkpoint)
        tok.src_lang = "en"

        vocab_path = engine.model_dir / "model" / "shared_vocabulary.json"
        ours = json.loads(vocab_path.read_text(encoding="utf-8")) if vocab_path.is_file() else \
            (engine.model_dir / "model" / "shared_vocabulary.txt").read_text(encoding="utf-8").splitlines()
        official = tok.convert_ids_to_tokens(list(range(len(tok))))
        differ = [i for i, (a, b) in enumerate(zip(ours, official)) if a != b]
        print(f"vocabulary: package {len(ours)}, official {len(official)}, differing ids {len(differ)}")
        if differ or len(ours) < len(official):
            problems.append(f"vocabulary differs from the official tokenizer (first ids {differ[:5]})")

        mismatched = 0
        for s in sentences:
            mine = engine.source_tokens(s)
            theirs = tok.convert_ids_to_tokens(tok(s)["input_ids"])
            if mine != theirs:
                mismatched += 1
                print(f"TOKENS DIFFER: {s}\n  booktrans: {mine}\n  official:  {theirs}")
        print(f"tokenization: {len(sentences) - mismatched}/{len(sentences)} identical")
        if mismatched:
            problems.append(f"{mismatched} sentence(s) tokenized differently from M2M100Tokenizer")

        mine = engine.translate_batch(sentences)
        model = transformers.AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
        model.eval()
        theirs = []
        with torch.no_grad():
            for s in sentences:
                gen = model.generate(**tok(s, return_tensors="pt"), num_beams=4, max_new_tokens=256,
                                     forced_bos_token_id=tok.get_lang_id("ko"))
                theirs.append(tok.batch_decode(gen, skip_special_tokens=True)[0])

    rows = []
    for s, a, b in zip(sentences, mine, theirs):
        sim = difflib.SequenceMatcher(None, a, b).ratio()
        issues = [i.code for i in review(s, a)]
        rows.append({"source": s, "booktrans": a, "official": b, "similarity": round(sim, 2), "review": issues})
        print(f"[{sim:.2f}] {s}\n  booktrans: {a}{'  (' + ', '.join(issues) + ')' if issues else ''}\n  official:  {b}")
    mean = sum(r["similarity"] for r in rows) / len(rows)
    not_korean = [r["source"] for r in rows if len(HANGUL.findall(r["booktrans"])) < 1]
    print(f"mean similarity to the official model: {mean:.3f} (minimum {MIN_MEAN_SIMILARITY})")
    if mean < MIN_MEAN_SIMILARITY:
        problems.append(f"translations disagree with the official model (mean similarity {mean:.2f})")
    if not_korean:
        problems.append(f"no Korean in the output for: {not_korean}")
    Path(out).write_text(json.dumps({"mean_similarity": round(mean, 3), "problems": problems, "rows": rows},
                                    ensure_ascii=False, indent=2), encoding="utf-8")
    print("RESULT:", "PASS" if not problems else "FAIL")
    for p in problems:
        print(f"  - {p}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
