"""Build the BookTrans model package from facebook/m2m100_418M (CI only).

    python build_m2m100_model.py <work-dir> <out.zip> [--twice]

1. Download the official checkpoint at the pinned revision (models.MODEL_REVISION).
2. Convert it with CTranslate2 to int8 (no transformers/torch needed at run time).
3. Add the SentencePiece model, booktrans-model.json, the license, the model card and a
   provenance file, and write a reproducible zip (sorted entries, fixed timestamps).

--twice converts a second time from the same download and fails unless both zips are
byte-identical, so the pinned MODEL_SHA256 describes a reproducible build.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from archive_tree import write_zip  # noqa: E402
from booktrans.engine import MODEL_META  # noqa: E402
from booktrans.models import MODEL_NAME, MODEL_REVISION  # noqa: E402

REPO = "facebook/m2m100_418M"
TOKENIZER = "sentencepiece.bpe.model"
SOURCE_TOKEN, TARGET_TOKEN = "__en__", "__ko__"
TOOLS = ("ctranslate2", "transformers", "torch", "safetensors", "huggingface-hub")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def download(snapshot: Path) -> list[Path]:
    from huggingface_hub import snapshot_download

    meta = ["config.json", "generation_config.json", "vocab.json", TOKENIZER, "tokenizer_config.json",
            "special_tokens_map.json", "README.md"]
    snapshot_download(REPO, revision=MODEL_REVISION, local_dir=str(snapshot), allow_patterns=[*meta, "*.safetensors"])
    if not any(snapshot.glob("*.safetensors")):
        snapshot_download(REPO, revision=MODEL_REVISION, local_dir=str(snapshot), allow_patterns=["pytorch_model.bin"])
    files = sorted(p for p in snapshot.iterdir() if p.is_file())
    missing = [m for m in (TOKENIZER, "config.json", "vocab.json", "README.md") if not (snapshot / m).is_file()]
    if missing:
        raise SystemExit(f"checkpoint at {MODEL_REVISION} lacks {missing}")
    return files


def vocabulary(model_dir: Path) -> list[str]:
    js, txt = model_dir / "shared_vocabulary.json", model_dir / "shared_vocabulary.txt"
    if js.is_file():
        return json.loads(js.read_text(encoding="utf-8"))
    return txt.read_text(encoding="utf-8").splitlines()


def build(snapshot: Path, sources: list[Path], pkg: Path, out: Path) -> str:
    from ctranslate2.converters import TransformersConverter

    if pkg.exists():
        shutil.rmtree(pkg)
    pkg.mkdir(parents=True)
    TransformersConverter(str(snapshot)).convert(str(pkg / "model"), quantization="int8", force=True)
    vocab = set(vocabulary(pkg / "model"))
    for token in (SOURCE_TOKEN, TARGET_TOKEN, "</s>"):
        if token not in vocab:
            raise SystemExit(f"converted vocabulary lacks {token}")
    shutil.copyfile(snapshot / TOKENIZER, pkg / TOKENIZER)
    shutil.copyfile(snapshot / "README.md", pkg / "MODEL_CARD.md")
    shutil.copyfile(HERE / "model-notices" / "m2m100-LICENSE.txt", pkg / "LICENSE")
    tools = {t: importlib.metadata.version(t) for t in TOOLS}
    meta = {
        "format": 1,
        "engine": "m2m100",
        "name": "m2m100_418M",
        "source": REPO,
        "revision": MODEL_REVISION,
        "license": "MIT (stated on the model card, see MODEL_CARD.md and LICENSE)",
        "quantization": "int8",
        "tokenizer": TOKENIZER,
        "source_lang_token": SOURCE_TOKEN,
        "target_lang_token": TARGET_TOKEN,
        "converter": tools,
    }
    (pkg / MODEL_META).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        f"Converted from https://huggingface.co/{REPO} at revision {MODEL_REVISION}",
        "with ctranslate2.converters.TransformersConverter, quantization int8.",
        "",
        "Source files (SHA-256):",
        *(f"  {sha256(p)}  {p.name}" for p in sources),
        "",
        "Tools:",
        *(f"  {k}=={v}" for k, v in tools.items()),
        "",
        "Rebuild: booktrans/packaging/build_m2m100_model.py in the BookTrans source.",
    ]
    (pkg / "PROVENANCE.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_zip(pkg, out, MODEL_NAME)
    return sha256(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("work", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--twice", action="store_true")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    snapshot = args.work / "snapshot"
    sources = download(snapshot)
    for p in sources:
        print(f"source {p.name}: {p.stat().st_size} bytes")
    first = build(snapshot, sources, args.work / "pkg", args.out)
    print(f"package {args.out.name}: {args.out.stat().st_size} bytes, sha256 {first}")
    for f in sorted((args.work / "pkg").rglob("*")):
        if f.is_file():
            print(f"  {f.stat().st_size:>12}  {f.relative_to(args.work / 'pkg').as_posix()}")
    if args.twice:
        again = args.work / "again.zip"
        second = build(snapshot, sources, args.work / "pkg2", again)
        print(f"second build sha256 {second}")
        if second != first:
            print("NOT REPRODUCIBLE: two conversions of the same checkpoint differ")
            return 1
        again.unlink()
        shutil.rmtree(args.work / "pkg2")
    return 0


if __name__ == "__main__":
    sys.exit(main())
