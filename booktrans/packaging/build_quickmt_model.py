"""Build the BookTrans model package from quickmt/quickmt-en-ko (CI only).

    python build_quickmt_model.py <work-dir> <out.zip> [--twice]

The model is published in CTranslate2 format, so nothing is converted: the files the
quickmt library loads (model.bin, config.json, vocabularies, src/tgt SentencePiece models)
are copied as published at the pinned revision (models.MODEL_REVISION), next to
booktrans-model.json, the model card, a CC-BY-4.0 attribution notice and a provenance
file, into a reproducible zip. The eole training checkpoint is left out.

--twice packs a second time and fails unless both zips are byte-identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from archive_tree import write_zip  # noqa: E402
from booktrans.engine import MODEL_META  # noqa: E402
from booktrans.models import MODEL_NAME, MODEL_PAGE, MODEL_REVISION  # noqa: E402

REPO = "quickmt/quickmt-en-ko"
MODEL_FILES = ["model.bin", "config.json", "source_vocabulary.json", "target_vocabulary.json"]
TOKENIZERS = ["src.spm.model", "tgt.spm.model"]
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def download(snapshot: Path) -> list[Path]:
    from huggingface_hub import snapshot_download

    wanted = [*MODEL_FILES, *TOKENIZERS, "README.md"]
    snapshot_download(REPO, revision=MODEL_REVISION, local_dir=str(snapshot), allow_patterns=wanted)
    missing = [f for f in wanted if not (snapshot / f).is_file()]
    if missing:
        raise SystemExit(f"{REPO} at {MODEL_REVISION} lacks {missing}")
    return [snapshot / f for f in wanted]


def build(snapshot: Path, sources: list[Path], pkg: Path, out: Path) -> str:
    if pkg.exists():
        shutil.rmtree(pkg)
    (pkg / "model").mkdir(parents=True)
    for f in MODEL_FILES:
        shutil.copyfile(snapshot / f, pkg / "model" / f)
    for f in TOKENIZERS:
        shutil.copyfile(snapshot / f, pkg / f)
    shutil.copyfile(snapshot / "README.md", pkg / "MODEL_CARD.md")
    meta = {
        "format": 1,
        "engine": "quickmt",
        "name": "quickmt-en-ko",
        "source": REPO,
        "revision": MODEL_REVISION,
        "license": "CC-BY-4.0 (Hugging Face model metadata; see ATTRIBUTION.txt and MODEL_CARD.md)",
        # CTranslate2 compute type BookTrans requests when loading the model on the CPU.
        "quantization": "int8",
        "source_tokenizer": TOKENIZERS[0],
        "target_tokenizer": TOKENIZERS[1],
    }
    (pkg / MODEL_META).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (pkg / "ATTRIBUTION.txt").write_text(
        "\n".join([
            "Translation model: quickmt-en-ko",
            f"Author: the quickmt project ({MODEL_PAGE})",
            f"Source: {MODEL_PAGE} at revision {MODEL_REVISION}",
            f"License: Creative Commons Attribution 4.0 International (CC BY 4.0), {LICENSE_URL}",
            "         legal code: https://creativecommons.org/licenses/by/4.0/legalcode",
            "Changes: none to the model. The files the quickmt library loads were copied as",
            "         published and repackaged for BookTrans (model/ folder, booktrans-model.json);",
            "         the eole training checkpoint was not included.",
            "No warranties are given; see the license.",
            "",
        ]),
        encoding="utf-8",
    )
    lines = [f"Repackaged from https://huggingface.co/{REPO} at revision {MODEL_REVISION} (no conversion).",
             "", "Files (SHA-256):", *(f"  {sha256(p)}  {p.name}" for p in sources), "",
             "Rebuild: booktrans/packaging/build_quickmt_model.py in the BookTrans source."]
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
    print((snapshot / "config.json").read_text(encoding="utf-8"))
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
            print("NOT REPRODUCIBLE: two packings of the same files differ")
            return 1
        again.unlink()
        shutil.rmtree(args.work / "pkg2")
    return 0


if __name__ == "__main__":
    sys.exit(main())
