"""Rebuild booktrans-model-quickmt-en-ko.zip from the published quickmt-en-ko model.

    pip install huggingface-hub==2.2.0
    python pack_quickmt_en_ko.py <work-dir> <out.zip>

Downloads quickmt/quickmt-en-ko at a fixed revision, copies the files the quickmt
library loads (no conversion, weights unchanged), adds metadata, the model card, a
CC BY 4.0 attribution notice and a provenance file, and writes a reproducible zip
(sorted entries, fixed timestamps). The result must have the SHA-256 in EXPECTED_SHA256.
Same output as packaging/build_quickmt_model.py in the BookTrans source.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

REPO = "quickmt/quickmt-en-ko"
REVISION = "a32fa395ebc7b722a0c021e213ca5fec4065c5df"
PAGE = "https://huggingface.co/quickmt/quickmt-en-ko"
PREFIX = "quickmt-en-ko"
EXPECTED_SHA256 = "2ef78d406cddf53fa621db0a5ec66a6cb8745932ec25a6d976fe94df6efda7cd"
MODEL_FILES = ["model.bin", "config.json", "source_vocabulary.json", "target_vocabulary.json"]
TOKENIZERS = ["src.spm.model", "tgt.spm.model"]
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def write_zip(root: Path, out: Path, prefix: str) -> None:
    entries = sorted((p for p in root.rglob("*") if p.is_file() and not p.is_symlink()),
                     key=lambda p: p.relative_to(root).as_posix())
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in entries:
            info = zipfile.ZipInfo(f"{prefix}/{p.relative_to(root).as_posix()}", date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, p.read_bytes())


def download(snapshot: Path) -> list[Path]:
    from huggingface_hub import snapshot_download

    wanted = [*MODEL_FILES, *TOKENIZERS, "README.md"]
    snapshot_download(REPO, revision=REVISION, local_dir=str(snapshot), allow_patterns=wanted)
    missing = [f for f in wanted if not (snapshot / f).is_file()]
    if missing:
        raise SystemExit(f"{REPO} at {REVISION} lacks {missing}")
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
        "revision": REVISION,
        "license": "CC-BY-4.0 (Hugging Face model metadata; see ATTRIBUTION.txt and MODEL_CARD.md)",
        "quantization": "int8",
        "source_tokenizer": TOKENIZERS[0],
        "target_tokenizer": TOKENIZERS[1],
    }
    (pkg / "booktrans-model.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (pkg / "ATTRIBUTION.txt").write_text(
        "\n".join([
            "Translation model: quickmt-en-ko",
            f"Author: the quickmt project ({PAGE})",
            f"Source: {PAGE} at revision {REVISION}",
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
    lines = [f"Repackaged from https://huggingface.co/{REPO} at revision {REVISION} (no conversion).",
             "", "Files (SHA-256):", *(f"  {sha256(p)}  {p.name}" for p in sources), "",
             "Rebuild: booktrans/packaging/build_quickmt_model.py in the BookTrans source."]
    (pkg / "PROVENANCE.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_zip(pkg, out, PREFIX)
    return sha256(out)


def main(work: str, out: str) -> int:
    work_dir, out_path = Path(work), Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sources = download(work_dir / "snapshot")
    digest = build(work_dir / "snapshot", sources, work_dir / "pkg", out_path)
    print(f"{out_path.name}: {out_path.stat().st_size} bytes, sha256 {digest}")
    if digest != EXPECTED_SHA256:
        print(f"MISMATCH: expected {EXPECTED_SHA256}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
