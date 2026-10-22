"""Write a directory tree to a reproducible .tar.gz (or .zip), skipping .git metadata.

    python archive_tree.py <dir> <out.tar.gz|out.zip> [prefix]

Entries are sorted and carry a fixed timestamp/owner, so identical input gives an
identical archive.
"""

import gzip
import io
import sys
import tarfile
import zipfile
from pathlib import Path

FIXED_TIME = 315532800  # 1980-01-01, the earliest time a zip can store


def files(root: Path):
    for p in sorted(root.rglob("*"), key=lambda p: p.relative_to(root).as_posix()):
        rel = p.relative_to(root)
        if ".git" in rel.parts or p.is_dir() or p.is_symlink():
            continue
        yield p, rel.as_posix()


def write_tar(root: Path, out: Path, prefix: str) -> int:
    count = 0
    with open(out, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=FIXED_TIME, filename="") as gz:
        with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tf:
            for p, rel in files(root):
                data = p.read_bytes()
                info = tarfile.TarInfo(f"{prefix}/{rel}" if prefix else rel)
                info.size, info.mtime, info.mode = len(data), FIXED_TIME, 0o644
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                tf.addfile(info, io.BytesIO(data))
                count += 1
    return count


def write_zip(root: Path, out: Path, prefix: str) -> int:
    count = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for p, rel in files(root):
            info = zipfile.ZipInfo(f"{prefix}/{rel}" if prefix else rel, date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, p.read_bytes())
            count += 1
    return count


def main(src: str, out: str, prefix: str = "") -> None:
    root, target = Path(src), Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    n = write_zip(root, target, prefix) if target.suffix == ".zip" else write_tar(root, target, prefix)
    print(f"wrote {target} ({n} files)")


if __name__ == "__main__":
    main(*sys.argv[1:4])
