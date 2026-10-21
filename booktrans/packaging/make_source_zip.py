"""Build the public BookTrans source ZIP from git-tracked files on an allowlist, then scan it.

    python packaging/make_source_zip.py --out dist/BookTrans-Source-0.1.1.zip

Only files tracked by git under the allowlisted paths are exported (no .git, no
untracked or ignored files). The result is scanned for secrets, personal data and
private-workspace references; any finding fails the build.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ALLOW = ["booktrans/", ".github/workflows/booktrans.yml"]
# Files placed at the archive root so the export can become a standalone repository.
ROOT_COPIES = {"README.md": "booktrans/README.md", "LICENSE": "booktrans/LICENSE", "LICENSE-MIT": "booktrans/LICENSE-MIT"}
FORBIDDEN_PATHS = [
    "*.pdf", "*.sqlite3", "*.log", "*.argosmodel", "*model.bin", "*sentencepiece.model", "*sentencepiece.bpe.model", "*model-archive/*", "*.part",
    "*/.env", ".env", "*.pem", "*.key", "*/__pycache__/*", "*.pyc", "*/models/*", "*/dist/*", "*/build/*",
]
SECRET_PATTERNS = {
    "GitHub token": r"\b(ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}",
    "AWS key": r"\bAKIA[0-9A-Z]{16}\b",
    "private key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "Slack token": r"\bxox[abprs]-[A-Za-z0-9-]{10,}",
    "Anthropic/OpenAI key": r"\b(sk-ant-[A-Za-z0-9_-]{10,}|sk-[A-Za-z0-9]{32,})",
    "email address": r"[A-Za-z0-9._%+-]+@(gmail|naver|daum|hanmail|kakao|outlook|hotmail|yahoo)\.[a-z.]+",
    "session link": r"claude\.ai/code/session|Claude-Session",
}
# The scanner's own pattern table necessarily contains the words it looks for.
SCAN_EXEMPT = {"booktrans/packaging/make_source_zip.py"}


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=REPO, capture_output=True, check=True).stdout
    files = [f for f in out.decode("utf-8").split("\0") if f]
    return sorted(f for f in files if any(f == a or f.startswith(a) for a in ALLOW))


def private_patterns() -> dict[str, str]:
    """Private names (e.g. the source repository and owner) come from the environment so this
    public file never contains them. CI passes them from the GitHub context."""
    terms = [t.strip() for t in os.environ.get("BOOKTRANS_PRIVATE_TERMS", "").split(",") if t.strip()]
    return {f"private term #{i + 1}": re.escape(t) for i, t in enumerate(terms)}


def scan(entries: dict[str, bytes]) -> list[str]:
    patterns = {**SECRET_PATTERNS, **private_patterns()}
    problems = []
    for name, data in entries.items():
        if name.startswith(".git/") or "/.git/" in name:
            problems.append(f"{name}: git metadata")
        for pat in FORBIDDEN_PATHS:
            if fnmatch.fnmatch(name, pat) or fnmatch.fnmatch("/" + name, pat):
                problems.append(f"{name}: forbidden file type ({pat})")
        if name in SCAN_EXEMPT:
            continue
        text = data.decode("utf-8", errors="ignore")
        for label, pat in patterns.items():
            for m in re.finditer(pat, text, re.I):
                line = text.count("\n", 0, m.start()) + 1
                problems.append(f"{name}:{line}: {label}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prefix", default="")
    args = ap.parse_args()
    if not private_patterns():
        print("BOOKTRANS_PRIVATE_TERMS is not set; refusing to export without the private-name scan")
        return 2

    entries: dict[str, bytes] = {}
    for rel in tracked_files():
        entries[rel] = (REPO / rel).read_bytes()
    for root_name, src in ROOT_COPIES.items():
        entries[root_name] = (REPO / src).read_bytes()

    problems = scan(entries)
    if problems:
        print("source export blocked:")
        for p in problems:
            print(f"  {p}")
        return 1

    prefix = args.prefix.rstrip("/") + "/" if args.prefix else ""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(entries):
            info = zipfile.ZipInfo(prefix + name, date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, entries[name])
    print(f"wrote {args.out} ({len(entries)} files)")
    for name in sorted(entries):
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
