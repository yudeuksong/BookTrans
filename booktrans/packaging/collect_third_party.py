"""Collect third-party license notices and corresponding source for the frozen Windows build.

Run inside the exact build environment after PyInstaller:

    python packaging/collect_third_party.py --dist dist/BookTrans --sources third-party-src

Writes into --dist:  THIRD_PARTY_NOTICES.txt, licenses/<component>/...
Writes into --sources: downloaded source archives, SOURCES_MANIFEST.txt, BUNDLED_FILES.txt,
BUILD_ENV.txt. Anything that could not be collected is listed as UNRESOLVED instead of
being silently skipped; --strict makes that an error.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import archive_tree  # noqa: E402
import cpython_externals  # noqa: E402
from packaging.markers import default_environment
from packaging.requirements import Requirement

ROOTS = ["pymupdf", "ctranslate2", "sentencepiece"]
USER_AGENT = "Mozilla/5.0 BookTrans-build"
LICENSE_NAME = re.compile(r"(licen[cs]e|copying|notice|authors|copyright|third-party-programs)", re.I)
FLAG_BINARIES = re.compile(r"(mkl|iomp|cudnn|cublas|cudart|nvrtc|intel)", re.I)


@dataclass
class Component:
    name: str
    version: str
    license: str
    homepage: str
    role: str
    license_files: list[str] = field(default_factory=list)
    license_dir: str = ""
    source: str = "UNRESOLVED"
    source_sha256: str = ""
    notes: list[str] = field(default_factory=list)


def fetch(url: str, dest: Path, expected_sha256: str | None = None) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    h = hashlib.sha256()
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req, timeout=300) as resp, tmp.open("wb") as fh:
        while chunk := resp.read(1 << 20):
            h.update(chunk)
            fh.write(chunk)
    digest = h.hexdigest()
    if expected_sha256 and digest != expected_sha256:
        tmp.unlink()
        raise RuntimeError(f"sha256 mismatch for {url}: {digest} != {expected_sha256}")
    tmp.replace(dest)
    return digest


def runtime_closure() -> list[md.Distribution]:
    env = default_environment()
    seen: dict[str, md.Distribution] = {}
    todo = list(ROOTS)
    while todo:
        name = todo.pop()
        key = name.lower().replace("_", "-")
        if key in seen:
            continue
        dist = md.distribution(name)
        seen[key] = dist
        for raw in dist.requires or []:
            req = Requirement(raw)
            if req.marker and not req.marker.evaluate({**env, "extra": ""}):
                continue
            todo.append(req.name)
    return sorted(seen.values(), key=lambda d: d.metadata["Name"].lower())


def license_of(dist: md.Distribution) -> str:
    meta = dist.metadata
    expr = meta.get("License-Expression") or meta.get("License") or ""
    classifiers = [c.split("::")[-1].strip() for c in meta.get_all("Classifier") or [] if c.startswith("License")]
    expr = expr.strip().splitlines()[0] if expr.strip() else ""
    return expr or ", ".join(classifiers) or "UNKNOWN (see license files)"


def homepage_of(dist: md.Distribution) -> str:
    meta = dist.metadata
    urls = [u.split(",", 1)[-1].strip() for u in meta.get_all("Project-URL") or []]
    return meta.get("Home-page") or (urls[0] if urls else "")


def copy_license_files(dist: md.Distribution, dest: Path) -> list[str]:
    copied = []
    for f in dist.files or []:
        parts = Path(str(f)).parts
        in_dist_info = parts[0].endswith(".dist-info")
        if not LICENSE_NAME.search(f.name) and not (in_dist_info and "licenses" in parts):
            continue
        if not in_dist_info and f.suffix in (".py", ".pyc", ".pyi"):
            continue
        src = Path(dist.locate_file(f))
        if not src.is_file():
            continue
        target = dest / Path(*parts[1:]) if in_dist_info else dest / Path(*parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
        copied.append(target.relative_to(dest).as_posix())
    return copied


def licenses_from_archive(archive: Path, dest: Path, max_depth: int = 3) -> list[str]:
    """Copy license-like files (top few directory levels) out of a source archive."""
    copied: list[str] = []

    def take(name: str, read) -> None:
        parts = Path(name).parts
        if len(parts) < 2 or len(parts) > max_depth + 1 or not LICENSE_NAME.search(parts[-1]):
            return
        if Path(name).suffix.lower() in (".c", ".h", ".py", ".cc", ".cpp", ".html", ".in"):
            return
        data = read()
        if data is None or len(data) > 1_000_000:
            return
        target = dest / Path(*parts[1:])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        copied.append(target.relative_to(dest).as_posix())

    if archive.suffix == ".zip":
        import zipfile

        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                if not info.is_dir():
                    take(info.filename, lambda i=info: zf.read(i))
    else:
        with tarfile.open(archive) as tf:
            for m in tf:
                if m.isfile():
                    take(m.name, lambda m=m: tf.extractfile(m).read())
    return copied


def pypi_sdist(name: str, version: str) -> tuple[str, str, str] | None:
    url = f"https://pypi.org/pypi/{name}/{version}/json"
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}), timeout=60) as resp:
        data = json.load(resp)
    for u in data["urls"]:
        if u["packagetype"] == "sdist":
            return u["url"], u["filename"], u["digests"]["sha256"]
    return None


def collect_source(comp: Component, sources: Path, pypi_name: str | None, local: dict[str, Path]) -> None:
    key = comp.name.lower()
    if key in local:
        src = local[key]
        shutil.copyfile(src, sources / src.name)
        comp.source, comp.source_sha256 = src.name, hashlib.sha256(src.read_bytes()).hexdigest()
        comp.notes.append(f"source built into this wheel, supplied by the build ({src.name})")
        return
    try:
        if pypi_name:
            sdist = pypi_sdist(pypi_name, comp.version)
            if sdist:
                url, filename, digest = sdist
                fetch(url, sources / filename, digest)
                comp.source, comp.source_sha256 = filename, digest
                comp.notes.append(f"PyPI sdist {url} (sha256 verified against PyPI)")
                return
        comp.notes.append("no source location known")
    except Exception as exc:  # recorded, never hidden
        comp.notes.append(f"source download failed: {exc}")


def try_urls(comp: Component, sources: Path, candidates: list[str], filename: str) -> None:
    for url in candidates:
        try:
            comp.source_sha256 = fetch(url, sources / filename)
            comp.source = filename
            comp.notes.append(f"downloaded from {url}")
            return
        except Exception as exc:
            comp.notes.append(f"{url}: {exc}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", type=Path, required=True)
    ap.add_argument("--sources", type=Path)
    ap.add_argument("--zip", type=Path, help="write the --sources directory to this zip")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--status-json", type=Path, help="write the gate result (what is unresolved) here")
    ap.add_argument("--local-source", action="append", default=[], metavar="NAME=ARCHIVE",
                    help="source archive for a component built in this pipeline (e.g. ctranslate2=...tar.gz)")
    ap.add_argument("--extra", action="append", default=[], metavar="NAME|VERSION|LICENSE|ROLE|ARCHIVE",
                    help="statically linked component not visible as a Python package")
    args = ap.parse_args()
    local = {k.lower(): Path(v) for k, v in (a.split("=", 1) for a in args.local_source)}

    licenses = args.dist / "licenses"
    if licenses.exists():
        shutil.rmtree(licenses)
    licenses.mkdir(parents=True)
    if args.sources:
        args.sources.mkdir(parents=True, exist_ok=True)

    comps: list[Component] = []
    for dist in runtime_closure():
        name, version = dist.metadata["Name"], dist.version
        comp = Component(name, version, license_of(dist), homepage_of(dist), "Python package bundled in the app")
        comp.license_dir = f"{name}-{version}"
        comp.license_files = copy_license_files(dist, licenses / comp.license_dir)
        if args.sources:
            collect_source(comp, args.sources, name, local)
        comps.append(comp)

    for spec in args.extra:
        name, version, lic, role, archive = spec.split("|")
        comp = Component(name, version, lic, "", role)
        if args.sources:
            src = Path(archive)
            shutil.copyfile(src, args.sources / src.name)
            comp.source, comp.source_sha256 = src.name, hashlib.sha256(src.read_bytes()).hexdigest()
            comp.notes.append(f"source supplied by the build ({src.name})")
        comps.append(comp)

    pyi = md.distribution("pyinstaller")
    comp = Component("PyInstaller (bootloader only)", pyi.version, license_of(pyi), homepage_of(pyi),
                     "bootloader embedded in BookTrans.exe; GPL-2.0 with bootloader exception")
    comp.license_dir = f"pyinstaller-{pyi.version}"
    comp.license_files = copy_license_files(pyi, licenses / comp.license_dir)
    if args.sources:
        collect_source(comp, args.sources, "pyinstaller", local)
    comps.append(comp)

    import pymupdf

    mupdf_version = pymupdf.VersionFitz
    comp = Component("MuPDF (inside PyMuPDF)", mupdf_version, "AGPL-3.0 (or Artifex commercial)", "https://mupdf.com",
                     "C library statically included in the PyMuPDF wheel, with its thirdparty/ libraries")
    if args.sources:
        try_urls(comp, args.sources, [f"https://mupdf.com/downloads/archive/mupdf-{mupdf_version}-source.tar.gz"],
                 f"mupdf-{mupdf_version}-source.tar.gz")
    comps.append(comp)

    pyver = platform.python_version()
    comp = Component("CPython", pyver, "PSF-2.0 (Windows build also lists bundled library licenses)",
                     "https://www.python.org", "Python runtime (python3*.dll and stdlib) bundled by PyInstaller")
    lic = Path(sys.base_prefix) / "LICENSE.txt"
    if lic.is_file():
        (licenses / f"python-{pyver}").mkdir()
        shutil.copyfile(lic, licenses / f"python-{pyver}" / "LICENSE.txt")
        comp.license_files = ["LICENSE.txt"]
        comp.license_dir = f"python-{pyver}"
    if args.sources:
        try_urls(comp, args.sources, [f"https://www.python.org/ftp/python/{pyver}/Python-{pyver}.tar.xz"],
                 f"Python-{pyver}.tar.xz")
    comps.append(comp)

    # Libraries inside the CPython Windows build, traced via PCbuild/get_externals.bat.
    externals_problems: list[str] = []
    if args.sources:
        exts, externals_problems = cpython_externals.collect(
            pyver, args.dist, Path(sys.base_prefix) / "DLLs", args.sources
        )
        for e in exts:
            if not e.bundled:
                continue
            comp = Component(f"{e.project} (CPython {pyver} Windows build)", e.version, "see CPython LICENSE.txt and source",
                             "", f"bundled as {', '.join(sorted(set(e.bundled)))}", notes=list(e.evidence))
            comp.source, comp.source_sha256 = e.source, e.source_sha256
            if not e.mapped:
                comp.notes.append("PARTIAL: bundled binary not mapped to this source")
            comps.append(comp)

    comps.append(Component(
        "Microsoft Visual C++ runtime (vcruntime140*.dll, msvcp140*.dll, vcomp140.dll)", "14.x",
        "Microsoft Visual C++ Redistributable license", "https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist",
        "compiler runtime shipped next to the binaries; a System Library under AGPL-3.0 section 1 (no source required)",
        source="not applicable (System Library)"))

    if args.sources:
        for c in comps:
            if not c.license_files and (args.sources / c.source).is_file():
                c.license_dir = c.license_dir or f"{c.name.split(' ')[0]}-{c.version}"
                c.license_files = licenses_from_archive(args.sources / c.source, licenses / c.license_dir)
                if c.license_files:
                    c.notes.append("license files taken from the source archive (not present in the wheel)")

    # Inventory of every native binary actually shipped, so nothing hides behind a package name.
    rows = []
    for f in sorted(args.dist.rglob("*")):
        if f.is_file() and f.suffix.lower() in (".dll", ".pyd", ".exe", ".so"):
            digest = hashlib.sha256(f.read_bytes()).hexdigest()
            flag = "  <-- REVIEW (vendor runtime, check redistribution/AGPL compatibility)" if FLAG_BINARIES.search(f.name) else ""
            rows.append(f"{digest}  {f.stat().st_size:>10}  {f.relative_to(args.dist).as_posix()}{flag}")
    flagged = [r for r in rows if "REVIEW" in r]

    unresolved = [c for c in comps if args.sources and (c.source == "UNRESOLVED" or any("PARTIAL" in n for n in c.notes))]
    missing_license = [c for c in comps if not c.license_files and "CPython" not in c.name and "System Library" not in c.role]

    lines = [
        "BookTrans — THIRD-PARTY NOTICES",
        "================================",
        "",
        "BookTrans 설치 파일은 아래 구성 요소를 포함합니다. 각 구성 요소의 라이선스 원문은",
        "licenses/ 폴더에 있습니다. 결합 배포물은 AGPL-3.0-only(LICENSE)로 배포되며,",
        "BookTrans 자체 소스 코드는 MIT(LICENSE-MIT)로도 제공됩니다.",
        "",
        "번역 모델(Argos Translate translate-en_ko 1.1)은 포함되어 있지 않습니다. 사용자가 동의하면",
        "배포처에서 내려받으며, 모델의 라이선스는 배포처에 명시되어 있지 않습니다.",
        "",
        "Windows 설치 프로그램은 Inno Setup(https://jrsoftware.org/isinfo.php)으로 만들어졌습니다.",
        "",
    ]
    for c in comps:
        lines += [f"* {c.name} {c.version}", f"    License: {c.license}"]
        if c.homepage:
            lines.append(f"    Homepage: {c.homepage}")
        lines.append(f"    Role: {c.role}")
        if c.license_files:
            lines.append(f"    License files: licenses/{c.license_dir}/ ({len(c.license_files)} files)")
        else:
            lines.append("    License files: NOT FOUND in the installed package (see homepage)")
        if args.sources:
            lines.append(f"    Source: {c.source}" + (f" sha256={c.source_sha256}" if c.source_sha256 else ""))
        lines.append("")
    (args.dist / "THIRD_PARTY_NOTICES.txt").write_text("\n".join(lines), encoding="utf-8")

    if args.sources:
        manifest = ["# Corresponding source collected for this build", ""]
        for c in comps:
            manifest.append(f"{c.name} {c.version}: {c.source} {c.source_sha256}")
            manifest += [f"    {n}" for n in c.notes]
        manifest += ["", "# UNRESOLVED / PARTIAL"] + ([f"- {c.name} {c.version}: {c.source}" for c in unresolved] or ["(none)"])
        manifest += ["", "# CPython externals problems"] + (externals_problems or ["(none)"])
        manifest += ["", "# Shipped binaries flagged for review"] + (flagged or ["(none)"])
        (args.sources / "SOURCES_MANIFEST.txt").write_text("\n".join(manifest), encoding="utf-8")
        (args.sources / "BUNDLED_FILES.txt").write_text("\n".join(rows), encoding="utf-8")
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze", "--all"], capture_output=True, text=True, check=True).stdout
        (args.sources / "BUILD_ENV.txt").write_text(
            f"python {sys.version}\nplatform {platform.platform()}\n\n# pip freeze --all\n{freeze}", encoding="utf-8"
        )
        if args.zip:
            archive_tree.main(str(args.sources), str(args.zip), args.zip.stem)

    print(f"components: {len(comps)}; native binaries: {len(rows)}; flagged binaries: {len(flagged)}")
    for r in flagged:
        print(f"FLAGGED: {r}")
    for c in unresolved:
        print(f"UNRESOLVED: {c.name} {c.version}: {'; '.join(c.notes[-2:])}")
    for c in missing_license:
        print(f"NO LICENSE FILE: {c.name} {c.version}")
    for p in externals_problems:
        print(f"EXTERNALS: {p}")
    failed = unresolved or missing_license or externals_problems or flagged
    if args.status_json:
        args.status_json.write_text(json.dumps({
            "passed": not failed,
            "unresolved_sources": [f"{c.name} {c.version}: {c.source}" for c in unresolved],
            "missing_license_files": [f"{c.name} {c.version}" for c in missing_license],
            "cpython_externals_problems": externals_problems,
            "flagged_binaries": flagged,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    return 1 if args.strict and failed else 0


if __name__ == "__main__":
    sys.exit(main())
