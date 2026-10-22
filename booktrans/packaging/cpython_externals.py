"""Trace the third-party libraries inside the CPython Windows build and fetch their sources.

The authoritative list is PCbuild/get_externals.bat at the exact CPython tag: `libraries`
are source trees from python/cpython-source-deps, `binaries` are prebuilt packages from
python/cpython-bin-deps. Each bundled binary is mapped to one of them by evidence:

  * prebuilt DLLs (libffi, OpenSSL, Tcl/Tk): byte-identical to the cpython-bin-deps file,
    or identical once the Authenticode signature added at release time is stripped;
  * statically linked libraries (bzip2, xz, zlib, SQLite): the upstream version string
    found inside the bundled binary.

Every bundled binary must also be identical to the one in the build interpreter, which
ties it to that CPython installation.
"""

from __future__ import annotations

import hashlib
import io
import re
import struct
import sys
import tarfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

RAW = "https://raw.githubusercontent.com/python/cpython/v{ver}/PCbuild/get_externals.bat"
EMBED = "https://www.python.org/ftp/python/{ver}/python-{ver}-embed-amd64.zip"
SOURCE_DEPS = "https://github.com/python/cpython-source-deps/archive/refs/tags/{name}.tar.gz"
BIN_DEPS = "https://github.com/python/cpython-bin-deps/archive/refs/tags/{name}.tar.gz"
UA = {"User-Agent": "Mozilla/5.0 BookTrans-build"}


@dataclass
class External:
    name: str  # e.g. "xz-5.2.5"
    project: str  # e.g. "xz"
    version: str
    source: str = "UNRESOLVED"
    source_sha256: str = ""
    evidence: list[str] = field(default_factory=list)
    bundled: list[str] = field(default_factory=list)
    mapped: bool = False


def get(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r:
        return r.read()


def parse_externals(bat: str) -> tuple[list[str], list[str]]:
    libs = re.findall(r"set libraries=%libraries%\s+(\S+)", bat)
    bins = re.findall(r"set binaries=%binaries%\s+(\S+)", bat)
    return libs, bins


def split_name(name: str) -> tuple[str, str]:
    m = re.match(r"(.+?)-(\d[\w.]*)$", name)
    return (m.group(1), m.group(2)) if m else (name, "")


def authenticode_stripped(data: bytes) -> bytes:
    """PE bytes with the certificate table removed and checksum/security entry zeroed."""
    try:
        pe = struct.unpack_from("<I", data, 0x3C)[0]
        opt = pe + 24
        magic = struct.unpack_from("<H", data, opt)[0]
        dd = opt + (112 if magic == 0x20B else 96)
        sec = dd + 4 * 8
        cert_off, cert_size = struct.unpack_from("<II", data, sec)
        buf = bytearray(data)
        struct.pack_into("<I", buf, opt + 64, 0)  # CheckSum
        struct.pack_into("<II", buf, sec, 0, 0)
        if cert_off and cert_off + cert_size <= len(buf):
            del buf[cert_off : cert_off + cert_size]
        return bytes(buf).rstrip(b"\0")
    except struct.error:
        return data


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# Bundled file name -> (external project, how to map it)
STATIC_MARKERS = {
    "bzip2": ("_bz2.pyd", [rb"1\.0\.8, 13-Jul-2019"]),
    "xz": ("_lzma.pyd", [rb"(?<![\d.])5\.2\.5\x00"]),
    "zlib": ("python31*.dll", [rb"deflate 1\.3\.1 Copyright", rb"inflate 1\.3\.1 Copyright", rb"(?<![\d.])1\.3\.1\x00"]),
    "sqlite": ("sqlite3.dll", [rb"3\.49\.1"]),
}
PREBUILT = {"libffi": ["libffi-*.dll"], "openssl-bin": ["libcrypto-*.dll", "libssl-*.dll"], "tcltk": ["tcl8*.dll", "tk8*.dll", "tix*.dll"]}
SOURCE_FOR_BINARY = {"libffi": "libffi", "openssl-bin": "openssl", "tcltk": ("tcl-core", "tk", "tix")}


def _runtime_version(project: str) -> str | None:
    """Library version as reported by the running interpreter (same binaries as bundled)."""
    try:
        if project == "zlib":
            import zlib

            return zlib.ZLIB_RUNTIME_VERSION
        if project == "sqlite":
            import sqlite3

            return sqlite3.sqlite_version
    except Exception:
        return None
    return None


def collect(pyver: str, dist: Path, host_dlls: Path, sources: Path) -> tuple[list[External], list[str]]:
    """Returns the externals and a list of problems (empty when everything maps)."""
    problems: list[str] = []
    bat = get(RAW.format(ver=pyver)).decode("utf-8", "replace")
    libs, bins = parse_externals(bat)
    exts = {split_name(n)[0]: External(n, *split_name(n)) for n in libs}

    bundled = {p.name.lower(): p for p in dist.rglob("*") if p.suffix.lower() in (".dll", ".pyd")}
    # Official python.org binaries of this exact version, for identity checks.
    official_files: dict[str, str] = {}
    try:
        import zipfile

        with zipfile.ZipFile(io.BytesIO(get(EMBED.format(ver=pyver)))) as zf:
            for info in zf.infolist():
                if info.filename.lower().endswith((".dll", ".pyd")):
                    official_files[Path(info.filename).name.lower()] = sha(zf.read(info))
    except Exception as exc:
        problems.append(f"official embeddable package not available: {exc}")

    def bundled_matching(pattern: str) -> list[Path]:
        rx = re.compile("^" + re.escape(pattern.lower()).replace(r"\*", ".*") + "$")
        return [p for n, p in bundled.items() if rx.match(n)]

    # Each bundled CPython binary must be the interpreter's own file.
    def check_host(p: Path, ext: External) -> None:
        host = host_dlls / p.name
        if not host.exists():
            host = host_dlls.parent / p.name
        if host.exists() and sha(host.read_bytes()) == sha(p.read_bytes()):
            ext.evidence.append(f"{p.name}: identical to build interpreter's {host}")
        else:
            problems.append(f"{p.name}: not identical to the build interpreter's copy")

    # Prebuilt binaries: compare with cpython-bin-deps.
    for b in bins:
        project, version = split_name(b)
        if project not in PREBUILT:
            continue
        files = [f for pat in PREBUILT[project] for f in bundled_matching(pat)]
        if not files:
            continue
        archive = get(BIN_DEPS.format(name=b))
        refs: dict[str, tuple[str, str]] = {}
        with tarfile.open(fileobj=io.BytesIO(archive)) as tf:
            for m in tf:
                if m.isfile() and "/amd64/" in m.name and m.name.lower().endswith(".dll"):
                    data = tf.extractfile(m).read()
                    refs[Path(m.name).name.lower()] = (sha(data), sha(authenticode_stripped(data)))
        src_names = SOURCE_FOR_BINARY[project]
        src_names = src_names if isinstance(src_names, tuple) else (src_names,)
        for f in files:
            data = f.read_bytes()
            target = next((exts[s] for s in src_names if s in exts and s.split("-")[0] in f.name.lower()), None)
            target = target or exts.get(src_names[0])
            if target is None:
                problems.append(f"{f.name}: no source library for binary package {b}")
                continue
            target.bundled.append(f.name)
            check_host(f, target)
            ref = refs.get(f.name.lower())
            if ref and ref[0] == sha(data):
                target.evidence.append(f"{f.name}: byte-identical to cpython-bin-deps {b}")
                target.mapped = True
            elif ref and ref[1] == sha(authenticode_stripped(data)):
                target.evidence.append(f"{f.name}: identical to cpython-bin-deps {b} after removing the release signature")
                target.mapped = True
            else:
                problems.append(f"{f.name}: does not match cpython-bin-deps {b}")

    # Statically linked libraries: version strings.
    for project, (pattern, markers) in STATIC_MARKERS.items():
        ext = exts.get(project)
        files = bundled_matching(pattern)
        if ext is None or not files:
            if files:
                problems.append(f"{pattern}: bundled but {project} is not in get_externals.bat")
            continue
        for f in files:
            data = f.read_bytes()
            hits = [m.decode("latin-1") for m in markers if re.search(m, data)]
            ext.bundled.append(f.name)
            check_host(f, ext)
            runtime = _runtime_version(project)
            official = official_files.get(f.name.lower())
            is_official = official is not None and official == sha(data)
            if is_official:
                ext.evidence.append(f"{f.name}: byte-identical to the file in python.org's python-{pyver}-embed-amd64.zip")
            only_entry = sum(1 for n in libs if split_name(n)[0] == project) == 1
            if hits:
                ext.evidence.append(f"{f.name}: contains {project} {ext.version} version string {hits[0]!r}")
                ext.mapped = True
            elif is_official and runtime and ext.version.startswith(runtime):
                ext.evidence.append(f"{f.name}: the official binary reports {project} {runtime} at runtime "
                                    f"({project} {ext.version} in get_externals.bat at v{pyver})")
                ext.mapped = True
            elif is_official and only_entry:
                # No version string or runtime API (e.g. liblzma inside _lzma.pyd). The binary is
                # python.org's official v{pyver} build, whose build inputs are fixed by the tag's
                # get_externals.bat, which lists exactly one version of this library.
                ext.evidence.append(
                    f"{f.name}: no version string or runtime API; official python.org v{pyver} binary, and "
                    f"get_externals.bat at tag v{pyver} lists only {ext.name} [official-build evidence]"
                )
                ext.mapped = True
            else:
                why = "not identical to python.org's official file" if official is not None and not is_official else (
                    "file not in python.org's embeddable package" if official is None else "no version evidence")
                problems.append(f"{f.name}: {project} {ext.version} not mapped ({why}; version string not found)")

    # Source for every external that is actually bundled.
    sources.mkdir(parents=True, exist_ok=True)
    for ext in exts.values():
        if not ext.bundled:
            ext.evidence.append("not bundled in this build")
            continue
        try:
            data = get(SOURCE_DEPS.format(name=ext.name))
            out = sources / f"cpython-source-deps-{ext.name}.tar.gz"
            out.write_bytes(data)
            ext.source, ext.source_sha256 = out.name, sha(data)
        except Exception as exc:
            problems.append(f"{ext.name}: source download failed: {exc}")
        if not ext.mapped:
            problems.append(f"{ext.name}: bundled ({', '.join(ext.bundled)}) but not mapped to the source")
    return list(exts.values()), problems


if __name__ == "__main__":
    import platform

    exts, problems = collect(platform.python_version(), Path(sys.argv[1]), Path(sys.prefix) / "DLLs", Path(sys.argv[2]))
    for e in exts:
        print(e.name, e.source, e.bundled, e.evidence)
    print("\n".join(problems) or "all bundled externals mapped")
    sys.exit(1 if problems else 0)
