"""Static DLL dependency check for the frozen app, plus analysis of a reference wheel.

    python check_pe_imports.py --dist dist/BookTrans --report pe-report.txt [--wheel X.whl ...]

For every .exe/.dll/.pyd shipped, every imported DLL (normal and delay-load) must be
either shipped too or a Windows system DLL. Microsoft Visual C++ runtime DLLs are not
part of a clean Windows installation, so they must be shipped. --wheel lists the DLLs a
wheel ships, what its native libraries import, and whether vendor libraries (Intel MKL,
CUDA, cuDNN) are linked in.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
import zipfile
from pathlib import Path

import pefile

VC_REDIST = re.compile(r"^(vcruntime140(_1)?|msvcp140(_\d|_atomic_wait|_codecvt_ids)?|vcomp140|concrt140)\.dll$", re.I)
SYSTEM32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
VENDOR_MARKERS = {
    "Intel MKL": [rb"Intel\(R\) Math Kernel Library", rb"Intel\(R\) oneAPI Math Kernel Library", rb"mkl_serv_", rb"MKL_VERBOSE"],
    "Intel OpenMP": [rb"libiomp5md\.dll", rb"__kmpc_"],
    "CUDA runtime": [rb"cudaGetDeviceCount", rb"cudart64_"],
    "cuDNN": [rb"cudnnCreate", rb"cudnn64_"],
    "cuBLAS": [rb"cublasCreate", rb"cublas64_"],
    "oneDNN": [rb"dnnl_sgemm", rb"ONEDNN_VERBOSE", rb"DNNL_VERBOSE"],
    "MSVC OpenMP": [rb"vcomp140\.dll"],
}


def imports_of(path: Path) -> tuple[list[str], list[str]]:
    pe = pefile.PE(str(path), fast_load=True)
    pe.parse_data_directories(directories=[
        pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"],
        pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT"],
    ])
    normal = [e.dll.decode().lower() for e in getattr(pe, "DIRECTORY_ENTRY_IMPORT", [])]
    delay = [e.dll.decode().lower() for e in getattr(pe, "DIRECTORY_ENTRY_DELAY_IMPORT", [])]
    pe.close()
    return normal, delay


def is_system(name: str) -> bool:
    if name.startswith(("api-ms-win-", "ext-ms-")):
        return True
    if VC_REDIST.match(name):
        return False
    return (SYSTEM32 / name).exists()


def check_dist(dist: Path, out: list[str]) -> list[str]:
    pes = [p for p in sorted(dist.rglob("*")) if p.suffix.lower() in (".exe", ".dll", ".pyd")]
    shipped = {p.name.lower() for p in pes}
    problems = []
    for p in pes:
        try:
            normal, delay = imports_of(p)
        except pefile.PEFormatError:
            continue
        for name, kind in [(n, "import") for n in normal] + [(n, "delay-import") for n in delay]:
            if name in shipped:
                continue
            if is_system(name):
                continue
            problems.append(f"{p.relative_to(dist).as_posix()}: {kind} {name} is neither shipped nor a Windows system DLL")
    out.append(f"# {len(pes)} PE files checked in {dist}")
    out.append("# shipped Visual C++ runtime DLLs: " + ", ".join(sorted(n for n in shipped if VC_REDIST.match(n))))
    out += problems or ["no unresolved dependencies"]
    return problems


def analyze_wheel(wheel: Path, out: list[str]) -> None:
    out.append(f"\n# wheel {wheel.name}")
    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(wheel) as zf:
        for info in zf.infolist():
            if not info.filename.lower().endswith((".dll", ".pyd")):
                continue
            target = Path(tmp) / Path(info.filename).name
            target.write_bytes(zf.read(info))
            data = target.read_bytes()
            normal, delay = imports_of(target)
            found = [label for label, pats in VENDOR_MARKERS.items() if any(re.search(p, data) for p in pats)]
            out.append(f"{info.filename} ({info.file_size} bytes)")
            out.append(f"    imports: {', '.join(normal)}")
            out.append(f"    delay-imports: {', '.join(delay) or '(none)'}")
            out.append(f"    vendor markers found: {', '.join(found) or '(none)'}")
            dependents = [d for d in normal + delay if "cudnn" in d]
            if "cudnn" in info.filename.lower():
                users = []
                for other in zf.namelist():
                    if other.lower().endswith((".dll", ".pyd")) and "cudnn" not in other.lower():
                        o = Path(tmp) / ("_" + Path(other).name)
                        o.write_bytes(zf.read(other))
                        n2, d2 = imports_of(o)
                        if any("cudnn" in d for d in n2 + d2) or re.search(rb"cudnn64_", o.read_bytes()):
                            users.append(Path(other).name)
                out.append(f"    referenced by: {', '.join(users) or 'NO other binary in the wheel imports or names it'}")
            elif dependents:
                out.append(f"    cuDNN imported: {dependents}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", type=Path)
    ap.add_argument("--wheel", type=Path, action="append", default=[])
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()
    out: list[str] = []
    problems = check_dist(args.dist, out) if args.dist else []
    for w in args.wheel:
        analyze_wheel(w, out)
    args.report.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
