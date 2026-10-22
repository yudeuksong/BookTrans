"""Check BookTrans output for one input PDF: structure (always) and translation probes (optional).

    python verify_output.py --input book.pdf --outdir out --render-dir png [--probes probes.json] [--json result.json]

Structure: both PDFs open with the input's page count, stay under the size budget, embed
no font program twice, contain Korean on every page that had translatable English, have no
overlapping text, no text below the readable size and no overflow (from the job report),
and each page renders to a non-blank image (saved for review).

Probes: for given source sentences, the translation shown must match required patterns and
avoid forbidden ones. A probe is "pass", "flagged" (BookTrans withheld the translation as
unreliable) or "wrong" (a translation was shown and it is wrong). Structure failures and
probe failures are reported separately so a build can pass while quality does not.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from booktrans.pdfdoc import CONTINUATION_NOTE, MIN_READABLE_PT, font_streams  # noqa: E402
from booktrans.segment import split_sentences  # noqa: E402

HANGUL = re.compile(r"[가-힣]")
LATIN = re.compile(r"[A-Za-z]")
BUDGET = 5 * 1024 * 1024


def _spans(page) -> list[tuple[pymupdf.Rect, float, str, int]]:
    out = []
    for bi, b in enumerate(page.get_text("dict")["blocks"]):
        if b.get("type") != 0:
            continue
        for li, ln in enumerate(b["lines"]):
            for s in ln["spans"]:
                if s["text"].strip():
                    out.append((pymupdf.Rect(s["bbox"]), s["size"], s["text"], bi * 10000 + li))
    return out


def _overlaps(spans) -> list[str]:
    found = []
    for i in range(len(spans)):
        a, _, ta, la = spans[i]
        for j in range(i + 1, len(spans)):
            b, _, tb, lb = spans[j]
            if la == lb:
                continue
            inter = a & b
            if inter.is_empty:
                continue
            small = min(a.get_area(), b.get_area()) or 1
            if inter.get_area() / small > 0.3:
                found.append(f"{ta[:30]!r} overlaps {tb[:30]!r}")
    return found


def check_structure(src: Path, outdir: Path, render_dir: Path, budget: int) -> tuple[list[str], dict]:
    problems: list[str] = []
    facts: dict = {}
    stem = src.stem
    report = json.loads((outdir / f"{stem}.report.json").read_text(encoding="utf-8"))
    facts["report_status"] = report["status"]
    if report["status"] == "nothing-translated":
        problems.append("nothing was translated")
    if report["fit_overflow"]:
        problems.append(f"{report['fit_overflow']} paragraph(s) overflowed their box")
    with pymupdf.open(src) as s:
        n_pages = s.page_count
        src_latin = [len(LATIN.findall(p.get_text())) for p in s]
    render_dir.mkdir(parents=True, exist_ok=True)
    for name in (f"{stem}.ko.pdf", f"{stem}.en-ko.pdf"):
        path = outdir / name
        info = {"bytes": path.stat().st_size, **font_streams(path)}
        facts[name] = info
        if info["bytes"] > budget:
            problems.append(f"{name}: {info['bytes']} bytes exceeds the {budget}-byte budget")
        if info["duplicates"]:
            problems.append(f"{name}: {info['duplicates']} duplicated embedded font program(s)")
        with pymupdf.open(path) as doc:
            if doc.page_count != n_pages:
                problems.append(f"{name}: {doc.page_count} pages, expected {n_pages}")
            for i, page in enumerate(doc):
                spans = _spans(page)
                hangul = sum(len(HANGUL.findall(t)) for _, _, t, _ in spans)
                small = [t for _, size, t, _ in spans if HANGUL.search(t) and size < MIN_READABLE_PT]
                over = _overlaps(spans)
                facts.setdefault("pages", []).append(
                    {"file": name, "page": i + 1, "hangul": hangul, "small_spans": len(small), "overlaps": len(over)}
                )
                if src_latin[i] >= 200 and hangul < 0.1 * src_latin[i]:
                    problems.append(f"{name} p.{i + 1}: only {hangul} Hangul characters for {src_latin[i]} English letters")
                if small:
                    problems.append(f"{name} p.{i + 1}: {len(small)} Korean span(s) below {MIN_READABLE_PT}pt")
                if over:
                    problems.append(f"{name} p.{i + 1}: overlapping text: {over[:3]}")
                pix = page.get_pixmap(dpi=110)
                png = render_dir / f"{path.stem}-p{i + 1}.png"
                pix.save(png)
                samples = pix.samples
                dark = sum(1 for k in range(0, len(samples), pix.n * 97) if samples[k] < 128)
                if dark < 20:
                    problems.append(f"{name} p.{i + 1}: renders blank")
    return problems, facts


def _md_sentences(md: str) -> dict[str, list[tuple[str, str]]]:
    """Map each source paragraph to its per-sentence (status, shown text) list."""
    out: dict[str, list[tuple[str, str]]] = {}
    current = None
    for block in md.split("\n\n"):
        block = block.strip()
        if not block or block.startswith("##"):
            continue
        if block.startswith(">"):
            if current is None:
                continue
            for line in block.splitlines():
                line = line[1:].strip()
                if line.startswith("⚠ 검토 필요"):
                    out[current].append(("flagged", line.split(":", 1)[-1].strip()))
                elif line == CONTINUATION_NOTE:
                    continue
                else:
                    out[current].append(("ok", line))
        else:
            current = block
            out.setdefault(current, [])
    return out


def check_probes(src: Path, outdir: Path, probes_path: Path) -> tuple[list[dict], dict]:
    md = (outdir / f"{src.stem}.en-ko.md").read_text(encoding="utf-8")
    paragraphs = _md_sentences(md)
    results = []
    for probe in json.loads(probes_path.read_text(encoding="utf-8")):
        sentence = probe["source"]
        status, shown = "missing", ""
        for para, sents in paragraphs.items():
            if sentence not in para:
                continue
            split = split_sentences(re.sub(r" … \d+$", "", para))
            idx = next((i for i, s in enumerate(split) if sentence in s), 0)
            if idx < len(sents):
                status, shown = sents[idx]
            else:
                shown = f"(no translation line for sentence {idx + 1}; paragraph had {len(sents)})"
            break
        else:
            shown = "(source paragraph not found in the .md)"
        verdict = status
        if status == "ok":
            ok = all(re.search(p, shown) for p in probe.get("require", []))
            ok = ok and not any(re.search(p, shown) for p in probe.get("forbid", []))
            verdict = "pass" if ok else "wrong"
        results.append({"source": sentence, "meaning": probe.get("meaning", ""), "verdict": verdict, "shown": shown})
    summary = {v: sum(r["verdict"] == v for r in results) for v in ("pass", "flagged", "wrong", "missing")}
    return results, summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--render-dir", type=Path, required=True)
    ap.add_argument("--probes", type=Path)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--quality-only", action="store_true", help="exit status reflects only the probes")
    args = ap.parse_args()

    problems, facts = check_structure(args.input, args.outdir, args.render_dir, args.budget)
    result = {"structure_problems": problems, "facts": facts}
    print("STRUCTURE:", "PASS" if not problems else "FAIL")
    for p in problems:
        print(f"  - {p}")
    for name in (f"{args.input.stem}.ko.pdf", f"{args.input.stem}.en-ko.pdf"):
        f = facts[name]
        print(f"  {name}: {f['bytes']} bytes, font programs {f['font_programs']} (distinct {f['distinct']}, duplicates {f['duplicates']})")
    quality_failed = False
    if args.probes:
        probes, summary = check_probes(args.input, args.outdir, args.probes)
        result["probes"], result["probe_summary"] = probes, summary
        quality_failed = summary["pass"] != len(probes)
        print("QUALITY PROBES:", "PASS" if not quality_failed else "FAIL", summary)
        for r in probes:
            print(f"  [{r['verdict']}] {r['source']}  =>  {r['shown']}")
    if args.json:
        args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.quality_only:
        return 1 if quality_failed else 0
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
