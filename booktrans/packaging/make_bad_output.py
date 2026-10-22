"""Write a deliberately broken BookTrans output (the 0.1.1 failure: a full font copy per text
box, tiny overlapping text, no real translation) so CI can prove the verifier rejects it.

    python make_bad_output.py <fixture.pdf> <good-outdir> <bad-outdir>
"""

import shutil
import sys
from pathlib import Path

import pymupdf


def main(fixture: str, good: str, bad: str) -> None:
    src, out = Path(fixture), Path(bad)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(good) / f"{src.stem}.report.json", out)
    for name in (f"{src.stem}.ko.pdf", f"{src.stem}.en-ko.pdf"):
        doc = pymupdf.open(src)
        for i in range(30):
            doc[0].insert_htmlbox(pymupdf.Rect(20, 20 + i, 40, 30 + i), "<p style='font-size:3px'>x가</p>")
        doc.save(out / name)


if __name__ == "__main__":
    main(*sys.argv[1:4])
