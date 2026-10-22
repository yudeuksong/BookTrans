"""Synthetic 3-page "book" PDF for layout and quality tests (all text written for this test).

Each visual line is written separately, the way many typeset PDFs store text, so the
extractor has to rebuild paragraphs. Contents: an all-caps title, a CONTENTS page with
leader-dot entries, a PROBLEM heading, a hyphenated line break, a paragraph that continues
onto the next page, numbers, a "not only ... but" sentence, a two-column section, running
heads and page numbers. No images.

    python make_fixture.py out.pdf
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pymupdf

LEFT, RIGHT, TOP, BOTTOM = 72, 523, 80, 760
SIZE, LEAD = 10.5, 14.0

INTRO = (
    "Every reader who opens an old book meets a voice from another age. The eighteenth century was a time "
    "of great change, and many writers of that period believed that a person could be completely trans"
    "formed by new ideas. In 1750 there were twenty-one printing houses in the city, and each of them "
    "published sermons, letters and essays for ordinary people who wanted to understand the world. "
    "Readers did not accept every argument they found, but they argued about them in homes, in shops and "
    "in the streets of the growing towns, where news travelled faster than"
)
CONTINUED = (
    "ever before. This continuation of the previous paragraph begins on a new page, and a good reader "
    "does not notice the break at all."
)
HEART = (
    "Not only can your heart be changed, but your whole life can be renewed. The writer does not promise "
    "that the road will be easy. He says that change begins with a single decision, and that the decision "
    "must be made again every morning."
)
NUMBERS = (
    "The second edition appeared in 1762 with three new chapters. By the nineteenth century the book had "
    "been printed more than forty times, and copies were sold in at least twelve countries."
)
COLUMN_LEFT = (
    "In the left column the author describes the market town. Farmers arrive before dawn with their carts, "
    "and the first customers are the bakers who need flour for the day. Prices are written on small boards."
)
COLUMN_RIGHT = (
    "In the right column a different story is told. A traveller reaches the town at night, finds every inn "
    "full, and sleeps in a barn. In the morning he hears the noise of the market and walks toward it."
)
CLOSING = (
    "Nothing in these pages was written for scholars alone. The questions they ask are simple: what does it "
    "mean to begin again, and who can help us do it? The answer is never given in a single word."
)
TOC = [("Preface", 1), ("The Problem", 3), ("A New Beginning", 17), ("The Eighteenth Century", 29),
       ("Questions for Readers", 41), ("Notes", 55)]


def _lines(text: str, width: int) -> list[str]:
    # Break at the hyphenation point written into INTRO ("trans" + "formed").
    text = text.replace("transformed", "trans-\u0000formed")
    out = []
    for line in textwrap.wrap(text, width=width, break_on_hyphens=False):
        out.extend(line.split("\u0000"))
    return [ln.strip() for ln in out if ln.strip()]


def _para(page, text: str, y: float, *, x0=LEFT, width=86, indent=18) -> float:
    for i, line in enumerate(_lines(text, width)):
        page.insert_text((x0 + (indent if i == 0 else 0), y), line, fontsize=SIZE, fontname="tiro")
        y += LEAD
    return y + LEAD * 0.6


def _furniture(page, number: int) -> None:
    page.insert_text((LEFT, 40), "A SHORT BOOK ABOUT BEGINNINGS", fontsize=8, fontname="helv")
    page.insert_text((297, 815), str(number), fontsize=9, fontname="helv")


def build(path: Path) -> None:
    doc = pymupdf.open()
    # Page 1: title, contents, problem heading, long paragraph running off the page.
    p = doc.new_page()
    _furniture(p, 1)
    p.insert_text((LEFT, TOP + 10), "YOU MUST BE BORN AGAIN", fontsize=20, fontname="hebo")
    p.insert_text((LEFT, TOP + 50), "CONTENTS", fontsize=13, fontname="hebo")
    y = TOP + 72
    for title, num in TOC:
        p.insert_text((LEFT, y), f"{title} {'.' * (60 - len(title))} {num}", fontsize=SIZE, fontname="tiro")
        y += LEAD
    p.insert_text((LEFT, y + 20), "PROBLEM", fontsize=13, fontname="hebo")
    y += 44
    filler = ("Many letters from this period survive, and they show how often people wrote to one another about "
              "what they had read.")
    intro = _lines(INTRO, 86)
    while y + LEAD * (len(intro) + 4) < BOTTOM:
        y = _para(p, filler, y)
    # The last paragraph of page 1 stops mid-sentence and continues on page 2.
    _para(p, INTRO, BOTTOM - LEAD * (len(intro) - 1))
    # Page 2: continuation, heart, numbers, two columns.
    p = doc.new_page()
    _furniture(p, 2)
    y = TOP
    y = _para(p, CONTINUED, y, indent=0)
    y = _para(p, HEART, y + 6)
    y = _para(p, NUMBERS, y)
    p.insert_text((LEFT, y + 10), "Two Views of the Town", fontsize=12, fontname="hebo")
    y += 30
    yl = _para(p, COLUMN_LEFT, y, width=40, indent=0)
    yr = _para(p, COLUMN_RIGHT, y, x0=310, width=40, indent=0)
    y = max(yl, yr) + 10
    # Page 3: closing paragraphs.
    p = doc.new_page()
    _furniture(p, 3)
    p.insert_text((LEFT, TOP), "QUESTIONS FOR READERS", fontsize=13, fontname="hebo")
    y = TOP + 26
    y = _para(p, CLOSING, y)
    for q in ["What does the author mean by a new beginning?",
              "Why did readers in 1750 argue about books in public places?",
              "Can a person change without help from others?"]:
        p.insert_text((LEFT + 12, y), q, fontsize=SIZE, fontname="tiro")
        y += LEAD * 1.6
    y = _para(p, HEART.replace("your", "a reader's").replace("Not only", "Not only"), y + 10)
    y = _para(p, NUMBERS, y)
    y = _para(p, CLOSING, y)
    y = _para(p, COLUMN_LEFT, y)
    y = _para(p, COLUMN_RIGHT, y)
    doc.save(path, garbage=3, deflate=True)


if __name__ == "__main__":
    build(Path(sys.argv[1]))
