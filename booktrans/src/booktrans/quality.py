"""Heuristic self-review of machine translations (English -> Korean).

These checks catch recognizable failure modes of small offline NMT models: empty or
echoed output, loops, truncation, numbers that changed, negation that was lost or
invented, English words left behind, and web boilerplate that has nothing to do with the
source. They cannot prove a translation correct: a fluent sentence with the wrong meaning
passes. BookTrans therefore labels output as "machine translation, checked for common
errors", never as verified.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_HANGUL = re.compile(r"[가-힣]")
_LATIN = re.compile(r"[A-Za-z]")
_WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*")

# --- numbers -----------------------------------------------------------------
_UNITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
          "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_ORD_UNITS = ["zeroth", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
              "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth",
              "eighteenth", "nineteenth"]
_ORD_TENS = {"twentieth": 20, "thirtieth": 30, "fortieth": 40, "fiftieth": 50, "sixtieth": 60, "seventieth": 70,
             "eightieth": 80, "ninetieth": 90}
_SMALL: dict[str, int] = {w: i for i, w in enumerate(_UNITS)}
_SMALL.update({w: i for i, w in enumerate(_ORD_UNITS)})
_SMALL.update(_TENS)
_SMALL.update(_ORD_TENS)
_SCALES = {"hundred": 100, "hundredth": 100, "thousand": 1000, "thousandth": 1000, "million": 10**6, "millionth": 10**6,
           "billion": 10**9}
# Alone, these are usually not quantities ("no one", "at first", "a second chance").
_AMBIGUOUS = {"one", "first", "second", "zero", "zeroth"}
_TOKEN = re.compile(r"[A-Za-z]+|\d[\d,]*(?:\.\d+)?|[^\sA-Za-z\d]")


def _english_number_phrases(text: str) -> list[int]:
    """Values of written-out number phrases: "two hundred" -> 200, "forty thousand" -> 40000,
    "twenty-one" -> 21, "one hundred and five" -> 105, "a thousand" -> 1000."""
    words = [w.lower() for w in re.findall(r"[A-Za-z]+", text.replace("-", " "))]
    values, i = [], 0
    while i < len(words):
        w = words[i]
        starts = w in _SMALL or w in _SCALES or (w == "a" and i + 1 < len(words) and words[i + 1] in _SCALES)
        if not starts:
            i += 1
            continue
        total = current = 0
        used: list[str] = []
        while i < len(words):
            w = words[i]
            if w in _SMALL:
                current += _SMALL[w]
            elif w == "a" and not used and i + 1 < len(words) and words[i + 1] in _SCALES:
                current = 1
            elif w in _SCALES:
                scale = _SCALES[w]
                if scale == 100:
                    current = (current or 1) * 100
                else:
                    total += (current or 1) * scale
                    current = 0
            elif w == "and" and used and i + 1 < len(words) and words[i + 1] in _SMALL:
                pass
            else:
                break
            used.append(w)
            i += 1
        if len(used) == 1 and used[0] in _AMBIGUOUS:
            continue
        values.append(total + current)
    return values


def source_numbers(text: str) -> set[int]:
    nums = set()
    for m in re.finditer(r"\d[\d,]*(?:\.\d+)?", text):
        raw = m.group().replace(",", "")
        if raw.isdigit():
            nums.add(int(raw))
    nums.update(_english_number_phrases(text))
    return nums


_KO_DIGIT = {"일": 1, "이": 2, "삼": 3, "사": 4, "오": 5, "육": 6, "칠": 7, "팔": 8, "구": 9}
_KO_SMALL_UNIT = {"십": 10, "백": 100, "천": 1000}
_KO_BIG_UNIT = {"만": 10**4, "억": 10**8}
_NATIVE = {"하나": 1, "한": 1, "둘": 2, "두": 2, "셋": 3, "세": 3, "넷": 4, "네": 4, "다섯": 5, "여섯": 6, "일곱": 7,
           "여덟": 8, "아홉": 9, "열": 10, "스물": 20, "스무": 20, "서른": 30, "마흔": 40, "쉰": 50}
_COUNTER = r"(?:개|명|권|장|번|가지|해|살|시간|마리|대|채|곳|군데|쪽|주|달|사람|째|번째)"


def _sino_value(token: str) -> int | None:
    """십팔 -> 18, 이백 -> 200, 사만 -> 40000, 천오백 -> 1500."""
    total = section = digit = 0
    for ch in token:
        if ch in _KO_DIGIT:
            digit = _KO_DIGIT[ch]
        elif ch in _KO_SMALL_UNIT:
            section += (digit or 1) * _KO_SMALL_UNIT[ch]
            digit = 0
        elif ch in _KO_BIG_UNIT:
            total += (section + digit or 1) * _KO_BIG_UNIT[ch]
            section = digit = 0
        else:
            return None
    return total + section + digit


def _korean_numbers(text: str) -> set[int]:
    found: set[int] = set()
    # Arabic digits, optionally with Korean units: 4만 -> 40000, 1만 5천 -> 15000, 2백 -> 200.
    for m in re.finditer(r"(\d[\d,]*)(?:\s?([만억천백])(?:\s?(\d+)\s?([천백]))?)?", text):
        base = int(m.group(1).replace(",", ""))
        unit = m.group(2)
        if not unit:
            found.add(base)
            continue
        value = base * {"만": 10**4, "억": 10**8, "천": 1000, "백": 100}[unit]
        if m.group(3):
            value += int(m.group(3)) * {"천": 1000, "백": 100}[m.group(4)]
        found.add(value)
    # Sino-Korean numerals with a unit character (십, 백, 천, 만, 억): 십팔, 이백, 사만.
    for m in re.finditer(r"(?<![가-힣])[일이삼사오육칠팔구]*[십백천만억][일이삼사오육칠팔구십백천만억]*", text):
        token = m.group()
        if len(token) >= 2:
            value = _sino_value(token)
            if value:
                found.add(value)
    for m in re.finditer(r"(?<![가-힣])(스물|서른|마흔|쉰|열)\s?(하나|한|둘|두|셋|세|넷|네|다섯|여섯|일곱|여덟|아홉)", text):
        found.add(_NATIVE[m.group(1)] + _NATIVE[m.group(2)])
    # Native numbers only as numbers: standalone, or before a counter word ("세 개", "두 사람").
    for m in re.finditer(rf"(?<![가-힣])(하나|둘|셋|넷|다섯|여섯|일곱|여덟|아홉|열|스물|스무|서른|마흔|쉰|한|두|세|네)\s?{_COUNTER}", text):
        found.add(_NATIVE[m.group(1)])
    for m in re.finditer(r"(?<![가-힣])(하나|둘|셋|넷|다섯|여섯|일곱|여덟|아홉|스물|서른|마흔|쉰)(?![가-힣])", text):
        found.add(_NATIVE[m.group(1)])
    return found


# --- negation ------------------------------------------------------------------
_SRC_NEG = re.compile(r"\b(not|no|never|nothing|none|nobody|neither|nor|cannot|without|nowhere)\b|n['’]t\b", re.I)
_NOT_ONLY = re.compile(r"\bnot\s+(only|just|merely)\b", re.I)
# A negating predicate is required; adverbs such as 결코/절대/전혀 alone do not negate
# ("그녀는 결코 약속을 잊었다" still says she forgot).
_TGT_NEG = re.compile(r"않|(?<![가-힣])안\s|못|없|아니|말[고라아자]|마라|마십시오|모르|모른|모릅|몰라|몰랐|금지|불가능")
# Obligation ("you must be born again") must survive as obligation/command, not a statement
# of fact ("다시 태어났습니다").
_SRC_OBLIGATION = re.compile(r"\b(must|have to|has to|had to|need to|needs to|ought to|should)\b", re.I)
_TGT_OBLIGATION = re.compile(r"야\s?[하한합할했해함]|야만|해야|하라|하십시오|하세요|해라|필요|반드시|마땅|할 것")
_MALFORMED_CONTRAST = re.compile(r"[은는이가]\s?뿐(만)?\s?아니라")
_TGT_POSITIVE_CONTRAST = re.compile(r"뿐(만)?\s?아니라|만이\s?아니라|만\s?아니라|아니라")

# --- web boilerplate that small models hallucinate on short inputs ---------------
_WEB_PHRASES = {
    "관련 기사": r"related|article",
    "(주)": r"\b(inc|co|corp|corporation|company|ltd|limited|llc)\b",
    "가사 보기": r"lyric",
    "제품 소개": r"product",
    "강좌": r"course|lecture|class|lesson|seminar|tutorial",
    "계정 관리": r"account",
    "로그인": r"log\s?in|sign\s?in",
    "회원가입": r"sign\s?up|register",
    "구독": r"subscri",
    "댓글": r"comment",
    "무단 전재": r"copyright|reproduc",
    "저작권": r"copyright",
    "클릭": r"click",
    "홈페이지": r"home\s?page|website",
    "바로가기": r"shortcut|go to",
    "공유하기": r"share",
    "개인정보": r"privacy|personal",
    "이메일": r"e-?mail",
}

_TERMINAL = re.compile(r"[.!?]['\"”’)]*$")


@dataclass(frozen=True)
class QualityIssue:
    code: str
    detail: str


def _has_loop(text: str) -> bool:
    if re.search(r"(.)\1{9,}", text):
        return True
    words = text.split()
    for n in range(1, 5):
        if len(words) < n * 4:
            break
        for i in range(len(words) - n * 4 + 1):
            gram = words[i : i + n]
            if all(words[i + k * n : i + (k + 1) * n] == gram for k in range(1, 4)):
                return True
    return False


def _whole_repeat(text: str) -> bool:
    """The entire output is one phrase said two or more times ("제7장 제7장")."""
    words = re.sub(r"[.!?。]+$", "", text.strip()).split()
    for n in range(1, len(words) // 2 + 1):
        if len(words) % n == 0 and words == words[:n] * (len(words) // n):
            return True
    return False


def _english_left(source: str, target: str) -> list[str]:
    """Common English words copied into the output (proper nouns and acronyms are allowed)."""
    proper = set()
    for i, m in enumerate(_WORD.finditer(source)):
        w = m.group()
        if w[0].isupper() and i > 0 and not w.isupper():
            proper.add(w.lower())
    left = []
    for m in _WORD.finditer(target):
        w = m.group()
        lw = w.lower()
        # Acronyms ("NASA") may stay, but only if the source itself writes them that way.
        acronym = w.isupper() and len(w) <= 5 and not source.isupper() and re.search(rf"\b{re.escape(w)}\b", source)
        if len(w) < 4 or lw in proper or acronym:
            continue
        if re.search(rf"\b{re.escape(lw)}\b", source, re.I):
            left.append(w)
    return left


def check_translation(source: str, target: str) -> QualityIssue | None:
    """Return the first problem found, or None."""
    issues = review(source, target)
    return issues[0] if issues else None


def review(source: str, target: str) -> list[QualityIssue]:
    src = source.strip()
    tgt = (target or "").strip()
    src_letters = len(_LATIN.findall(src))
    if src_letters < 3:
        return []  # nothing meaningful to judge
    if not tgt:
        return [QualityIssue("empty", "translation is empty")]
    issues: list[QualityIssue] = []
    if (_has_loop(tgt) and not _has_loop(src)) or (_whole_repeat(tgt) and not _whole_repeat(src)):
        issues.append(QualityIssue("repetition", "output repeats a phrase"))
    hangul = len(_HANGUL.findall(tgt))
    latin = len(_LATIN.findall(tgt))
    src_words = len(_WORD.findall(src))
    if src_words >= 2 and hangul == 0:
        issues.append(QualityIssue("untranslated", "no Korean in output"))
    elif src_letters >= 40 and latin > 0.7 * (latin + hangul):
        issues.append(QualityIssue("mostly-latin", "output is mostly untranslated English"))
    if len(src) >= 30:
        ratio = len(tgt) / len(src)
        if ratio < 0.12:
            issues.append(QualityIssue("truncated", f"output too short (ratio {ratio:.2f})"))
        if ratio > 2.5:
            issues.append(QualityIssue("overlong", f"output too long (ratio {ratio:.2f})"))
    if src_words >= 8 and hangul < 0.8 * src_words:
        issues.append(QualityIssue("omission", f"{hangul} Hangul syllables for {src_words} English words"))

    missing = sorted(n for n in source_numbers(src) if n not in _korean_numbers(tgt))
    if missing:
        issues.append(QualityIssue("number", f"number(s) {missing} from the source are missing"))

    src_wo_not_only = _NOT_ONLY.sub(" ", src)
    tgt_wo_contrast = _TGT_POSITIVE_CONTRAST.sub(" ", tgt)
    if _SRC_NEG.search(src_wo_not_only) and not _TGT_NEG.search(tgt):
        issues.append(QualityIssue("negation-lost", "source is negated, translation is not"))
    elif _NOT_ONLY.search(src) and not _SRC_NEG.search(src_wo_not_only) and _TGT_NEG.search(tgt_wo_contrast):
        issues.append(QualityIssue("negation-added", "'not only … but' became a negation"))
    if _SRC_OBLIGATION.search(src) and not _TGT_OBLIGATION.search(tgt):
        issues.append(QualityIssue("obligation-lost", "'must/should' became a plain statement"))
    if _NOT_ONLY.search(src) and _MALFORMED_CONTRAST.search(tgt):
        issues.append(QualityIssue("contrast-malformed", "'A뿐만 아니라 B' has no A (content dropped)"))

    left = _english_left(src, tgt)
    if left:
        issues.append(QualityIssue("english-left", f"English left in output: {', '.join(left[:5])}"))

    for phrase, allowed in _WEB_PHRASES.items():
        if phrase in tgt and not re.search(allowed, src, re.I):
            issues.append(QualityIssue("web-phrase", f"'{phrase}' does not come from the source"))
            break
    return issues
