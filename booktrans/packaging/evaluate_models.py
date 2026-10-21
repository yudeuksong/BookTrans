"""Evidence for the model decision: license, size, CPU speed, peak RAM and meaning probes
for en->ko candidates, measured the way BookTrans would run them. CI only; nothing ships.

    python evaluate_models.py <argos-model-root> <out-dir>

* Every model gets the same input after BookTrans preprocessing (optimizer.model_input),
  runs in its own process through CTranslate2 int8 on CPU (peak RSS from /proc), and its
  output goes through BookTrans's quality review.
* Hugging Face models are downloaded at an exact revision. Before their scores count, the
  official Transformers model (same revision) translates the first sentences too; if the
  CTranslate2 path disagrees with that baseline, or the baseline itself is not Korean, the
  model is marked "evaluation-invalid" instead of being scored.
* Probes demand the content words AND the meaning marker (negation predicate, number,
  contrast), not a single keyword. "auto-accept" = probe matched and review raised nothing.
  This is an automatic screen; the sample outputs still need a person to read them.
* DEV holds the failures users reported. HELD_OUT was written separately and is not used
  to tune anything.
"""

from __future__ import annotations

import difflib
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

NEG = "않|못|없|아니|모르|모른"
NOT_ONLY_POS = "뿐(만)?\\s?아니라|만이\\s?아니라|만\\s?아니라|이자\\s|이면서"
REVERSED = "없(다|습니다|고|어|지만)|않(는다|습니다|고|아|지만)|못(한다|합니다|하지만)"
# (sentence, all of these must match, none of these may match, what it checks)
DEV = [
    ("YOU MUST BE BORN AGAIN", ["거듭나|다시 태어나|새로 태어나", "야|하라|십시오"], ["찾아|AGAIN|again"], "command: must be born again"),
    ("CONTENTS", ["목차|차례"], ["관련 기사"], "table of contents"),
    ("PROBLEM", ["문제"], ["계정"], "problem"),
    ("The eighteenth century was a time of great change.", ["18|십팔", "세기", "변화"], [], "18th century"),
    ("Not only can your heart be changed, but your whole life can be renewed.",
     ["마음", "삶|인생|생애", NOT_ONLY_POS], [REVERSED], "not only A but B stays positive"),
    ("Readers did not accept every argument they found.", ["독자", "주장|논증|논거|논쟁", NEG], [], "negation kept"),
    ("In 1750 there were twenty-one printing houses in the city.", ["1750", "21|스물\\s?한|이십일", "인쇄"], [], "numbers"),
]
HELD_OUT = [
    ("THE NARROW GATE", ["좁은", "문"], [], "all-caps title"),
    ("INTRODUCTION", ["서론|소개|머리말|들어가|도입"], [], "heading"),
    ("Chapter Seven", ["7|칠|일곱"], ["17|70"], "chapter number"),
    ("In 1848 the city had nearly forty thousand inhabitants.", ["1848", "40[,.]?000|4만|사만", "주민|인구|사람"],
     ["4천|4,000(?!0)"], "forty thousand"),
    ("Two hundred years later, the church was rebuilt.", ["200|이백", "교회", "재건|다시 (지|세)|복원"], [], "two hundred"),
    ("The nineteenth chapter explains why the letters were burned.", ["19|십구|열아홉", "편지", "태|소각"], [], "ordinal"),
    ("She never forgot the promise she had made.", ["약속", "잊지\\s?않|잊지\\s?못|잊은\\s?적이?\\s?없"], [], "never forgot"),
    ("Grace is not earned; it is received as a gift.", ["은혜|은총", "선물", NEG], [], "negation"),
    ("Without faith, the work would have been impossible.", ["믿음|신앙", "없|불가능"], [], "without"),
    ("It is not only a duty but also a joy.", ["의무", "기쁨", NOT_ONLY_POS], ["[은는]\\s?뿐", REVERSED], "not only duty but joy"),
    ("Please read the instructions carefully before you begin the experiment.", ["지침|설명|지시", "실험", "읽"], [], "imperative"),
    ("The committee will publish its final report next spring.", ["위원회", "보고서", "봄"], [], "plain prose"),
    ("Photosynthesis converts light energy into chemical energy stored in glucose.",
     ["광합성", "빛|광", "화학", "포도당|글루코스"], ["가벼운|Photosynthesis"], "terminology"),
    ("He had never seen the ocean before, and the sound of the waves frightened him.",
     ["바다", "파도", "본 적이?\\s?없|보지 못|처음", "두려|무섭|겁|놀라"], [], "never + content"),
    ("Despite the rain, thousands of people gathered in the square to hear her speak.",
     ["비", "수천", "광장", "그녀"], [], "content kept"),
    ("The letter arrived three days late, but nobody was surprised.", ["편지", "3|사흘|삼|세\\s?날|세\\s?일", "늦", NEG + "|아무도"], [], "number + negation"),
    ("Unless we act now, the river will flood the village.", ["강", "마을", "않으면|않는다면|없으면|하지 않"], [], "unless"),
    ("Neither the king nor the bishop accepted the offer.", ["왕", "주교", NEG + "|거절"], [], "neither nor"),
]
CANDIDATES = [
    "facebook/m2m100_418M",
    "facebook/m2m100_1.2B",
    "facebook/mbart-large-50-one-to-many-mmt",
    "Helsinki-NLP/opus-mt-tc-big-en-ko",
    "Helsinki-NLP/opus-mt-en-mul",
    "facebook/nllb-200-distilled-600M",
]
PERMISSIVE = {"mit", "apache-2.0", "cc-by-4.0", "cc0-1.0", "bsd-3-clause"}
MAX_WEIGHTS = 5_500_000_000
BASELINE_N = 6
UA = {"User-Agent": "BookTrans-evidence"}
HANGUL = re.compile(r"[가-힣]")


def hf(path: str):
    with urllib.request.urlopen(urllib.request.Request(f"https://huggingface.co/api/{path}", headers=UA), timeout=60) as r:
        return json.load(r)


def model_info(mid: str) -> dict:
    info = hf(f"models/{mid}?blobs=true")
    tags = info.get("tags", [])
    lic = (info.get("cardData") or {}).get("license") or next((t.split(":", 1)[1] for t in tags if t.startswith("license:")), "UNKNOWN")
    weights = max(
        sum(s.get("size") or 0 for s in info.get("siblings", []) if s["rfilename"].endswith(ext))
        for ext in (".safetensors", ".bin")
    )
    return {"license": str(lic), "weights_bytes": weights, "revision": info.get("sha")}


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def peak_rss_mb() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / 1024
    return 0.0


def items() -> list[tuple]:
    return [("dev", *d) for d in DEV] + [("held-out", *h) for h in HELD_OUT]


# ---------------------------------------------------------------- language/prefix handling
def hf_setup(mid: str, tok) -> dict:
    """How the official model card says to request Korean output."""
    cfg = {"src_prefix": None, "target_prefix": None, "forced_bos": None, "notes": []}
    name = mid.lower()
    if "m2m100" in name:
        tok.src_lang = "en"
        cfg["forced_bos"] = tok.get_lang_id("ko")
    elif "nllb" in name:
        tok.src_lang = "eng_Latn"
        cfg["forced_bos"] = tok.convert_tokens_to_ids("kor_Hang")
    elif "mbart" in name:
        tok.src_lang = "en_XX"
        cfg["forced_bos"] = tok.convert_tokens_to_ids("ko_KR")
    else:  # Marian: multi-target models need a >>lang<< token at the start of the source
        langs = getattr(tok, "supported_language_codes", None) or [t for t in tok.get_vocab() if re.fullmatch(r">>\w+<<", t)]
        kor = sorted(t for t in langs if "kor" in t)
        cfg["notes"].append(f"language tokens: {len(langs)} (kor: {kor[:4]})")
        if kor:
            cfg["src_prefix"] = kor[0]
    if cfg["forced_bos"] is not None:
        cfg["target_prefix"] = [tok.convert_ids_to_tokens(cfg["forced_bos"])]
    sep = getattr(tok, "separate_vocabs", None)
    spm = [k for k in ("spm_source", "spm_target") if hasattr(tok, k)]
    cfg["notes"].append(f"tokenizer {type(tok).__name__}, separate_vocabs={sep}, spm={spm}")
    return cfg


# ---------------------------------------------------------------- workers (own process each)
def worker_ct2(spec: dict) -> dict:
    from booktrans.optimizer import model_input
    from booktrans.quality import review

    texts = [model_input(i[1]) for i in items()]
    start = time.perf_counter()
    if spec["kind"] == "argos":
        from booktrans.engine import ArgosEngine, CAREFUL, FAST, find_model_dir

        eng = ArgosEngine(find_model_dir(Path(spec["path"])))
        load = time.perf_counter() - start
        start = time.perf_counter()
        outs = eng.translate_batch(texts, careful=spec.get("careful", False))
        notes = [eng.engine_id, f"decoding {'CAREFUL' if spec.get('careful') else 'FAST'}: {CAREFUL if spec.get('careful') else FAST}"]
    else:
        import ctranslate2
        import transformers

        tok = transformers.AutoTokenizer.from_pretrained(spec["path"])
        cfg = hf_setup(spec["id"], tok)
        translator = ctranslate2.Translator(str(Path(spec["path"]) / "ct2"), device="cpu", compute_type="int8")
        load = time.perf_counter() - start
        start = time.perf_counter()
        outs = []
        for t in texts:
            src = f"{cfg['src_prefix']} {t}" if cfg["src_prefix"] else t
            tokens = tok.convert_ids_to_tokens(tok.encode(src))
            tp = cfg["target_prefix"]
            res = translator.translate_batch([tokens], target_prefix=[tp] if tp else None, beam_size=4)
            hyp = res[0].hypotheses[0]
            if tp and hyp[: len(tp)] == tp:
                hyp = hyp[len(tp):]
            outs.append(tok.decode(tok.convert_tokens_to_ids(hyp), skip_special_tokens=True))
        notes = cfg["notes"]
    elapsed = time.perf_counter() - start
    rows = []
    for (group, src, req, forbid, what), text, out in zip(items(), texts, outs):
        issues = [i.code for i in review(text, out)]
        probe = all(re.search(r, out) for r in req) and not any(re.search(f, out) for f in forbid)
        rows.append({"group": group, "source": src, "output": out, "probe": what, "probe_pass": probe,
                     "review_issues": issues, "auto_accept": probe and not issues})
    return {"rows": rows, "load_s": round(load, 2), "sec_per_sentence": round(elapsed / len(texts), 3),
            "peak_rss_mb": round(peak_rss_mb()), "notes": notes}


def worker_baseline(spec: dict) -> dict:
    """Official Transformers inference (same local snapshot) on the first sentences."""
    import torch
    import transformers
    from booktrans.optimizer import model_input

    tok = transformers.AutoTokenizer.from_pretrained(spec["path"])
    cfg = hf_setup(spec["id"], tok)
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(spec["path"])
    model.eval()
    outs = []
    for _, src, *_ in items()[:BASELINE_N]:
        text = model_input(src)
        text = f"{cfg['src_prefix']} {text}" if cfg["src_prefix"] else text
        enc = tok(text, return_tensors="pt")
        kwargs = {"num_beams": 4, "max_new_tokens": 128}
        if cfg["forced_bos"] is not None:
            kwargs["forced_bos_token_id"] = cfg["forced_bos"]
        with torch.no_grad():
            gen = model.generate(**enc, **kwargs)
        outs.append(tok.batch_decode(gen, skip_special_tokens=True)[0])
    return {"baseline": outs, "notes": cfg["notes"]}


def run(kind: str, spec: dict) -> dict:
    proc = subprocess.run([sys.executable, __file__, f"--{kind}", json.dumps(spec)], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-3000:])
    return json.loads(proc.stdout.strip().splitlines()[-1])


# ---------------------------------------------------------------- parent
def validate_against_baseline(entry: dict) -> None:
    ct2 = [r["output"] for r in entry["rows"][:BASELINE_N]]
    base = entry["baseline"]
    sims = [difflib.SequenceMatcher(None, a, b).ratio() for a, b in zip(ct2, base)]
    korean = sum(1 for b in base if len(HANGUL.findall(b)) >= 2) / max(len(base), 1)
    entry["baseline_similarity"] = round(sum(sims) / len(sims), 2) if sims else 0.0
    entry["baseline_korean_share"] = round(korean, 2)
    if korean < 0.5:
        entry["status"] = "evaluation-invalid: official Transformers baseline is not Korean (language/prefix setup unverified)"
    elif entry["baseline_similarity"] < 0.6:
        entry["status"] = "evaluation-invalid: CTranslate2 output disagrees with the official baseline (conversion/tokenizer path)"
    else:
        entry["status"] = "evaluated"


def summarize(r: dict) -> dict:
    out = {}
    for group in ("dev", "held-out"):
        rows = [x for x in r["rows"] if x["group"] == group]
        out[group] = (f"probe {sum(x['probe_pass'] for x in rows)}/{len(rows)}, "
                      f"auto-accept {sum(x['auto_accept'] for x in rows)}/{len(rows)}")
    out["flagged"] = sum(1 for x in r["rows"] if x["review_issues"])
    return out


def main(argos_root: str, out_dir: str) -> int:
    from booktrans.engine import find_model_dir

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    work = out / "work"
    results: list[dict] = []
    errors = 0

    argos_dir = find_model_dir(Path(argos_root))
    for careful in (False, True):
        entry = {"id": f"argos translate-en_ko 1.1 ({'careful' if careful else 'default'} decoding)",
                 "license": "not stated for the model package", "size_bytes": dir_size(argos_dir), "status": "evaluated"}
        try:
            entry.update(run("ct2", {"kind": "argos", "path": str(argos_dir), "careful": careful}))
        except Exception as exc:
            entry["status"], errors = f"error: {exc}", errors + 1
        results.append(entry)

    from huggingface_hub import snapshot_download

    for mid in CANDIDATES:
        entry = {"id": mid}
        try:
            entry.update(model_info(mid))
            if entry["license"].lower() not in PERMISSIVE:
                entry["status"] = f"skipped: license {entry['license']} does not allow unrestricted use"
            elif entry["weights_bytes"] > MAX_WEIGHTS:
                entry["status"] = "skipped: weights larger than 5.5 GB"
            else:
                local = work / mid.replace("/", "__")
                snapshot_download(mid, revision=entry["revision"], local_dir=str(local),
                                  allow_patterns=["*.json", "*.model", "*.spm", "*.txt", "*.safetensors", "sentencepiece*", "*.bpe"],
                                  ignore_patterns=["*onnx*", "*tf_model*", "*flax*", "*rust*"])
                if not any(local.glob("*.safetensors")):
                    snapshot_download(mid, revision=entry["revision"], local_dir=str(local), allow_patterns=["pytorch_model*.bin"])
                subprocess.run(["ct2-transformers-converter", "--model", str(local), "--output_dir", str(local / "ct2"),
                                "--quantization", "int8", "--force"], check=True, capture_output=True, text=True)
                entry["size_bytes"] = dir_size(local / "ct2")
                spec = {"kind": "hf", "id": mid, "path": str(local)}
                entry.update(run("ct2", spec))
                entry.update(run("baseline", spec))
                validate_against_baseline(entry)
                shutil.rmtree(local, ignore_errors=True)
        except Exception as exc:
            entry["status"] = f"error: {type(exc).__name__}: {str(exc)[-1500:]}"
            errors += 1
        results.append(entry)
        print(json.dumps({k: v for k, v in entry.items() if k not in ("rows",)}, ensure_ascii=False), flush=True)

    (out / "model-candidates.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    mb = lambda b: f"{b / 1e6:.0f} MB" if b else "-"
    lines = ["# en→ko offline model comparison (automatic screen, not a human review)", "",
             "CTranslate2 int8 on the CI CPU (GitHub ubuntu-latest), beam 4, BookTrans preprocessing and review applied.",
             "probe = required content and meaning markers matched; auto-accept = probe matched and BookTrans review raised nothing.", "",
             "| model | license | revision | status | int8 size | load s | s/sentence | peak RAM | dev | held-out | review-flagged | baseline sim |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        s = summarize(r) if "rows" in r else {"dev": "-", "held-out": "-", "flagged": "-"}
        lines.append(f"| {r['id']} | {r.get('license')} | {str(r.get('revision', ''))[:10]} | {r.get('status', '')[:110]} | "
                     f"{mb(r.get('size_bytes', 0))} | {r.get('load_s', '-')} | {r.get('sec_per_sentence', '-')} | "
                     f"{r.get('peak_rss_mb', '-')} MB | {s['dev']} | {s['held-out']} | {s['flagged']} | {r.get('baseline_similarity', '-')} |")
    lines += ["", "## Baseline check (official Transformers vs CTranslate2, first sentences)", ""]
    for r in results:
        if "baseline" in r:
            lines.append(f"**{r['id']}** — {'; '.join(r.get('notes', []))}")
            for i, b in enumerate(r["baseline"]):
                lines.append(f"- {items()[i][1]}\n  - transformers: {b}\n  - ctranslate2: {r['rows'][i]['output']}")
            lines.append("")
    lines += ["## Outputs (✓ auto-accept, ✗ probe failed, ⚑ flagged by BookTrans review)", ""]
    with_rows = [r for r in results if "rows" in r]
    for i, (group, src, *_rest, what) in enumerate(items()):
        lines.append(f"**[{group}] {src}** — {what}")
        for r in with_rows:
            x = r["rows"][i]
            mark = ("✓" if x["auto_accept"] else ("✗" if not x["probe_pass"] else "·")) + ("⚑" if x["review_issues"] else "")
            tag = "" if r.get("status") == "evaluated" else " [evaluation-invalid]"
            lines.append(f"- {mark} {r['id']}{tag}: {x['output']}" + (f"  ({', '.join(x['review_issues'])})" if x["review_issues"] else ""))
        lines.append("")
    (out / "model-candidates.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 1 if errors else 0


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] in ("--ct2", "--baseline"):
        fn = worker_ct2 if sys.argv[1] == "--ct2" else worker_baseline
        print(json.dumps(fn(json.loads(sys.argv[2])), ensure_ascii=False))
        sys.exit(0)
    sys.exit(main(sys.argv[1], sys.argv[2]))
