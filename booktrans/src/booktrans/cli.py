"""Command-line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from booktrans import __version__
from booktrans.engine import ModelNotFoundError, load_engine
from booktrans.legal import ABOUT_TEXT
from booktrans.models import (
    NO_DOWNLOAD_TEXT,
    ModelDownloadError,
    consent_text,
    download_model,
    install_model_file,
    locate_model,
    model_urls,
    user_data_dir,
)
from booktrans.optimizer import TranslationMemory
from booktrans.pipeline import OUTPUT_MODES, JobOptions, run_job

EXIT_NO_MODEL = 3
EXIT_NO_CONSENT = 4
EXIT_DOWNLOAD_FAILED = 5
EXIT_WITH_WARNINGS = 7  # output written, but some text was not translated reliably
EXIT_NOTHING_TRANSLATED = 8


def _ask_consent(assume_yes: bool, urls: tuple[str, ...]) -> bool:
    print(consent_text(urls))
    if assume_yes:
        print("--yes 옵션으로 동의했습니다.")
        return True
    stdin = sys.stdin
    if stdin is None or not stdin.isatty():
        print("동의를 물을 수 없는 환경입니다. 동의하려면 --yes 를 함께 지정하세요.")
        return False
    return input("내려받으려면 y 를 입력하세요 [y/N]: ").strip().lower() in ("y", "yes")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="booktrans", description="영어 PDF를 한국어로 오프라인 번역합니다.")
    parser.add_argument("pdf", nargs="?", type=Path, help="번역할 PDF 파일")
    parser.add_argument("-o", "--output-dir", type=Path, help="결과 폴더 (기본: PDF와 같은 폴더)")
    parser.add_argument("--mode", action="append", choices=OUTPUT_MODES, help="출력 형식 (여러 번 지정 가능)")
    parser.add_argument("--pages", default="", help="페이지 범위, 예: 1-20,25")
    parser.add_argument("--download-model", action="store_true", help="번역 모델을 내려받습니다 (동의 확인)")
    parser.add_argument("--yes", action="store_true", help="--download-model 의 동의 질문에 미리 동의합니다")
    parser.add_argument("--install-model", type=Path, metavar="FILE", help="받아 둔 모델 파일(.zip)을 SHA-256 확인 후 설치합니다")
    parser.add_argument("--about", action="store_true", help="저작권·라이선스 고지를 표시합니다")
    parser.add_argument("--version", action="version", version=f"booktrans {__version__}")
    args = parser.parse_args(argv)

    if args.about:
        print(ABOUT_TEXT)
        return 0
    if args.install_model:
        try:
            print(f"모델 설치 완료: {install_model_file(args.install_model)}")
        except (OSError, ModelDownloadError) as exc:
            print(f"모델 설치 실패: {exc}", file=sys.stderr)
            return EXIT_DOWNLOAD_FAILED
    if args.download_model:
        urls = model_urls()
        if not urls:
            print(NO_DOWNLOAD_TEXT, file=sys.stderr)
            return EXIT_DOWNLOAD_FAILED
        if not _ask_consent(args.yes, urls):
            print("모델을 내려받지 않았습니다.")
            return EXIT_NO_CONSENT
        try:
            path = download_model(consent=True, urls=urls, progress=_progress_bytes)
        except ModelDownloadError as exc:
            print(f"\n{exc}", file=sys.stderr)
            return EXIT_DOWNLOAD_FAILED
        print(f"\n모델 설치 완료: {path}")
    if not args.pdf:
        if args.download_model or args.install_model:
            return 0
        parser.error("PDF 파일을 지정하세요.")
    if not args.pdf.is_file():
        parser.error(f"파일이 없습니다: {args.pdf}")

    try:
        model_dir = locate_model()
    except ModelNotFoundError as exc:
        print(exc, file=sys.stderr)
        return EXIT_NO_MODEL
    engine = load_engine(model_dir)
    memory = TranslationMemory(user_data_dir() / "memory.sqlite3")
    opts = JobOptions(
        input_pdf=args.pdf,
        output_dir=args.output_dir or args.pdf.parent,
        modes=tuple(args.mode or ("overlay", "side-by-side")),
        pages=args.pages,
    )

    def progress(stage: str, done: int, total: int) -> None:
        print(f"\r[{stage}] {done}/{total}", end="", file=sys.stderr, flush=True)

    try:
        result = run_job(opts, engine, memory, progress=progress)
    finally:
        memory.close()
    print(file=sys.stderr)
    for path in result.outputs:
        print(f"저장: {path}")
    print(result.stats.summary())
    rep = result.report
    print(
        f"상태: {rep.status} — 문단 {rep.units}개 중 번역 {rep.units_ok}, 일부 원문 유지 {rep.units_partial}, "
        f"원문 유지 {rep.units_unreliable}; 문장 {rep.sentences}개 중 검사 실패 {rep.sentences_unreliable}"
    )
    for w in result.warnings:
        print(f"주의: {w}")
    print("이 결과는 기계 번역입니다. 자동 검사는 흔한 오류만 찾을 뿐 뜻이 맞는지 보장하지 않습니다.")
    if rep.status == "nothing-translated":
        return EXIT_NOTHING_TRANSLATED
    return EXIT_WITH_WARNINGS if rep.status != "complete" else 0


def _progress_bytes(done: int, total: int) -> None:
    mb = done / 1_048_576
    text = f"{mb:.1f}/{total / 1_048_576:.1f} MB" if total else f"{mb:.1f} MB"
    print(f"\r내려받는 중 {text}", end="", file=sys.stderr, flush=True)


if __name__ == "__main__":
    sys.exit(main())
