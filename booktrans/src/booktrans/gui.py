"""Tkinter desktop UI (ships with Python, so the Windows build needs no extra GUI toolkit)."""

from __future__ import annotations

import os
import queue
import threading
import traceback
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from booktrans import __version__
from booktrans.engine import ModelNotFoundError, load_engine
from booktrans.legal import ABOUT_TEXT
from booktrans.models import (
    MODEL_FILE,
    NO_DOWNLOAD_TEXT,
    DownloadCancelled,
    consent_text,
    download_model,
    install_model_file,
    locate_model,
    model_urls,
    user_data_dir,
)
from booktrans.optimizer import CancelledError, TranslationMemory
from booktrans.pipeline import JobOptions, run_job

STAGES = {"extract": "텍스트 추출", "translate": "번역", "render": "PDF 생성", "model": "모델 준비"}


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(f"BookTrans {__version__} — 영어 원서 PDF 한국어 번역")
        self.geometry("680x500")
        self.minsize(560, 420)
        self.events: queue.Queue = queue.Queue()
        self.cancel = threading.Event()
        self.worker: threading.Thread | None = None
        self.engine: ArgosEngine | None = None

        self.pdf_var = tk.StringVar()
        self.out_var = tk.StringVar()
        self.pages_var = tk.StringVar()
        self.mode_overlay = tk.BooleanVar(value=True)
        self.mode_dual = tk.BooleanVar(value=True)
        self.mode_text = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="PDF 파일을 선택하세요.")
        self._build()
        self.after(100, self._poll)

    def _build(self) -> None:
        pad = {"padx": 8, "pady": 4}
        frm = ttk.Frame(self, padding=10)
        frm.pack(fill="both", expand=True)
        frm.columnconfigure(1, weight=1)

        ttk.Label(frm, text="원서 PDF").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.pdf_var).grid(row=0, column=1, sticky="ew", **pad)
        ttk.Button(frm, text="찾아보기…", command=self._pick_pdf).grid(row=0, column=2, **pad)

        ttk.Label(frm, text="저장 폴더").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.out_var).grid(row=1, column=1, sticky="ew", **pad)
        ttk.Button(frm, text="찾아보기…", command=self._pick_out).grid(row=1, column=2, **pad)

        ttk.Label(frm, text="페이지 범위").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.pages_var).grid(row=2, column=1, sticky="ew", **pad)
        ttk.Label(frm, text="예: 1-20,25 (비우면 전체)").grid(row=2, column=2, sticky="w", **pad)

        modes = ttk.LabelFrame(frm, text="출력 형식", padding=6)
        modes.grid(row=3, column=0, columnspan=3, sticky="ew", **pad)
        ttk.Checkbutton(modes, text="번역본 (레이아웃 유지)", variable=self.mode_overlay).pack(anchor="w")
        ttk.Checkbutton(modes, text="좌우 대조본 (원문 | 번역)", variable=self.mode_dual).pack(anchor="w")
        ttk.Checkbutton(modes, text="문단별 대조 텍스트 (.md)", variable=self.mode_text).pack(anchor="w")

        btns = ttk.Frame(frm)
        btns.grid(row=4, column=0, columnspan=3, sticky="ew", **pad)
        self.start_btn = ttk.Button(btns, text="번역 시작", command=self._start)
        self.start_btn.pack(side="left")
        self.cancel_btn = ttk.Button(btns, text="취소", command=self.cancel.set, state="disabled")
        self.cancel_btn.pack(side="left", padx=6)
        ttk.Button(btns, text="정보·라이선스", command=self._about).pack(side="right")
        self.install_btn = ttk.Button(btns, text="모델 파일 설치…", command=self._install_model_file)
        self.install_btn.pack(side="right", padx=6)
        self.open_btn = ttk.Button(btns, text="결과 폴더 열기", command=self._open_out, state="disabled")
        self.open_btn.pack(side="right")

        self.bar = ttk.Progressbar(frm, mode="determinate")
        self.bar.grid(row=5, column=0, columnspan=3, sticky="ew", **pad)
        ttk.Label(frm, textvariable=self.status_var).grid(row=6, column=0, columnspan=3, sticky="w", **pad)

        self.log = tk.Text(frm, height=10, wrap="word", state="disabled")
        self.log.grid(row=7, column=0, columnspan=3, sticky="nsew", **pad)
        frm.rowconfigure(7, weight=1)

    def _pick_pdf(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("PDF", "*.pdf")])
        if path:
            self.pdf_var.set(path)
            if not self.out_var.get():
                self.out_var.set(str(Path(path).parent))

    def _pick_out(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.out_var.set(path)

    def _open_out(self) -> None:
        out = self.out_var.get()
        if out and hasattr(os, "startfile"):
            os.startfile(out)  # Windows only

    def _about(self) -> None:
        messagebox.showinfo("BookTrans 정보", ABOUT_TEXT)

    def _install_model_file(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("BookTrans 모델", "*.zip"), ("모든 파일", "*.*")])
        if not path:
            return
        try:
            installed = install_model_file(Path(path))
        except Exception as exc:
            messagebox.showerror("BookTrans", f"모델 설치 실패:\n{exc}")
            return
        self.engine = None
        self._write(f"모델 설치 완료: {installed}")

    def _model_ready(self) -> bool:
        try:
            locate_model()
            return True
        except ModelNotFoundError:
            return False

    def _write(self, line: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", line + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _start(self) -> None:
        pdf = Path(self.pdf_var.get())
        if not pdf.is_file():
            messagebox.showerror("BookTrans", "PDF 파일을 선택하세요.")
            return
        modes = tuple(
            m for m, on in (("overlay", self.mode_overlay), ("side-by-side", self.mode_dual), ("text", self.mode_text)) if on.get()
        )
        if not modes:
            messagebox.showerror("BookTrans", "출력 형식을 하나 이상 선택하세요.")
            return
        opts = JobOptions(pdf, Path(self.out_var.get() or pdf.parent), modes=modes, pages=self.pages_var.get())
        download = False
        if self.engine is None and not self._model_ready():
            urls = model_urls()
            if not urls:
                messagebox.showinfo("번역 모델 준비", NO_DOWNLOAD_TEXT)
                return
            # Never download without an explicit yes from the user.
            if not messagebox.askyesno("번역 모델 준비", consent_text(urls)):
                messagebox.showinfo(
                    "BookTrans",
                    f"모델 없이는 번역할 수 없습니다. 모델 파일({MODEL_FILE})이 있으면 "
                    "'모델 파일 설치…' 버튼으로 설치할 수 있습니다.",
                )
                return
            download = True
        self.cancel.clear()
        self.start_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.open_btn.configure(state="disabled")
        self.worker = threading.Thread(target=self._run, args=(opts, download), daemon=True)
        self.worker.start()

    def _run(self, opts: JobOptions, download: bool) -> None:
        emit = self.events.put
        try:
            if self.engine is None:
                emit(("progress", "model", 0, 1))
                if download:
                    emit(("log", "동의에 따라 번역 모델을 내려받습니다 (최초 1회)…"))
                    model_dir = download_model(
                        consent=True,
                        cancel=self.cancel,
                        progress=lambda d, t: emit(("progress", "model", d, t or max(d, 1))),
                    )
                else:
                    model_dir = locate_model()
                self.engine = load_engine(model_dir)
                emit(("log", f"모델: {self.engine.engine_id}"))
            memory = TranslationMemory(user_data_dir() / "memory.sqlite3")
            try:
                result = run_job(
                    opts, self.engine, memory,
                    progress=lambda s, d, t: emit(("progress", s, d, t)),
                    cancel=self.cancel,
                )
            finally:
                memory.close()
            for p in result.outputs:
                emit(("log", f"저장: {p}"))
            emit(("log", result.stats.summary()))
            rep = result.report
            emit(("log", f"문단 {rep.units}개: 번역 {rep.units_ok}, 일부 원문 유지 {rep.units_partial}, 원문 유지 {rep.units_unreliable}"))
            for w in result.warnings:
                emit(("log", f"주의: {w}"))
            if rep.status == "complete":
                emit(("done", "완료되었습니다. (기계 번역 — 중요한 내용은 원문과 함께 확인하세요)"))
            else:
                emit(("warn", "완료했지만 확인이 필요합니다:\n\n" + "\n".join(result.warnings)))
        except (CancelledError, DownloadCancelled):
            emit(("done", "취소되었습니다."))
        except Exception as exc:
            emit(("log", traceback.format_exc()))
            emit(("error", str(exc)))

    def _poll(self) -> None:
        try:
            while True:
                kind, *rest = self.events.get_nowait()
                if kind == "progress":
                    stage, done, total = rest
                    self.bar.configure(maximum=max(total, 1), value=done)
                    self.status_var.set(f"{STAGES.get(stage, stage)}: {done}/{total}")
                elif kind == "log":
                    self._write(rest[0])
                else:
                    self.status_var.set(rest[0])
                    self.start_btn.configure(state="normal")
                    self.cancel_btn.configure(state="disabled")
                    if kind in ("done", "warn"):
                        self.open_btn.configure(state="normal")
                    if kind == "warn":
                        self.status_var.set("완료 — 확인 필요 (로그 참고)")
                        messagebox.showwarning("BookTrans", rest[0])
                    elif kind == "error":
                        messagebox.showerror("BookTrans", rest[0])
        except queue.Empty:
            pass
        self.after(100, self._poll)


def main() -> int:
    App().mainloop()
    return 0


if __name__ == "__main__":
    main()
