"""Locating, downloading (with consent) and installing the offline translation model.

The model is never shipped inside BookTrans and never downloaded without an
explicit user decision. Every archive, downloaded or supplied by hand, must match
MODEL_SHA256 before anything is extracted.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Callable

from booktrans import __version__
from booktrans.engine import ModelNotFoundError, find_model_dir

MODEL_NAME = "quickmt-en-ko"
MODEL_FILE = "booktrans-model-quickmt-en-ko.zip"
# SHA-256 of the archive packaging/build_quickmt_model.py produces from quickmt/quickmt-en-ko at
# MODEL_REVISION. The build is reproducible; CI rebuilds it and fails if the hash differs.
MODEL_SHA256 = "2ef78d406cddf53fa621db0a5ec66a6cb8745932ec25a6d976fe94df6efda7cd"
MODEL_REVISION = "a32fa395ebc7b722a0c021e213ca5fec4065c5df"
MODEL_PAGE = "https://huggingface.co/quickmt/quickmt-en-ko"
# The package as released by the BookTrans repository's model-release workflow, which rebuilds
# it from the published model and releases it only if it matches MODEL_SHA256. Offline users
# can install the same file with --install-model / "모델 파일 설치…".
MODEL_URLS: tuple[str, ...] = (
    f"https://github.com/yudeuksong/BookTrans/releases/download/quickmt-en-ko-a32fa39/{MODEL_FILE}",
)
MAX_DOWNLOAD_BYTES = 1024 * 1024 * 1024
MAX_UNPACKED_BYTES = 1536 * 1024 * 1024
MAX_MEMBERS = 2000
# Some mirrors sit behind bot filters that reject Python's default User-Agent.
USER_AGENT = f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) BookTrans/{__version__}"

Progress = Callable[[int, int], None]


class ModelDownloadError(RuntimeError):
    pass


class ModelIntegrityError(ModelDownloadError):
    pass


class DownloadCancelled(RuntimeError):
    pass


def open_url(url: str, timeout: int = 60):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}), timeout=timeout)


def app_dir() -> Path:
    """Directory of the installed app (PyInstaller bundle) or the source checkout."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parents[2]


def user_data_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    path = Path(base) / "BookTrans"
    path.mkdir(parents=True, exist_ok=True)
    return path


def models_dir() -> Path:
    return user_data_dir() / "models"


def model_urls() -> tuple[str, ...]:
    """Download locations. BOOKTRANS_MODEL_URLS (';'-separated) exists for tests; the hash still applies."""
    override = os.environ.get("BOOKTRANS_MODEL_URLS")
    return tuple(u for u in override.split(";") if u) if override else MODEL_URLS


def _installed_model(root: Path, sha256: str, name: str = MODEL_NAME) -> Path | None:
    marker = root / name / ".complete"
    try:
        if marker.read_text(encoding="utf-8").strip() != sha256:
            return None
        return find_model_dir(root / name)
    except (OSError, ModelNotFoundError):
        return None


def locate_model() -> Path:
    """Return an installed model directory. Never downloads anything."""
    if env := os.environ.get("BOOKTRANS_MODEL"):
        return find_model_dir(Path(env))
    if not getattr(sys, "frozen", False):
        try:  # development checkout: booktrans/models/
            return find_model_dir(app_dir() / "models")
        except ModelNotFoundError:
            pass
    found = _installed_model(models_dir(), MODEL_SHA256)
    if found:
        return found
    raise ModelNotFoundError(
        f"번역 모델이 아직 준비되지 않았습니다. 모델 파일({MODEL_FILE})을 프로그램의 "
        "'모델 파일 설치…' 버튼이나 'BookTrans.exe --install-model <파일>'로 설치하세요."
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _safe_extract(archive: Path, dest: Path) -> None:
    """Extract a zip, refusing absolute/parent paths, links, and oversized content."""
    base = dest.resolve()
    with zipfile.ZipFile(archive) as zf:
        members = zf.infolist()
        if len(members) > MAX_MEMBERS:
            raise ModelIntegrityError(f"모델 압축 파일 항목이 너무 많습니다 ({len(members)})")
        if sum(m.file_size for m in members) > MAX_UNPACKED_BYTES:
            raise ModelIntegrityError("모델 압축을 풀면 크기가 허용 한도를 넘습니다")
        for m in members:
            name = m.filename.replace("\\", "/")
            parts = PurePosixPath(name).parts
            if name.startswith("/") or ".." in parts or (parts and ":" in parts[0]):
                raise ModelIntegrityError(f"모델 압축 파일에 안전하지 않은 경로가 있습니다: {m.filename}")
            if stat.S_ISLNK(m.external_attr >> 16):
                raise ModelIntegrityError(f"모델 압축 파일에 링크가 있습니다: {m.filename}")
            target = (base / name).resolve()
            if not target.is_relative_to(base):
                raise ModelIntegrityError(f"모델 압축 파일에 안전하지 않은 경로가 있습니다: {m.filename}")
        written = 0
        for m in members:
            target = base / m.filename.replace("\\", "/")
            if m.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(m) as src, target.open("wb") as out:
                # Count real bytes; header sizes can lie.
                while chunk := src.read(1 << 20):
                    written += len(chunk)
                    if written > MAX_UNPACKED_BYTES:
                        raise ModelIntegrityError("모델 압축을 풀면 크기가 허용 한도를 넘습니다")
                    out.write(chunk)


def install_model_file(
    archive: Path, *, sha256: str = MODEL_SHA256, dest_root: Path | None = None, name: str = MODEL_NAME
) -> Path:
    """Verify a model archive (zip) against the pinned hash and install it atomically."""
    dest_root = dest_root or models_dir()
    dest_root.mkdir(parents=True, exist_ok=True)
    actual = sha256_file(archive)
    if actual != sha256:
        raise ModelIntegrityError(f"모델 파일 해시가 다릅니다.\n  기대값: {sha256}\n  실제값: {actual}")
    staging = Path(tempfile.mkdtemp(prefix=".install-", dir=dest_root))
    try:
        _safe_extract(archive, staging)
        model_dir = find_model_dir(staging)
        (staging / ".complete").write_text(sha256, encoding="utf-8")
        final = dest_root / name
        if final.exists():
            shutil.rmtree(final)
        staging.rename(final)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return final / model_dir.relative_to(staging)


def _fetch(url: str, dest_root: Path, sha256: str, progress: Progress | None, cancel: threading.Event, timeout: int) -> Path:
    if not url.lower().startswith(("https://", "http://")):
        raise ModelDownloadError(f"지원하지 않는 주소입니다: {url}")
    fd, name = tempfile.mkstemp(prefix=".download-", suffix=".part", dir=dest_root)
    tmp = Path(name)
    try:
        h = hashlib.sha256()
        with os.fdopen(fd, "wb") as fh, open_url(url, timeout=timeout) as resp:
            total = int(resp.headers.get("Content-Length") or 0)
            if total > MAX_DOWNLOAD_BYTES:
                raise ModelDownloadError(f"내려받을 파일이 너무 큽니다 ({total} bytes)")
            done = 0
            while chunk := resp.read(1 << 16):
                if cancel.is_set():
                    raise DownloadCancelled("모델 내려받기를 취소했습니다.")
                done += len(chunk)
                if done > MAX_DOWNLOAD_BYTES:
                    raise ModelDownloadError("내려받는 파일이 허용 크기를 넘었습니다")
                h.update(chunk)
                fh.write(chunk)
                if progress:
                    progress(done, total)
        actual = h.hexdigest()
        if actual != sha256:
            raise ModelIntegrityError(f"{url}: 해시 불일치 (기대 {sha256}, 실제 {actual})")
        return tmp
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _retryable(exc: Exception) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in (408, 429) or exc.code >= 500
    return isinstance(exc, (urllib.error.URLError, TimeoutError, ConnectionError, OSError))


def download_model(
    *,
    consent: bool,
    progress: Progress | None = None,
    cancel: threading.Event | None = None,
    urls: tuple[str, ...] | None = None,
    sha256: str = MODEL_SHA256,
    dest_root: Path | None = None,
    name: str = MODEL_NAME,
    attempts: int = 3,
    backoff: float = 2.0,
    timeout: int = 60,
) -> Path:
    """Download, verify and install the model. Requires the caller to pass consent=True."""
    if not consent:
        raise PermissionError("사용자 동의 없이 모델을 내려받지 않습니다.")
    cancel = cancel or threading.Event()
    dest_root = dest_root or models_dir()
    dest_root.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    urls = urls or model_urls()
    if not urls:
        raise ModelDownloadError(NO_DOWNLOAD_TEXT)
    for url in urls:
        for attempt in range(1, attempts + 1):
            if cancel.is_set():
                raise DownloadCancelled("모델 내려받기를 취소했습니다.")
            try:
                archive = _fetch(url, dest_root, sha256, progress, cancel, timeout)
            except (DownloadCancelled, ModelIntegrityError) as exc:
                if isinstance(exc, DownloadCancelled):
                    raise
                errors.append(str(exc))
                break  # a wrong file will not get better by retrying this mirror
            except Exception as exc:
                errors.append(f"{url} (시도 {attempt}/{attempts}): {exc}")
                if not _retryable(exc) or attempt == attempts:
                    break
                if cancel.wait(backoff * 2 ** (attempt - 1)):
                    raise DownloadCancelled("모델 내려받기를 취소했습니다.")
                continue
            try:
                return install_model_file(archive, sha256=sha256, dest_root=dest_root, name=name)
            finally:
                archive.unlink(missing_ok=True)
    raise ModelDownloadError("모델을 내려받지 못했습니다:\n" + "\n".join(errors))


MODEL_DESCRIPTION = (
    "번역 모델: quickmt-en-ko(quickmt 프로젝트, CC BY 4.0). BookTrans에는 포함되지 않습니다.\n"
    "· 설치 후 디스크 약 370MB, 번역 중 메모리 약 0.5GB가 필요합니다.\n"
    "· 파일은 고정된 SHA-256 값과 일치할 때만 설치됩니다.\n"
    "· 저장 위치: %LOCALAPPDATA%\\BookTrans\\models"
)

NO_DOWNLOAD_TEXT = (
    "이 버전에는 번역 모델을 자동으로 내려받을 주소가 아직 없습니다.\n"
    f"모델 파일({MODEL_FILE})을 받아 '모델 파일 설치…' 버튼이나 "
    "'BookTrans.exe --install-model <파일>'로 설치하세요.\n\n" + MODEL_DESCRIPTION
)


def consent_text(urls: tuple[str, ...]) -> str:
    return (
        "BookTrans가 번역하려면 영어→한국어 번역 모델이 필요합니다.\n\n"
        + MODEL_DESCRIPTION
        + f"\n· 내려받는 곳: {urls[0]}"
        + (" (실패하면 다른 등록 주소)" if len(urls) > 1 else "")
        + "\n\n준비가 끝나면 번역은 인터넷 없이 PC 안에서만 이루어집니다. 내려받을까요?"
    )
