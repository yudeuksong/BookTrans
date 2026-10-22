from __future__ import annotations

import hashlib
import http.server
import io
import threading
import zipfile
from pathlib import Path

import pytest

from booktrans import models
from booktrans.models import (
    DownloadCancelled,
    ModelDownloadError,
    ModelIntegrityError,
    download_model,
    install_model_file,
)


def make_model_zip(extra: dict[str, bytes] | None = None, *, symlink: str | None = None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("en_ko/model/model.bin", b"weights")
        zf.writestr("en_ko/sentencepiece.model", b"sp")
        zf.writestr("en_ko/metadata.json", b'{"from_code": "en", "to_code": "ko"}')
        for name, data in (extra or {}).items():
            zf.writestr(name, data)
        if symlink:
            info = zipfile.ZipInfo(symlink)
            info.external_attr = (0o120777 << 16)
            zf.writestr(info, "/etc/passwd")
    return buf.getvalue()


GOOD = make_model_zip()
GOOD_SHA = hashlib.sha256(GOOD).hexdigest()


class Server:
    """Local HTTP server with a scripted list of responses per path."""

    def __init__(self):
        self.script: dict[str, list] = {}
        self.hits: list[str] = []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                outer.hits.append(self.path)
                queue = outer.script.get(self.path, [404])
                item = queue.pop(0) if len(queue) > 1 else queue[0]
                if isinstance(item, int):
                    self.send_error(item)
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(item)))
                self.end_headers()
                self.wfile.write(item)

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}{path}"

    def close(self):
        self.httpd.shutdown()


@pytest.fixture
def server():
    s = Server()
    yield s
    s.close()


def leftovers(root: Path) -> list[str]:
    return sorted(p.name for p in root.iterdir() if p.name.startswith("."))


def test_download_requires_consent(tmp_path, server):
    server.script["/m"] = [GOOD]
    with pytest.raises(PermissionError):
        download_model(consent=False, urls=(server.url("/m"),), sha256=GOOD_SHA, dest_root=tmp_path)
    assert server.hits == []  # nothing was fetched


def test_download_verifies_and_installs(tmp_path, server):
    server.script["/m"] = [GOOD]
    path = download_model(consent=True, urls=(server.url("/m"),), sha256=GOOD_SHA, dest_root=tmp_path)
    assert (path / "model" / "model.bin").read_bytes() == b"weights"
    assert (tmp_path / models.MODEL_NAME / ".complete").read_text() == GOOD_SHA
    assert models._installed_model(tmp_path, GOOD_SHA) == path
    assert models._installed_model(tmp_path, "0" * 64) is None  # marker must match the pinned hash
    assert leftovers(tmp_path) == []


def test_hash_mismatch_is_rejected_and_falls_through(tmp_path, server):
    server.script["/bad"] = [make_model_zip({"x": b"tampered"})]
    server.script["/good"] = [GOOD]
    path = download_model(
        consent=True, urls=(server.url("/bad"), server.url("/good")), sha256=GOOD_SHA, dest_root=tmp_path, backoff=0
    )
    assert path.exists()
    assert server.hits.count("/bad") == 1  # a wrong file is not retried on the same mirror
    assert leftovers(tmp_path) == []


def test_hash_mismatch_everywhere_installs_nothing(tmp_path, server):
    server.script["/bad"] = [make_model_zip({"x": b"tampered"})]
    with pytest.raises(ModelDownloadError, match="해시"):
        download_model(consent=True, urls=(server.url("/bad"),), sha256=GOOD_SHA, dest_root=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_retries_server_errors(tmp_path, server):
    server.script["/m"] = [503, 500, GOOD]
    download_model(consent=True, urls=(server.url("/m"),), sha256=GOOD_SHA, dest_root=tmp_path, backoff=0)
    assert server.hits.count("/m") == 3


def test_does_not_retry_404(tmp_path, server):
    with pytest.raises(ModelDownloadError):
        download_model(consent=True, urls=(server.url("/missing"),), sha256=GOOD_SHA, dest_root=tmp_path, backoff=0)
    assert server.hits.count("/missing") == 1


def test_cancel_mid_download_cleans_up(tmp_path, server):
    big = GOOD + b"\0" * (1 << 20)
    server.script["/m"] = [big]
    cancel = threading.Event()

    def progress(done, total):
        cancel.set()  # cancel after the first chunk

    with pytest.raises(DownloadCancelled):
        download_model(
            consent=True, urls=(server.url("/m"),), sha256=GOOD_SHA, dest_root=tmp_path, cancel=cancel, progress=progress
        )
    assert list(tmp_path.iterdir()) == []


def test_cancel_before_start(tmp_path, server):
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(DownloadCancelled):
        download_model(consent=True, urls=(server.url("/m"),), sha256=GOOD_SHA, dest_root=tmp_path, cancel=cancel)
    assert server.hits == []


def test_rejects_non_http_urls(tmp_path):
    with pytest.raises(ModelDownloadError):
        download_model(consent=True, urls=("file:///etc/passwd",), sha256=GOOD_SHA, dest_root=tmp_path)


@pytest.mark.parametrize(
    "archive",
    [
        make_model_zip({"../escape.txt": b"x"}),
        make_model_zip({"/abs.txt": b"x"}),
        make_model_zip({"C:/win.txt": b"x"}),
        make_model_zip(symlink="en_ko/link"),
    ],
    ids=["parent", "absolute", "drive", "symlink"],
)
def test_install_rejects_unsafe_archives(tmp_path, archive):
    f = tmp_path / "m.argosmodel"
    f.write_bytes(archive)
    dest = tmp_path / "models"
    with pytest.raises(ModelIntegrityError):
        install_model_file(f, sha256=hashlib.sha256(archive).hexdigest(), dest_root=dest)
    assert not (tmp_path / "escape.txt").exists()
    assert list(dest.iterdir()) == []


def test_install_rejects_oversized(tmp_path, monkeypatch):
    archive = make_model_zip({"big.bin": b"\0" * 5000})
    f = tmp_path / "m.argosmodel"
    f.write_bytes(archive)
    monkeypatch.setattr(models, "MAX_UNPACKED_BYTES", 1000)
    with pytest.raises(ModelIntegrityError):
        install_model_file(f, sha256=hashlib.sha256(archive).hexdigest(), dest_root=tmp_path / "models")


def test_install_model_file_checks_pinned_hash(tmp_path):
    f = tmp_path / "m.argosmodel"
    f.write_bytes(GOOD)
    with pytest.raises(ModelIntegrityError):
        install_model_file(f, dest_root=tmp_path / "models")  # default = official hash


def test_locate_model_never_downloads(tmp_path, monkeypatch):
    monkeypatch.delenv("BOOKTRANS_MODEL", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(models, "app_dir", lambda: tmp_path / "nowhere")
    called = []
    monkeypatch.setattr(models, "download_model", lambda **kw: called.append(kw))
    with pytest.raises(models.ModelNotFoundError):
        models.locate_model()
    assert called == []


def test_cli_refuses_download_without_consent(monkeypatch, capsys):
    from booktrans import cli

    called = []
    monkeypatch.setenv("BOOKTRANS_MODEL_URLS", "https://example.invalid/m.zip")
    monkeypatch.setattr(cli, "download_model", lambda **kw: called.append(kw))
    monkeypatch.setattr("sys.stdin", None)
    assert cli.main(["--download-model"]) == cli.EXIT_NO_CONSENT
    assert called == []


def test_no_download_location_means_install_from_file(monkeypatch, tmp_path, capsys):
    """No hosting is approved: downloading must say so and point to the model file, not ask."""
    from booktrans import cli

    monkeypatch.delenv("BOOKTRANS_MODEL_URLS", raising=False)
    monkeypatch.setattr(models, "MODEL_URLS", ())
    called = []
    monkeypatch.setattr(cli, "download_model", lambda **kw: called.append(kw))
    assert cli.main(["--download-model", "--yes"]) == cli.EXIT_DOWNLOAD_FAILED
    err = capsys.readouterr().err
    assert models.MODEL_FILE in err and "--install-model" in err
    assert called == []
    with pytest.raises(ModelDownloadError, match="--install-model"):
        download_model(consent=True, dest_root=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_official_location_is_https_and_names_the_pinned_file():
    assert models.MODEL_URLS and all(u.startswith("https://") and u.endswith("/" + models.MODEL_FILE) for u in models.MODEL_URLS)
    assert models.MODEL_REVISION[:7] in models.MODEL_URLS[0]


def test_pinned_model_identity():
    assert len(models.MODEL_SHA256) == 64 and len(models.MODEL_REVISION) == 40
    assert models.MODEL_FILE.endswith(".zip") and models.MODEL_NAME in models.MODEL_FILE


def test_cli_reports_missing_model(tmp_path, monkeypatch, sample_pdf):
    from booktrans import cli

    monkeypatch.delenv("BOOKTRANS_MODEL", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(models, "app_dir", lambda: tmp_path / "nowhere")
    assert cli.main([str(sample_pdf)]) == cli.EXIT_NO_MODEL
