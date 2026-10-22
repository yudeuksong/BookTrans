"""PyInstaller entry point: GUI when double-clicked, CLI when given arguments."""

import faulthandler
import multiprocessing
import os
import sys
import traceback


def _setup_cli_output() -> None:
    """A windowed (no-console) exe has no stdout; send CLI output somewhere visible.

    BOOKTRANS_LOG=<file> redirects output to a file (used by CI); otherwise the calling
    terminal's console is borrowed when there is one.
    """
    log_path = os.environ.get("BOOKTRANS_LOG")
    if log_path:
        stream = open(log_path, "w", encoding="utf-8", buffering=1)
        sys.stdout = sys.stderr = stream
        faulthandler.enable(stream)
        # A hang would otherwise be silent; dump every thread's stack and exit instead.
        faulthandler.dump_traceback_later(int(os.environ.get("BOOKTRANS_HANG_SECONDS", "300")), exit=True, file=stream)
        return
    if sys.stdout is not None:
        return
    stream = None
    if sys.platform == "win32":
        import ctypes

        if ctypes.windll.kernel32.AttachConsole(-1):  # ATTACH_PARENT_PROCESS
            stream = open("CONOUT$", "w", encoding="utf-8")
    sys.stdout = sys.stderr = stream or open(os.devnull, "w", encoding="utf-8")


def _run_cli() -> int:
    _setup_cli_output()
    from booktrans.cli import main

    try:
        return main()
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1
    except Exception:
        # Report instead of letting the windowed bootloader open a blocking error dialog.
        text = traceback.format_exc()
        print(text, file=sys.stderr, flush=True)
        try:
            from booktrans.models import user_data_dir

            (user_data_dir() / "last-error.log").write_text(text, encoding="utf-8")
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    multiprocessing.freeze_support()
    if len(sys.argv) > 1:
        code = _run_cli()
        sys.stdout.flush()
        sys.exit(code)
    from booktrans.gui import main

    sys.exit(main())
