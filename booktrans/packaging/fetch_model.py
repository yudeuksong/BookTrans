"""Prepare a model for CI/development (running this script is the consent).

    python fetch_model.py --archive <booktrans-model-...zip> <dest>   install the BookTrans model
                                                                    package (pinned MODEL_SHA256)
    python fetch_model.py --argos <dest>                              download the Argos en->ko 1.1
                                                                    package (model comparison only)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from booktrans.models import MODEL_SHA256, download_model, install_model_file  # noqa: E402

# The Argos package BookTrans 0.1.0-0.1.1 used; kept as a comparison point in the model evidence.
ARGOS_NAME = "translate-en_ko-1_1"
ARGOS_SHA256 = "e03d8e65e6d44525ec5808c3409fcf8728c76c2c76925372b6d3dc3278de17fc"
ARGOS_URLS = (
    f"https://argos-net.com/v1/{ARGOS_NAME}.argosmodel",
    "https://ipfs.io/ipfs/QmWecr5i4tJNnokusm97rTUyQtUqNNPufGF7ake1hJVu6G",
)


def main() -> None:
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--archive", type=Path)
    group.add_argument("--argos", action="store_true")
    ap.add_argument("dest", type=Path)
    args = ap.parse_args()
    if args.archive:
        model_dir = install_model_file(args.archive, dest_root=args.dest)
        print(f"verified sha256 {MODEL_SHA256}")
    else:
        model_dir = download_model(consent=True, urls=ARGOS_URLS, sha256=ARGOS_SHA256, dest_root=args.dest, name=ARGOS_NAME)
        print(f"verified sha256 {ARGOS_SHA256}")
    print(f"model ready: {model_dir}")


if __name__ == "__main__":
    main()
