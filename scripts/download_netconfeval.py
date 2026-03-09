import argparse
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import requests


DEFAULT_ZIP = "https://codeload.github.com/RedHatResearch/conext24-NetConfEval/zip/refs/heads/main"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", type=str, default=DEFAULT_ZIP)
    ap.add_argument("--out", type=str, default=".\\data\\netconfeval")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading: {args.url}")
    with requests.get(args.url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
            tmp_path = Path(tmp.name)
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    tmp.write(chunk)

    try:
        with zipfile.ZipFile(tmp_path) as z:
            z.extractall(out_dir)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    print(f"Extracted into: {out_dir}")
    print("Next:")
    print("  python scripts/inspect_netconfeval.py --root .\\data\\netconfeval")
    print("  python scripts/build_tasks_from_netconfeval.py --root .\\data\\netconfeval")


if __name__ == "__main__":
    main()
