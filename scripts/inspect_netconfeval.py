import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from intentloom.netconfeval_loader import find_dataset_root, list_dataset_files, load_records, summarize_records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".\\data\\netconfeval")
    ap.add_argument("--limit", type=int, default=2)
    args = ap.parse_args()

    root = Path(args.root)
    dataset_root = find_dataset_root(root)
    files = list_dataset_files(dataset_root)
    print(f"dataset_root={dataset_root}")
    print(f"files={len(files)}")

    for f in files[: max(args.limit, 0)]:
        recs = load_records(f.path)
        summary = summarize_records(recs)
        print({"file": str(f.path), "kind": f.kind, "summary": summary})


if __name__ == "__main__":
    main()
