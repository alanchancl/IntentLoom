from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class DatasetFile:
    path: Path
    kind: str  # json/jsonl/other


def find_dataset_root(netconfeval_root: Path) -> Path:
    """Try to locate the 'dataset' directory inside a downloaded NetConfEval repo."""
    direct_candidates = [
        netconfeval_root / "datasets",
        netconfeval_root / "dataset",
    ]
    for c in direct_candidates:
        if c.is_dir() and list_dataset_files(c):
            return c
    for c in direct_candidates:
        if c.is_dir():
            return c

    # common when extracting zip: <root>/conext24-NetConfEval-main/dataset
    dir_names = ("datasets", "dataset")
    for name in dir_names:
        for p in netconfeval_root.rglob(name):
            if not p.is_dir():
                continue
            if list_dataset_files(p):
                return p
            if p.parent.name.endswith("NetConfEval-main") or p.parent.name.endswith("NetConfEval"):
                return p
            if (p / ".." / "README.md").exists():
                return p

    raise FileNotFoundError(f"Could not find dataset directory under: {netconfeval_root}")


def list_dataset_files(dataset_root: Path) -> List[DatasetFile]:
    files: List[DatasetFile] = []
    for p in dataset_root.rglob("*"):
        if not p.is_file():
            continue
        suf = p.suffix.lower()
        if suf == ".json":
            files.append(DatasetFile(path=p, kind="json"))
        elif suf in {".jsonl", ".ndjson"}:
            files.append(DatasetFile(path=p, kind="jsonl"))
    return sorted(files, key=lambda x: str(x.path))


def load_records(p: Path) -> List[Dict[str, Any]]:
    if p.suffix.lower() == ".json":
        obj = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            return [obj]
        raise ValueError(f"Unsupported JSON root type: {type(obj)}")

    if p.suffix.lower() in {".jsonl", ".ndjson"}:
        recs: List[Dict[str, Any]] = []
        for line in p.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s:
                continue
            rec = json.loads(s)
            if isinstance(rec, dict):
                recs.append(rec)
        return recs

    raise ValueError(f"Unsupported file type: {p}")


def summarize_records(records: Sequence[Dict[str, Any]], max_keys: int = 30) -> Dict[str, Any]:
    keys = {}
    for r in records:
        for k in r.keys():
            keys[k] = keys.get(k, 0) + 1

    top = sorted(keys.items(), key=lambda kv: (-kv[1], kv[0]))[:max_keys]
    return {"count": len(records), "top_keys": top}
