import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List


def _template_from_task_id(task_id: str) -> str:
    parts = str(task_id).split("_")
    if len(parts) <= 2:
        return str(task_id)
    return "_".join(parts[2:])


def _read_tasks(path: Path) -> List[Dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise ValueError(f"Expected a JSON list in {path}")
    out: List[Dict[str, Any]] = []
    for t in obj:
        if isinstance(t, dict):
            out.append(t)
    return out


def _write_tasks(path: Path, tasks: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--size", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--balance-by-template",
        dest="balance_by_template",
        action="store_true",
    )
    ap.add_argument(
        "--balance_by_template",
        dest="balance_by_template",
        action="store_true",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    tasks = _read_tasks(in_path)

    n = int(args.size)
    if n <= 0:
        raise ValueError("--size must be > 0")
    if n > len(tasks):
        raise ValueError(f"--size {n} exceeds available tasks {len(tasks)}")

    rng = random.Random(int(args.seed))

    if not bool(args.balance_by_template):
        idxs = list(range(len(tasks)))
        rng.shuffle(idxs)
        picked = [tasks[i] for i in idxs[:n]]
        _write_tasks(out_path, picked)
        print(f"Wrote {len(picked)} tasks -> {out_path}")
        return

    by_tpl: Dict[str, List[Dict[str, Any]]] = {}
    for t in tasks:
        tid = t.get("id")
        tpl = _template_from_task_id(str(tid or ""))
        by_tpl.setdefault(tpl, []).append(t)

    templates = sorted(by_tpl.keys())
    for tpl in templates:
        rng.shuffle(by_tpl[tpl])

    base = n // len(templates)
    rem = n % len(templates)

    quota: Dict[str, int] = {}
    for i, tpl in enumerate(templates):
        quota[tpl] = base + (1 if i < rem else 0)

    picked: List[Dict[str, Any]] = []
    remaining: List[Dict[str, Any]] = []

    for tpl in templates:
        want = int(quota[tpl])
        src = by_tpl.get(tpl, [])
        take = min(want, len(src))
        picked.extend(src[:take])
        remaining.extend(src[take:])

    if len(picked) < n:
        rng.shuffle(remaining)
        picked.extend(remaining[: (n - len(picked))])

    picked = picked[:n]
    picked_sorted = sorted(picked, key=lambda t: str(t.get("id") or ""))

    counts: Dict[str, int] = {}
    for t in picked_sorted:
        tpl = _template_from_task_id(str(t.get("id") or ""))
        counts[tpl] = counts.get(tpl, 0) + 1

    _write_tasks(out_path, picked_sorted)
    print(f"Wrote {len(picked_sorted)} tasks -> {out_path}")
    for tpl in sorted(counts.keys()):
        print(f"{tpl}: {counts[tpl]}")


if __name__ == "__main__":
    main()
