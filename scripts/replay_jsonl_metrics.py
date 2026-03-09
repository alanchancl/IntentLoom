from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from intentloom.compiler import compile_intent
from intentloom.intent import IntentNode


def _intent_from_dict(d: Dict[str, Any]) -> IntentNode:
    op = d.get("op")
    if op == "PRIMITIVE":
        return IntentNode(
            op="PRIMITIVE",
            value={"name": d.get("name"), "params": d.get("params", {})},
        )
    children = tuple(_intent_from_dict(x) for x in (d.get("children") or []))
    return IntentNode(op=op, children=children)


def _cfg_satisfies_intent(intent_dict: Dict[str, Any], cfg: Dict[str, Any]) -> bool:
    intent = _intent_from_dict(intent_dict)
    compiled = compile_intent(intent)
    for c in compiled.constraints:
        if c.check(cfg):
            return False
    return True


@dataclass(frozen=True)
class TaskReplay:
    task_id: str
    rounds: int
    final_ok: bool
    first_attempt_ok: bool
    first_ok_round: Optional[int]


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def replay_task(jsonl_path: str) -> TaskReplay:
    task_id = os.path.splitext(os.path.basename(jsonl_path))[0]

    oks: List[bool] = []
    for obj in _iter_jsonl(jsonl_path):
        intent = obj["prompt"]["intent"]
        cfg = obj["candidate_cfg"]
        oks.append(_cfg_satisfies_intent(intent, cfg))

    first_ok_round: Optional[int] = None
    for i, ok in enumerate(oks, start=1):
        if ok:
            first_ok_round = i
            break

    return TaskReplay(
        task_id=task_id,
        rounds=len(oks),
        final_ok=(first_ok_round is not None),
        first_attempt_ok=(oks[0] if oks else False),
        first_ok_round=first_ok_round,
    )


def replay_folder(folder: str) -> Dict[str, TaskReplay]:
    out: Dict[str, TaskReplay] = {}
    for f in sorted(glob.glob(os.path.join(folder, "*.jsonl"))):
        tr = replay_task(f)
        out[tr.task_id] = tr
    return out


def load_summary_ok(summary_path: str) -> Dict[str, bool]:
    try:
        s = json.load(open(summary_path, "r", encoding="utf-8"))
    except Exception:
        return {}
    return {r["task_id"]: bool(r.get("ok")) for r in s.get("results", [])}


def metrics(task_replays: Dict[str, TaskReplay]) -> Dict[str, Any]:
    n = len(task_replays)
    succ = sum(1 for t in task_replays.values() if t.final_ok)
    first = sum(1 for t in task_replays.values() if t.first_attempt_ok)
    succ_rounds = [t.first_ok_round for t in task_replays.values() if t.final_ok and t.first_ok_round]
    avg_succ_rounds = (sum(succ_rounds) / len(succ_rounds)) if succ_rounds else 0.0
    return {
        "n": n,
        "success": succ,
        "success_rate": (succ / n if n else 0.0),
        "first_attempt": first,
        "first_attempt_rate": (first / n if n else 0.0),
        "avg_success_round": avg_succ_rounds,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+", help="Artifact folders containing *.jsonl")
    ap.add_argument(
        "--compare_summary",
        action="store_true",
        help="Compare replayed final_ok vs folder/summary.json if present",
    )
    args = ap.parse_args()

    for folder in args.folders:
        replays = replay_folder(folder)
        m = metrics(replays)
        print(
            folder,
            "n=", m["n"],
            "success=", f"{m['success']}/{m['n']} ({m['success_rate']:.2f})",
            "first_attempt=", f"{m['first_attempt']}/{m['n']} ({m['first_attempt_rate']:.2f})",
            "avg_success_round=", f"{m['avg_success_round']:.2f}",
        )

        if args.compare_summary:
            summary_path = os.path.join(folder, "summary.json")
            old_ok = load_summary_ok(summary_path)
            if old_ok:
                changed = sorted(
                    [
                        tid
                        for tid, tr in replays.items()
                        if tid in old_ok and old_ok[tid] != tr.final_ok
                    ]
                )
                print("changed_vs_summary=", len(changed))
                if changed:
                    print("changed_tasks=", ", ".join(changed[:50]))


if __name__ == "__main__":
    main()
