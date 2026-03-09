from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class TaskMetrics:
    task_id: str
    ok: bool
    rounds: int
    first_attempt_ok: bool
    has_contract_exception: bool


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _extract_ok_and_violations(record: Dict[str, Any]) -> Tuple[Optional[bool], List[Dict[str, Any]]]:
    v = record.get("verification")
    if isinstance(v, dict):
        ok = v.get("ok")
        violations = v.get("violations")
        if isinstance(violations, list):
            return (bool(ok) if ok is not None else None), violations
        return (bool(ok) if ok is not None else None), []

    prompt = record.get("prompt")
    if isinstance(prompt, dict):
        pv = prompt.get("verification")
        if isinstance(pv, dict):
            ok = pv.get("ok")
            violations = pv.get("violations")
            if isinstance(violations, list):
                return (bool(ok) if ok is not None else None), violations
            return (bool(ok) if ok is not None else None), []

    return None, []


def _has_contract_exception_in_violations(violations: List[Dict[str, Any]]) -> bool:
    for viol in violations:
        meta = viol.get("meta")
        if isinstance(meta, dict) and meta.get("contract_exception"):
            return True
    return False


def compute_metrics_from_dir(dir_path: Path) -> List[TaskMetrics]:
    if not dir_path.exists() or not dir_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    jsonl_files = sorted(p for p in dir_path.glob("*.jsonl") if p.is_file())
    out: List[TaskMetrics] = []

    for fp in jsonl_files:
        task_id = fp.stem
        records = list(_iter_jsonl(fp))
        if not records:
            continue

        first_ok, first_violations = _extract_ok_and_violations(records[0])
        last_ok, _last_violations = _extract_ok_and_violations(records[-1])

        has_contract_exception = False
        for r in records:
            _ok, violations = _extract_ok_and_violations(r)
            if _has_contract_exception_in_violations(violations):
                has_contract_exception = True
                break

        out.append(
            TaskMetrics(
                task_id=task_id,
                ok=bool(last_ok) if last_ok is not None else False,
                rounds=len(records),
                first_attempt_ok=bool(first_ok) if first_ok is not None else False,
                has_contract_exception=has_contract_exception,
            )
        )

    return out


def _safe_avg(xs: List[int]) -> Optional[float]:
    if not xs:
        return None
    return sum(xs) / len(xs)


def aggregate(tasks: List[TaskMetrics]) -> Dict[str, Any]:
    total = len(tasks)
    ok_count = sum(1 for t in tasks if t.ok)

    success_rounds = [t.rounds for t in tasks if t.ok]
    avg_rounds_success = _safe_avg(success_rounds)

    first_attempt_ok = sum(1 for t in tasks if t.first_attempt_ok)
    first_attempt_rate = (first_attempt_ok / total) if total else 0.0

    tasks_with_contract_exception = sum(1 for t in tasks if t.has_contract_exception)
    contract_exception_task_rate = (
        tasks_with_contract_exception / total if total else 0.0
    )

    return {
        "tasks": total,
        "ok": ok_count,
        "success_rate": (ok_count / total) if total else 0.0,
        "avg_rounds_success": avg_rounds_success,
        "first_attempt_ok": first_attempt_ok,
        "first_attempt_rate": first_attempt_rate,
        "tasks_with_contract_exception": tasks_with_contract_exception,
        "contract_exception_task_rate": contract_exception_task_rate,
    }


def format_rate(n: int, total: int) -> str:
    if total <= 0:
        return "0.00 (0/0)"
    return f"{(n/total):.2f} ({n}/{total})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="Directory containing baseline (python verifier) JSONL files")
    ap.add_argument("--symai", required=True, help="Directory containing SymbolicAI JSONL files")
    ap.add_argument("--out", required=False, default=None, help="Output markdown path")
    args = ap.parse_args()

    baseline_dir = Path(args.baseline)
    symai_dir = Path(args.symai)

    baseline_tasks = compute_metrics_from_dir(baseline_dir)
    symai_tasks = compute_metrics_from_dir(symai_dir)

    b = aggregate(baseline_tasks)
    s = aggregate(symai_tasks)

    md_lines: List[str] = []
    md_lines.append("# Experiment Comparison\n")
    md_lines.append(f"- Baseline dir: `{baseline_dir.as_posix()}`")
    md_lines.append(f"- SymbolicAI dir: `{symai_dir.as_posix()}`\n")
    md_lines.append("| Metric | Python Verifier | SymbolicAI Verifier |")
    md_lines.append("| --- | ---: | ---: |")
    md_lines.append(
        f"| Final Success Rate | {format_rate(b['ok'], b['tasks'])} | {format_rate(s['ok'], s['tasks'])} |"
    )
    b_avg = b.get("avg_rounds_success")
    s_avg = s.get("avg_rounds_success")
    md_lines.append(
        f"| Avg. Iteration Count (success only) | {('-' if b_avg is None else f'{b_avg:.2f}')} | {('-' if s_avg is None else f'{s_avg:.2f}')} |"
    )
    md_lines.append(
        f"| First-attempt Recovery Rate | {format_rate(b['first_attempt_ok'], b['tasks'])} | {format_rate(s['first_attempt_ok'], s['tasks'])} |"
    )
    md_lines.append(
        f"| Logical Exception Density (task-level) | - | {format_rate(s['tasks_with_contract_exception'], s['tasks'])} |"
    )

    md = "\n".join(md_lines) + "\n"

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md, encoding="utf-8")
        print(str(out_path))
    else:
        print(md)


if __name__ == "__main__":
    main()
