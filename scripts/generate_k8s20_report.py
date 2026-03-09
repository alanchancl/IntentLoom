from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_ok_rounds(r: Optional[Dict[str, Any]]) -> str:
    if not r:
        return "—"
    ok = bool(r.get("ok"))
    rounds = int(r.get("rounds", 0))
    return f"{'✅' if ok else '❌'} / {rounds}"


def _load_jsonl_last(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    last: Optional[str] = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                last = s
    if not last:
        return None
    try:
        return json.loads(last)
    except Exception:
        return None


def _extract_violation_excerpt(step: Dict[str, Any], *, max_chars: int = 360) -> str:
    prompt = step.get("prompt") if isinstance(step, dict) else None
    ver = prompt.get("verification") if isinstance(prompt, dict) else None
    vs = ver.get("violations") if isinstance(ver, dict) else None
    if not isinstance(vs, list) or not vs:
        return "(no violations recorded)"
    v0 = vs[0] if isinstance(vs[0], dict) else {}
    code = v0.get("code")
    msg = v0.get("message")
    meta = v0.get("meta") if isinstance(v0.get("meta"), dict) else {}
    ce = meta.get("contract_exception")
    parts: List[str] = []
    if code:
        parts.append(str(code))
    if msg:
        parts.append(str(msg))
    if isinstance(ce, str) and ce.strip():
        s = ce.strip().replace("\n", " ")
        if len(s) > max_chars:
            s = s[: max_chars - 3] + "..."
        parts.append(f"contract_exception: {s}")
    out = " | ".join(parts).strip()
    return out if out else "(empty violation)"


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    artifacts = repo / "artifacts"

    runs: Dict[str, Path] = {
        "rule_based": artifacts / "k8s20_loop_py_rulebased_v1" / "summary.json",
        "openrouter_py": artifacts / "k8s20_loop_openrouter_py_v1" / "summary.json",
        "openrouter_symai": artifacts / "k8s20_loop_openrouter_symai_v1" / "summary.json",
    }

    data = {k: _read_json(p) for k, p in runs.items()}
    idx = {k: {r["task_id"]: r for r in v.get("results", [])} for k, v in data.items()}
    all_ids = sorted(set().union(*[set(m.keys()) for m in idx.values()]))

    agg_rows: List[Tuple[str, Dict[str, Any]]] = []
    for k in ["rule_based", "openrouter_py", "openrouter_symai"]:
        v = data[k]
        agg_rows.append(
            (
                k,
                {
                    "tasks": v.get("tasks"),
                    "ok": v.get("ok"),
                    "success_rate": v.get("success_rate"),
                    "avg_rounds": v.get("avg_rounds"),
                },
            )
        )

    failures: Dict[str, List[str]] = {}
    for k in ["openrouter_py", "openrouter_symai"]:
        failures[k] = [
            r.get("task_id")
            for r in data[k].get("results", [])
            if isinstance(r, dict) and not r.get("ok")
        ]

    lines: List[str] = []
    lines.append("# Comparison: K8s NetworkPolicy (20 Tasks)")
    lines.append("")
    lines.append("Artifacts:")
    lines.append("")
    lines.append("- `artifacts/k8s20_loop_py_rulebased_v1/summary.json`")
    lines.append("- `artifacts/k8s20_loop_openrouter_py_v1/summary.json`")
    lines.append("- `artifacts/k8s20_loop_openrouter_symai_v1/summary.json`")
    lines.append("")

    lines.append("## Aggregate metrics")
    lines.append("")
    lines.append("| Run | Tasks | OK | Success rate | Avg rounds |")
    lines.append("| --- | --- | --- | --- | --- |")
    name_map = {
        "rule_based": "Baseline `rule_based` + python verifier",
        "openrouter_py": "OpenRouter `openai/gpt-4o-mini` + python verifier",
        "openrouter_symai": "OpenRouter `openai/gpt-4o-mini` + symbolicai verifier (witness in-loop)",
    }
    for k, row in agg_rows:
        lines.append(
            "| "
            + name_map.get(k, k)
            + f" | {row['tasks']} | {row['ok']} | {row['success_rate']} | {row['avg_rounds']} |"
        )
    lines.append("")

    lines.append("## Per-task comparison")
    lines.append("")
    lines.append("| Task ID | Baseline (ok/rounds) | OpenRouter+python (ok/rounds) | OpenRouter+symbolicai (ok/rounds) |")
    lines.append("| --- | --- | --- | --- |")
    for tid in all_ids:
        lines.append(
            f"| `{tid}` | {_fmt_ok_rounds(idx['rule_based'].get(tid))} | {_fmt_ok_rounds(idx['openrouter_py'].get(tid))} | {_fmt_ok_rounds(idx['openrouter_symai'].get(tid))} |"
        )
    lines.append("")

    lines.append("## Failures and qualitative notes")
    lines.append("")
    for k in ["openrouter_py", "openrouter_symai"]:
        fs = [x for x in failures.get(k, []) if isinstance(x, str) and x]
        if not fs:
            lines.append(f"- **{k}**: no failures")
            continue
        lines.append(f"- **{k}** failures:")
        for tid in fs:
            jsonl_path = artifacts / ("k8s20_loop_openrouter_py_v1" if k == "openrouter_py" else "k8s20_loop_openrouter_symai_v1") / f"{tid}.jsonl"
            last = _load_jsonl_last(jsonl_path)
            if last is None:
                lines.append(f"  - `{tid}`: (missing jsonl)")
                continue
            excerpt = _extract_violation_excerpt(last)
            lines.append(f"  - `{tid}`: {excerpt}")
    lines.append("")

    witness_tid = None
    if failures.get("openrouter_symai"):
        for tid in failures["openrouter_symai"]:
            if isinstance(tid, str) and tid:
                witness_tid = tid
                break

    if witness_tid:
        jsonl_path = artifacts / "k8s20_loop_openrouter_symai_v1" / f"{witness_tid}.jsonl"
        last = _load_jsonl_last(jsonl_path)
        if last is not None:
            lines.append("## Representative in-loop witness excerpt")
            lines.append("")
            lines.append(f"Task: `{witness_tid}`")
            lines.append("")
            lines.append("> " + _extract_violation_excerpt(last, max_chars=520))
            lines.append("")

    out_path = artifacts / "comparison_k8s_20tasks.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(str(out_path))


if __name__ == "__main__":
    main()
