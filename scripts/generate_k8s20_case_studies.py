from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "artifacts"

RUNS = {
    "openrouter_py": ART / "k8s20_loop_openrouter_py_v1" / "summary.json",
    "openrouter_symai": ART / "k8s20_loop_openrouter_symai_v1" / "summary.json",
}


def _read_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _iter_jsonl(p: Path) -> List[Dict[str, Any]]:
    steps: List[Dict[str, Any]] = []
    if not p.exists():
        return steps
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                steps.append(json.loads(s))
            except Exception:
                continue
    return steps


def _extract_policies(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    pols = obj.get("policies") if isinstance(obj, dict) else None
    if isinstance(pols, list):
        return [p for p in pols if isinstance(p, dict)]
    return []


def _canon_policy(p: Dict[str, Any]) -> Dict[str, Any]:
    # keep minimal keys and stable sort for pretty
    out = {
        "name": p.get("name"),
        "action": p.get("action"),
        "from": p.get("from"),
        "to": p.get("to"),
    }
    return out


def _by_name(policies: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(p.get("name")): p for p in policies if p.get("name") is not None}


def _pretty(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def _diff_policies(initial: List[Dict[str, Any]], final: List[Dict[str, Any]]) -> List[str]:
    ini = {k: _canon_policy(v) for k, v in _by_name(initial).items()}
    fin = {k: _canon_policy(v) for k, v in _by_name(final).items()}
    lines: List[str] = []
    # removed
    for k in sorted(set(ini.keys()) - set(fin.keys())):
        lines.append(f"- {k}: {_pretty(ini[k])}")
    # added
    for k in sorted(set(fin.keys()) - set(ini.keys())):
        lines.append(f"+ {k}: {_pretty(fin[k])}")
    # modified
    for k in sorted(set(ini.keys()) & set(fin.keys())):
        if _pretty(ini[k]) != _pretty(fin[k]):
            lines.append(f"~ {k} BEFORE: {_pretty(ini[k])}")
            lines.append(f"~ {k} AFTER:  {_pretty(fin[k])}")
    if not lines:
        lines.append("(no policy changes)")
    return lines


def _extract_blame_policies(step: Dict[str, Any]) -> List[str]:
    """Parse contract_exception to extract policy names allegedly enabling the violation."""
    prompt = step.get("prompt") if isinstance(step, dict) else None
    ver = prompt.get("verification") if isinstance(prompt, dict) else None
    vs = ver.get("violations") if isinstance(ver, dict) else None
    if not isinstance(vs, list) or not vs:
        return []
    v0 = vs[0] if isinstance(vs[0], dict) else {}
    meta = v0.get("meta") if isinstance(v0.get("meta"), dict) else {}
    ce = meta.get("contract_exception")
    if not isinstance(ce, str) or not ce:
        return []
    # crude extraction: policy=<name> tokens
    names = re.findall(r"policy=([A-Za-z0-9_-]+)", ce)
    # stable unique order
    seen = set()
    out: List[str] = []
    for n in names:
        if n not in seen:
            out.append(n)
            seen.add(n)
    return out


def _annotate_witness_blame(diff_lines: List[str], blame_policies: List[str]) -> List[str]:
    if not blame_policies:
        return diff_lines
    out: List[str] = []
    for line in diff_lines:
        tagged = False
        for name in blame_policies:
            token = f" {name}:"
            token2 = f" {name} BEFORE:"
            if token in line or token2 in line:
                out.append(line + " // witness-blamed")
                tagged = True
                break
        if not tagged:
            out.append(line)
    return out


def _policy_changes_between(prev: List[Dict[str, Any]], cur: List[Dict[str, Any]], limit: int = 4) -> str:
    """Return a compact summary of policy name changes between two rounds."""
    pn = set(_by_name(prev).keys())
    cn = set(_by_name(cur).keys())
    removed = sorted(pn - cn)
    added = sorted(cn - pn)
    modified: List[str] = []
    for k in sorted(pn & cn):
        if _pretty(_canon_policy(_by_name(prev)[k])) != _pretty(_canon_policy(_by_name(cur)[k])):
            modified.append(k)
    def pack(prefix: str, items: List[str]) -> List[str]:
        if not items:
            return []
        if len(items) > limit:
            head = ", ".join(items[:limit])
            return [f"{prefix} {head} (+{len(items)-limit} more)"]
        return [f"{prefix} " + ", ".join(items)]
    parts: List[str] = []
    parts += pack("-", removed)
    parts += pack("+", added)
    parts += pack("~", modified)
    return "; ".join(parts) if parts else "(no change)"


def _extract_witness(step: Dict[str, Any], *, max_chars: int = 560) -> str:
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
    return " | ".join(parts) if parts else "(empty violation)"


def build_case_studies() -> str:
    py = _read_json(RUNS["openrouter_py"])  # type: ignore[index]
    sy = _read_json(RUNS["openrouter_symai"])  # type: ignore[index]

    py_idx = {r["task_id"]: r for r in py.get("results", [])}
    sy_idx = {r["task_id"]: r for r in sy.get("results", [])}

    flips = [tid for tid, r in py_idx.items() if (not r.get("ok")) and sy_idx.get(tid, {}).get("ok")]
    flips.sort()

    lines: List[str] = []
    lines.append("## Case Study: Policy evolution on flips (Python fail → Symbolic success)")
    lines.append("")

    for tid in flips:
        lines.append(f"### `{tid}`")
        sy_jsonl = ART / "k8s20_loop_openrouter_symai_v1" / f"{tid}.jsonl"
        steps = _iter_jsonl(sy_jsonl)
        if not steps:
            lines.append("(missing symai jsonl)")
            lines.append("")
            continue

        # initial and final (successful) policies under symai
        first = steps[0]
        last_ok = None
        for st in steps[::-1]:
            ver = st.get("prompt", {}).get("verification", {})
            if ver.get("ok") is True:
                last_ok = st
                break
        if last_ok is None:
            last_ok = steps[-1]

        init_pols = _extract_policies(first.get("prompt", {}).get("candidate_cfg", {}))
        final_pols = _extract_policies(last_ok.get("prompt", {}).get("candidate_cfg", {}))

        # earliest violation witness (if any)
        vio_step = None
        for st in steps:
            if st.get("prompt", {}).get("verification", {}).get("ok") is False:
                vio_step = st
                break

        diff_lines = _diff_policies(init_pols, final_pols)
        blame = _extract_blame_policies(vio_step) if vio_step is not None else []
        diff_lines = _annotate_witness_blame(diff_lines, blame)
        lines.append("- **Policy diff (symai Round1 → final)**:")
        lines.append("")
        lines.append("```diff")
        for dl in diff_lines:
            lines.append(dl)
        lines.append("```")
        lines.append("")
        if vio_step is not None:
            lines.append("- **Witness excerpt (earliest violation)**:")
            lines.append("")
            lines.append("> " + _extract_witness(vio_step))
            lines.append("")
        else:
            lines.append("- **Witness excerpt**: (no violations recorded; converged in Round1)")
            lines.append("")

        # timeline view: Round -> Feedback snippet -> Policy changes
        lines.append("- **Timeline (round-by-round)**:")
        lines.append("")
        lines.append("| Round | Status | Feedback snippet | Policy changes |")
        lines.append("| --- | --- | --- | --- |")
        # iterate pairs to compute changes from prev -> cur
        for i, st in enumerate(steps):
            rnd = st.get("round")
            ok = st.get("prompt", {}).get("verification", {}).get("ok")
            status = "ok" if ok else "violation"
            snippet = _extract_witness(st, max_chars=180)
            cur_p = _extract_policies(st.get("prompt", {}).get("candidate_cfg", {}))
            if i == 0:
                change = "(initial)"
            else:
                prev = steps[i - 1]
                prev_p = _extract_policies(prev.get("prompt", {}).get("candidate_cfg", {}))
                change = _policy_changes_between(prev_p, cur_p)
            lines.append(f"| {rnd} | {status} | {snippet} | {change} |")
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    cs = build_case_studies()
    md = ART / "comparison_k8s_20tasks.md"
    text = md.read_text(encoding="utf-8") if md.exists() else ""
    # replace existing Case Study section if present to avoid duplication
    anchor = "## Case Study: Policy evolution on flips"
    if text:
        pos = text.find(anchor)
        if pos >= 0:
            text = text[:pos].rstrip() + "\n\n"
        if not text.endswith("\n"):
            text += "\n"
    text += cs
    md.write_text(text, encoding="utf-8")
    print(str(md))


if __name__ == "__main__":
    main()
