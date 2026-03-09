from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as st
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class ExpSpec:
    sem: str
    mode: str
    root: Path


def _mean_sd(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    if len(xs) == 1:
        return float(xs[0]), 0.0
    return st.mean(xs), st.pstdev(xs)


def _logsumexp(log_xs: List[float]) -> float:
    if not log_xs:
        return float("-inf")
    m = max(log_xs)
    if m == float("-inf"):
        return m
    s = 0.0
    for x in log_xs:
        s += math.exp(x - m)
    return m + math.log(s)


def _mcnemar_exact_pvalue(b: int, c: int) -> float:
    # Two-sided exact McNemar under Binomial(n=b+c, p=0.5)
    n = int(b + c)
    if n <= 0:
        return 1.0
    k = int(min(b, c))
    log2 = math.log(2.0)

    def _logpmf(i: int) -> float:
        # log(C(n,i) / 2^n)
        return math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * log2

    lo_logs = [_logpmf(i) for i in range(0, k + 1)]
    hi_logs = [_logpmf(i) for i in range(n - k, n + 1)]
    lo = math.exp(_logsumexp(lo_logs))
    hi = math.exp(_logsumexp(hi_logs))
    p = 2.0 * min(lo, hi)
    return min(1.0, max(0.0, p))


def _exact_sign_permutation_pvalue(diffs: List[float]) -> float:
    # Exact two-sided permutation test on mean(diffs) under random sign flips.
    n = len(diffs)
    if n <= 0:
        return 1.0
    obs = float(st.mean(diffs))
    vals: List[float] = []
    for mask in range(1 << n):
        xs = []
        for i, d in enumerate(diffs):
            xs.append(d if ((mask >> i) & 1) else -d)
        vals.append(float(st.mean(xs)))
    ge = sum(1 for v in vals if abs(v) >= abs(obs))
    return float(ge) / float(len(vals))


def _quantile(sorted_xs: List[float], q: float) -> float:
    if not sorted_xs:
        return 0.0
    if q <= 0:
        return float(sorted_xs[0])
    if q >= 1:
        return float(sorted_xs[-1])
    idx = int(q * (len(sorted_xs) - 1))
    return float(sorted_xs[idx])


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: List[Dict[str, Any]], *, fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            if not isinstance(r, dict):
                continue
            w.writerow({k: r.get(k) for k in fieldnames})


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def _template_from_task_id(task_id: str) -> str:
    parts = str(task_id).split("_")
    if len(parts) <= 2:
        return str(task_id)
    return "_".join(parts[2:])


def _policy_index(policies: Any) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(policies, list):
        return out
    for p in policies:
        if not isinstance(p, dict):
            continue
        name = p.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        out[name.strip()] = p
    return out


def _json_canon(x: Any) -> str:
    try:
        return json.dumps(x, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(x)


def _policy_diff(before: Any, after: Any) -> Dict[str, Any]:
    b = _policy_index(before)
    a = _policy_index(after)
    names = sorted(set(b.keys()) | set(a.keys()))
    added: List[str] = []
    removed: List[str] = []
    changed: List[str] = []
    unchanged: List[str] = []
    changes: Dict[str, Dict[str, Any]] = {}
    for n in names:
        if n not in b:
            added.append(n)
            changes[n] = {"before": None, "after": a.get(n)}
            continue
        if n not in a:
            removed.append(n)
            changes[n] = {"before": b.get(n), "after": None}
            continue
        if _json_canon(b.get(n)) == _json_canon(a.get(n)):
            unchanged.append(n)
            continue
        changed.append(n)
        changes[n] = {"before": b.get(n), "after": a.get(n)}
    return {
        "added": added,
        "removed": removed,
        "changed": changed,
        "unchanged": unchanged,
        "changes": changes,
    }


def _policy_edit_distance(before: Any, after: Any) -> int:
    d = _policy_diff(before, after)
    return int(len(d.get("added") or [])) + int(len(d.get("removed") or [])) + int(len(d.get("changed") or []))


def _max_contract_exception_len_from_verification(ver: Any) -> int:
    if not isinstance(ver, dict):
        return 0
    vs = ver.get("violations")
    if not isinstance(vs, list):
        return 0
    best = 0
    for v in vs:
        if not isinstance(v, dict):
            continue
        meta = v.get("meta")
        if not isinstance(meta, dict):
            continue
        ce = meta.get("contract_exception")
        if isinstance(ce, str):
            best = max(best, len(ce))
    return best


def _max_violation_message_len_from_verification(ver: Any) -> int:
    if not isinstance(ver, dict):
        return 0
    vs = ver.get("violations")
    if not isinstance(vs, list):
        return 0
    best = 0
    for v in vs:
        if not isinstance(v, dict):
            continue
        msg = v.get("message")
        if isinstance(msg, str):
            best = max(best, len(msg))
    return best


def _extract_best_violation_message(ver: Any) -> str:
    if not isinstance(ver, dict):
        return ""
    vs = ver.get("violations")
    if not isinstance(vs, list):
        return ""
    best: str = ""
    for v in vs:
        if not isinstance(v, dict):
            continue
        msg = v.get("message")
        if isinstance(msg, str) and len(msg) > len(best):
            best = msg
    return best


def _verification_json_len_from_prompt(prompt: Any) -> int:
    if not isinstance(prompt, dict):
        return 0
    ver = prompt.get("verification")
    try:
        return len(json.dumps(ver, ensure_ascii=False))
    except Exception:
        return 0


def _extract_best_contract_exception_text(ver: Any) -> str:
    if not isinstance(ver, dict):
        return ""
    vs = ver.get("violations")
    if not isinstance(vs, list):
        return ""
    best: str = ""
    for v in vs:
        if not isinstance(v, dict):
            continue
        meta = v.get("meta")
        if not isinstance(meta, dict):
            continue
        ce = meta.get("contract_exception")
        if isinstance(ce, str) and len(ce) > len(best):
            best = ce
    return best


def _blame_size_from_prompt(prompt: Any) -> int:
    if not isinstance(prompt, dict):
        return 0
    b = prompt.get("blame_set")
    if not isinstance(b, dict):
        return 0
    n = 0
    bl = b.get("blamed_links")
    bp = b.get("blamed_policies")
    if isinstance(bl, list):
        n += len(bl)
    if isinstance(bp, list):
        n += len(bp)
    return n


def _svg_escape(s: Any) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _svg_grouped_bars(
    *,
    title: str,
    y_label: str,
    groups: List[str],
    series: List[str],
    values: Dict[Tuple[str, str], float],
    errors: Optional[Dict[Tuple[str, str], float]] = None,
    y_max: Optional[float] = None,
    out_path: Path,
) -> None:
    W = 900
    H = 420
    ML = 70
    MR = 20
    MT = 45
    MB = 70

    if y_max is None:
        vmax = 0.0
        for mode in groups:
            for s in series:
                vmax = max(vmax, float(values.get((mode, s), 0.0)))
        y_max = 1.0 if vmax <= 1.0 else vmax
        if y_max <= 0:
            y_max = 1.0

    def xpix(x: float) -> float:
        return ML + x * (W - ML - MR)

    def ypix(v: float) -> float:
        v = max(0.0, min(float(y_max), float(v)))
        return MT + (1.0 - (v / float(y_max))) * (H - MT - MB)

    nG = max(1, len(groups))
    nS = max(1, len(series))
    group_span = (W - ML - MR) / nG
    gap = 0.22 * group_span
    bar_w = (group_span - gap) / nS

    palette = {
        "A": "#4C78A8",
        "B": "#F58518",
    }

    lines: List[str] = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">')
    lines.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')
    lines.append(f'<text x="{W/2:.1f}" y="{MT-18}" text-anchor="middle" font-size="16" font-family="Arial">{_svg_escape(title)}</text>')

    y0 = ypix(0.0)
    lines.append(f'<line x1="{ML}" y1="{y0:.1f}" x2="{W-MR}" y2="{y0:.1f}" stroke="#000" stroke-width="1"/>')
    lines.append(f'<line x1="{ML}" y1="{MT}" x2="{ML}" y2="{H-MB}" stroke="#000" stroke-width="1"/>')

    ticks = 5
    for i in range(ticks + 1):
        v = (float(y_max) * i) / ticks
        y = ypix(v)
        lines.append(f'<line x1="{ML-5}" y1="{y:.1f}" x2="{ML}" y2="{y:.1f}" stroke="#000" stroke-width="1"/>')
        lines.append(f'<text x="{ML-8}" y="{y+4:.1f}" text-anchor="end" font-size="12" font-family="Arial">{v:.2f}</text>')

    lines.append(f'<text x="{16}" y="{H/2:.1f}" transform="rotate(-90 {16} {H/2:.1f})" text-anchor="middle" font-size="12" font-family="Arial">{_svg_escape(y_label)}</text>')

    for gi, g in enumerate(groups):
        gx0 = ML + gi * group_span
        gx_center = gx0 + group_span / 2
        lines.append(f'<text x="{gx_center:.1f}" y="{H-MB+28}" text-anchor="middle" font-size="12" font-family="Arial">{_svg_escape(g)}</text>')

        for si, s in enumerate(series):
            v = float(values.get((g, s), 0.0))
            x = gx0 + gap / 2 + si * bar_w
            y = ypix(v)
            h = y0 - y
            color = palette.get(s, "#999")
            lines.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w-3:.1f}" height="{h:.1f}" fill="{color}"/>'
            )

            if errors is not None:
                e = float(errors.get((g, s), 0.0))
                y_hi = ypix(min(float(y_max), v + e))
                y_lo = ypix(max(0.0, v - e))
                cx = x + (bar_w - 3) / 2
                lines.append(f'<line x1="{cx:.1f}" y1="{y_hi:.1f}" x2="{cx:.1f}" y2="{y_lo:.1f}" stroke="#000" stroke-width="1"/>')
                lines.append(f'<line x1="{cx-6:.1f}" y1="{y_hi:.1f}" x2="{cx+6:.1f}" y2="{y_hi:.1f}" stroke="#000" stroke-width="1"/>')
                lines.append(f'<line x1="{cx-6:.1f}" y1="{y_lo:.1f}" x2="{cx+6:.1f}" y2="{y_lo:.1f}" stroke="#000" stroke-width="1"/>')

    lx = W - MR - 160
    ly = MT + 5
    lines.append(f'<rect x="{lx}" y="{ly}" width="150" height="52" fill="white" stroke="#ccc"/>')
    for i, s in enumerate(series):
        color = palette.get(s, "#999")
        lines.append(f'<rect x="{lx+10}" y="{ly+10+i*18}" width="14" height="14" fill="{color}"/>')
        lines.append(f'<text x="{lx+30}" y="{ly+22+i*18}" font-size="12" font-family="Arial">{_svg_escape(s)}</text>')

    lines.append("</svg>")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _svg_cdf(
    *,
    title: str,
    x_label: str,
    y_label: str,
    series: List[str],
    xs: List[int],
    cdf_values: Dict[str, List[float]],
    out_path: Path,
) -> None:
    W = 900
    H = 420
    ML = 70
    MR = 20
    MT = 45
    MB = 70

    x_min = min(xs) if xs else 0
    x_max = max(xs) if xs else 1
    if x_max == x_min:
        x_max = x_min + 1

    palette = {
        "A": "#4C78A8",
        "B": "#F58518",
    }

    def xpix(x: float) -> float:
        return ML + (x - x_min) * (W - ML - MR) / (x_max - x_min)

    def ypix(v: float) -> float:
        v = max(0.0, min(1.0, float(v)))
        return MT + (1.0 - v) * (H - MT - MB)

    lines: List[str] = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">')
    lines.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')
    lines.append(f'<text x="{W/2:.1f}" y="{MT-18}" text-anchor="middle" font-size="16" font-family="Arial">{_svg_escape(title)}</text>')

    lines.append(f'<line x1="{ML}" y1="{ypix(0):.1f}" x2="{W-MR}" y2="{ypix(0):.1f}" stroke="#000" stroke-width="1"/>')
    lines.append(f'<line x1="{ML}" y1="{MT}" x2="{ML}" y2="{H-MB}" stroke="#000" stroke-width="1"/>')

    for i in range(6):
        v = i / 5
        y = ypix(v)
        lines.append(f'<line x1="{ML-5}" y1="{y:.1f}" x2="{ML}" y2="{y:.1f}" stroke="#000" stroke-width="1"/>')
        lines.append(f'<text x="{ML-8}" y="{y+4:.1f}" text-anchor="end" font-size="12" font-family="Arial">{v:.1f}</text>')

    for x in range(x_min, x_max + 1):
        xp = xpix(x)
        lines.append(f'<line x1="{xp:.1f}" y1="{ypix(0):.1f}" x2="{xp:.1f}" y2="{ypix(0)+5:.1f}" stroke="#000" stroke-width="1"/>')
        lines.append(f'<text x="{xp:.1f}" y="{H-MB+28}" text-anchor="middle" font-size="12" font-family="Arial">{x}</text>')

    lines.append(f'<text x="{W/2:.1f}" y="{H-20}" text-anchor="middle" font-size="12" font-family="Arial">{_svg_escape(x_label)}</text>')
    lines.append(f'<text x="{16}" y="{H/2:.1f}" transform="rotate(-90 {16} {H/2:.1f})" text-anchor="middle" font-size="12" font-family="Arial">{_svg_escape(y_label)}</text>')

    for s in series:
        ys = cdf_values.get(s) or []
        if len(ys) != len(xs):
            continue
        pts = " ".join(f"{xpix(x):.1f},{ypix(y):.1f}" for x, y in zip(xs, ys))
        color = palette.get(s, "#999")
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{pts}"/>')

    lx = W - MR - 160
    ly = MT + 5
    lines.append(f'<rect x="{lx}" y="{ly}" width="150" height="52" fill="white" stroke="#ccc"/>')
    for i, s in enumerate(series):
        color = palette.get(s, "#999")
        y = ly + 10 + i * 18
        lines.append(f'<line x1="{lx+10}" y1="{y+7}" x2="{lx+24}" y2="{y+7}" stroke="{color}" stroke-width="2"/>')
        lines.append(f'<text x="{lx+30}" y="{y+11}" font-size="12" font-family="Arial">{_svg_escape(s)}</text>')

    lines.append("</svg>")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=str, default="data/k8s_policy_100_ports_tasks.json")
    ap.add_argument("--out", type=str, default="artifacts/ports100_multiseed_report")
    ap.add_argument("--seeds", type=str, default="1,2,3,4,5")
    ap.add_argument(
        "--sems",
        type=str,
        default="A,B",
        help="Comma-separated semantics labels to include (default: A,B). Use A,B,C when Option C runs exist.",
    )
    ap.add_argument(
        "--suffix_by_sem",
        type=str,
        default="",
        help=(
            "Optional per-semantics artifact directory suffix mapping. "
            "Format: 'A=_v2,B=,C=_fix'. "
            "Each semantics uses root artifacts/ports100_option_{sem}_{mode}<suffix>. "
            "Example: '--sems A,B,C --suffix_by_sem C=_fix' reads Option C from *_fix directories while A/B stay unchanged."
        ),
    )
    ap.add_argument("--case_studies_per_mode", type=int, default=5)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / args.out
    _ensure_dir(out_dir)

    seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]

    sems = [s.strip().upper() for s in str(args.sems).split(",") if s.strip()]
    modes = ["generic", "raw_witness", "blame_only"]

    suffix_by_sem: Dict[str, str] = {}
    if isinstance(args.suffix_by_sem, str) and args.suffix_by_sem.strip():
        for part in str(args.suffix_by_sem).split(","):
            p = part.strip()
            if not p:
                continue
            if "=" not in p:
                suffix_by_sem[p.strip().upper()] = ""
                continue
            k, v = p.split("=", 1)
            suffix_by_sem[k.strip().upper()] = v.strip()

    def _artifact_root(sem: str, mode: str) -> Path:
        s = str(sem).strip().upper()
        suffix = suffix_by_sem.get(s, "")
        return repo_root / f"artifacts/ports100_option_{s.lower()}_{mode}{suffix}"

    specs: List[ExpSpec] = []
    for sem in sems:
        for mode in modes:
            specs.append(ExpSpec(sem, mode, _artifact_root(sem, mode)))

    tasks_path = repo_root / args.tasks
    tasks_list = _read_json(tasks_path)
    task_by_id: Dict[str, Dict[str, Any]] = {}
    if isinstance(tasks_list, list):
        for t in tasks_list:
            if not isinstance(t, dict):
                continue
            tid = t.get("id")
            if isinstance(tid, str) and tid:
                task_by_id[tid] = t

    per_seed_metrics: List[Dict[str, Any]] = []
    pooled_task_results: Dict[Tuple[str, str], List[Tuple[int, str, bool, int]]] = {}

    for spec in specs:
        for sd in seeds:
            summ = spec.root / f"seed_{sd}" / "summary.json"
            d = _read_json(summ)
            ok = int(d.get("ok", 0))
            n = int(d.get("tasks", 0))
            sr = float(d.get("success_rate", 0.0))
            ar = float(d.get("avg_rounds", 0.0))
            per_seed_metrics.append(
                {
                    "sem": spec.sem,
                    "mode": spec.mode,
                    "seed": sd,
                    "tasks": n,
                    "ok": ok,
                    "success_rate": sr,
                    "avg_rounds": ar,
                }
            )

            key = (spec.sem, spec.mode)
            pooled_task_results.setdefault(key, [])
            for r in d.get("results", []):
                if not isinstance(r, dict):
                    continue
                tid = str(r.get("task_id") or "").strip()
                if not tid:
                    continue
                pooled_task_results[key].append((sd, tid, bool(r.get("ok")), int(r.get("rounds") or 0)))

    out_json = out_dir / "report_data.json"
    out_json.write_text(
        json.dumps(
            {
                "seeds": seeds,
                "per_seed_metrics": per_seed_metrics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    id2init_cfg: Dict[str, Dict[str, Any]] = {}
    for tid, t in task_by_id.items():
        init_cfg = t.get("initial_cfg") if isinstance(t, dict) and isinstance(t.get("initial_cfg"), dict) else {}
        id2init_cfg[str(tid)] = init_cfg

    edit_by_spec_seed: Dict[Tuple[str, str], List[float]] = {}
    for spec in specs:
        key = (spec.sem, spec.mode)
        edit_by_spec_seed[key] = []
        for sd in seeds:
            summ = spec.root / f"seed_{sd}" / "summary.json"
            d = _read_json(summ)
            rs = d.get("results", []) if isinstance(d, dict) else []
            eds: List[int] = []
            for r in rs:
                if not isinstance(r, dict):
                    continue
                tid = str(r.get("task_id") or "")
                if not tid:
                    continue
                init_cfg = id2init_cfg.get(tid) if isinstance(id2init_cfg.get(tid), dict) else {}
                before = init_cfg.get("policies") if isinstance(init_cfg, dict) else None
                final_cfg = r.get("final_cfg") if isinstance(r.get("final_cfg"), dict) else {}
                after = final_cfg.get("policies") if isinstance(final_cfg, dict) else None
                eds.append(_policy_edit_distance(before, after))
            edit_by_spec_seed[key].append(float(st.mean(eds)) if eds else 0.0)

    summary_rows: List[Dict[str, Any]] = []
    for spec in specs:
        srs = [r["success_rate"] for r in per_seed_metrics if r["sem"] == spec.sem and r["mode"] == spec.mode]
        ars = [r["avg_rounds"] for r in per_seed_metrics if r["sem"] == spec.sem and r["mode"] == spec.mode]
        okc = [r["ok"] for r in per_seed_metrics if r["sem"] == spec.sem and r["mode"] == spec.mode]
        ok_mean, ok_sd = _mean_sd([float(x) for x in okc])
        sr_mean = ok_mean / 100.0 if ok_mean else 0.0
        sr_sd = ok_sd / 100.0 if ok_sd else 0.0
        rounds_mean, rounds_sd = _mean_sd([float(x) for x in ars])
        edit_by_seed = edit_by_spec_seed.get((spec.sem, spec.mode)) or []
        edit_mean, edit_sd = _mean_sd([float(x) for x in edit_by_seed])
        summary_rows.append(
            {
                "sem": spec.sem,
                "mode": spec.mode,
                "ok_mean": ok_mean,
                "ok_sd": ok_sd,
                "sr_mean": sr_mean,
                "sr_sd": sr_sd,
                "avg_rounds_mean": rounds_mean,
                "avg_rounds_sd": rounds_sd,
                "avg_policy_edit_mean": edit_mean,
                "avg_policy_edit_sd": edit_sd,
            }
        )

    _write_csv(out_dir / "summary_multiseed.csv", summary_rows, fieldnames=list(summary_rows[0].keys()) if summary_rows else [])

    # Paired significance (success: McNemar exact; rounds/edit: exact sign-permutation over seeds)
    sig_rows: List[Dict[str, Any]] = []
    if "B" in sems and "C" in sems:
        for mode in modes:
            b_counts: List[int] = []
            c_counts: List[int] = []
            b_rounds: List[float] = []
            c_rounds: List[float] = []
            b_edit: List[float] = []
            c_edit: List[float] = []

            b_only = 0
            c_only = 0
            for sd in seeds:
                Bs = _read_json(_artifact_root("B", mode) / f"seed_{sd}" / "summary.json")
                Cs = _read_json(_artifact_root("C", mode) / f"seed_{sd}" / "summary.json")
                Brs = Bs.get("results", []) if isinstance(Bs, dict) else []
                Crs = Cs.get("results", []) if isinstance(Cs, dict) else []
                Bm = {str(r.get("task_id")): bool(r.get("ok")) for r in Brs if isinstance(r, dict)}
                Cm = {str(r.get("task_id")): bool(r.get("ok")) for r in Crs if isinstance(r, dict)}

                b_counts.append(sum(1 for ok in Bm.values() if ok))
                c_counts.append(sum(1 for ok in Cm.values() if ok))

                b_rounds.append(float(st.mean([int(r.get("rounds")) for r in Brs if isinstance(r, dict) and isinstance(r.get("rounds"), int)] or [0])))
                c_rounds.append(float(st.mean([int(r.get("rounds")) for r in Crs if isinstance(r, dict) and isinstance(r.get("rounds"), int)] or [0])))

                id2init: Dict[str, Any] = {}
                if isinstance(tasks_list, list):
                    for t in tasks_list:
                        if isinstance(t, dict) and t.get("id") is not None:
                            id2init[str(t.get("id"))] = (t.get("initial_cfg") or {})

                def _seed_edit(rs: List[Any]) -> float:
                    eds: List[int] = []
                    for r in rs:
                        if not isinstance(r, dict):
                            continue
                        tid = str(r.get("task_id"))
                        init_cfg = id2init.get(tid) if isinstance(id2init.get(tid), dict) else {}
                        before = init_cfg.get("policies") if isinstance(init_cfg, dict) else None
                        final_cfg = r.get("final_cfg") if isinstance(r.get("final_cfg"), dict) else {}
                        after = final_cfg.get("policies") if isinstance(final_cfg, dict) else None
                        eds.append(_policy_edit_distance(before, after))
                    return float(st.mean(eds)) if eds else 0.0

                b_edit.append(_seed_edit(Brs))
                c_edit.append(_seed_edit(Crs))

                for tid, bok in Bm.items():
                    cok = bool(Cm.get(tid))
                    if bok and (not cok):
                        b_only += 1
                    if cok and (not bok):
                        c_only += 1

            p_mcn = _mcnemar_exact_pvalue(b_only, c_only)
            p_rounds = _exact_sign_permutation_pvalue([b - c for b, c in zip(b_rounds, c_rounds)])
            p_edit = _exact_sign_permutation_pvalue([b - c for b, c in zip(b_edit, c_edit)])

            sig_rows.append(
                {
                    "compare": "B_vs_C",
                    "mode": mode,
                    "discordant_B_only": b_only,
                    "discordant_C_only": c_only,
                    "mcnemar_p": p_mcn,
                    "rounds_perm_p": p_rounds,
                    "policy_edit_perm_p": p_edit,
                }
            )

    if sig_rows:
        _write_csv(out_dir / "significance_tests.csv", sig_rows, fieldnames=list(sig_rows[0].keys()))

    groups = ["generic", "raw_witness", "blame_only"]
    series = sems

    sr_vals: Dict[Tuple[str, str], float] = {}
    sr_errs: Dict[Tuple[str, str], float] = {}
    rd_vals: Dict[Tuple[str, str], float] = {}
    rd_errs: Dict[Tuple[str, str], float] = {}

    for r in summary_rows:
        g = str(r["mode"])
        s = str(r["sem"])
        sr_vals[(g, s)] = float(r["sr_mean"])
        sr_errs[(g, s)] = float(r["sr_sd"])
        rd_vals[(g, s)] = float(r["avg_rounds_mean"])
        rd_errs[(g, s)] = float(r["avg_rounds_sd"])

    _svg_grouped_bars(
        title=f"Ports100 (seeds {args.seeds}): Success Rate (mean±sd)",
        y_label="Success Rate",
        groups=groups,
        series=series,
        values=sr_vals,
        errors=sr_errs,
        y_max=1.0,
        out_path=out_dir / "fig_success_rate_mean_sd.svg",
    )

    rd_ymax = max(rd_vals.values()) + max(rd_errs.values()) if rd_vals else 1.0
    _svg_grouped_bars(
        title=f"Ports100 (seeds {args.seeds}): Avg Rounds (mean±sd)",
        y_label="Avg Rounds",
        groups=groups,
        series=series,
        values=rd_vals,
        errors=rd_errs,
        y_max=max(1.0, rd_ymax),
        out_path=out_dir / "fig_avg_rounds_mean_sd.svg",
    )

    rounds_rows: List[Dict[str, Any]] = []
    for mode in groups:
        pooled: Dict[str, List[int]] = {"A": [], "B": [], "C": []}
        pooled = {str(sem): [] for sem in series}
        for sem in series:
            key = (sem, mode)
            for _, _, _, rounds in pooled_task_results.get(key, []):
                pooled[sem].append(int(rounds))

        all_rounds: List[int] = []
        for sem in series:
            all_rounds.extend(pooled.get(sem, []))
        all_rounds = sorted(set(all_rounds))
        if not all_rounds:
            continue

        max_r = max(all_rounds)
        xs = list(range(min(all_rounds), max_r + 1))
        cdf_vals: Dict[str, List[float]] = {}

        for sem in series:
            rs = pooled[sem]
            if not rs:
                cdf_vals[sem] = [0.0 for _ in xs]
                continue
            cdf = []
            for x in xs:
                cdf.append(sum(1 for v in rs if v <= x) / len(rs))
            cdf_vals[sem] = cdf

        _svg_cdf(
            title=f"Rounds CDF (pooled over seeds {args.seeds}) - mode={mode}",
            x_label="Rounds",
            y_label="CDF",
            series=series,
            xs=xs,
            cdf_values=cdf_vals,
            out_path=out_dir / f"fig_cdf_rounds_{mode}.svg",
        )

        for sem in series:
            rs = pooled[sem]
            if not rs:
                continue
            frac_ge5 = sum(1 for v in rs if v >= 5) / len(rs)
            frac_eq_max = sum(1 for v in rs if v == max_r) / len(rs)
            rounds_rows.append(
                {
                    "mode": mode,
                    "sem": sem,
                    "n": len(rs),
                    "max_rounds": max_r,
                    "frac_rounds_ge5": frac_ge5,
                    "frac_rounds_eq_max": frac_eq_max,
                }
            )

    rounds_csv = out_dir / "rounds_tail_stats.csv"
    with rounds_csv.open("w", encoding="utf-8", newline="") as f:
        cols = ["mode", "sem", "n", "max_rounds", "frac_rounds_ge5", "frac_rounds_eq_max"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rounds_rows:
            w.writerow(r)

    rounds_hist_rows: List[Dict[str, Any]] = []
    for mode in groups:
        pooled = {str(sem): [] for sem in series}
        for sem in series:
            key = (sem, mode)
            for _, _, _, rounds in pooled_task_results.get(key, []):
                pooled[sem].append(int(rounds))

        all_rounds: List[int] = []
        for sem in series:
            all_rounds.extend(pooled.get(sem, []))
        all_rounds = sorted(set(all_rounds))
        if not all_rounds:
            continue

        max_r = max(all_rounds)
        xs = list(range(min(all_rounds), max_r + 1))
        values: Dict[Tuple[str, str], float] = {}
        for sem in series:
            rs = pooled[sem]
            denom = len(rs) if rs else 1
            for x in xs:
                frac = sum(1 for v in rs if v == x) / denom
                values[(str(x), sem)] = frac
                rounds_hist_rows.append(
                    {
                        "mode": mode,
                        "sem": sem,
                        "rounds": x,
                        "count": sum(1 for v in rs if v == x),
                        "frac": frac,
                    }
                )

        _svg_grouped_bars(
            title=f"Rounds histogram (pooled, fractions) - mode={mode}",
            y_label="Fraction of tasks",
            groups=[str(x) for x in xs],
            series=series,
            values=values,
            errors=None,
            y_max=1.0,
            out_path=out_dir / f"fig_hist_rounds_{mode}.svg",
        )

    rounds_hist_csv = out_dir / "rounds_histogram.csv"
    with rounds_hist_csv.open("w", encoding="utf-8", newline="") as f:
        cols = ["mode", "sem", "rounds", "count", "frac"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rounds_hist_rows:
            w.writerow(r)

    evidence_rows: List[Dict[str, Any]] = []

    for sem in series:
        for mode in ["generic", "raw_witness", "blame_only"]:
            spec = ExpSpec(sem, mode, _artifact_root(sem, mode))
            for sd in seeds:
                summ = spec.root / f"seed_{sd}" / "summary.json"
                d = _read_json(summ)
                for r in d.get("results", []):
                    if not isinstance(r, dict):
                        continue
                    tid = str(r.get("task_id") or "").strip()
                    if not tid:
                        continue
                    jl = spec.root / f"seed_{sd}" / f"{tid}.jsonl"
                    max_ver_json_len = 0
                    max_msg_len = 0
                    max_blame_size = 0
                    if jl.exists():
                        for obj in _iter_jsonl(jl):
                            pr = obj.get("prompt")
                            ver_full = obj.get("verification")
                            max_ver_json_len = max(max_ver_json_len, _verification_json_len_from_prompt(pr))
                            max_msg_len = max(max_msg_len, _max_violation_message_len_from_verification(ver_full))
                            max_blame_size = max(max_blame_size, _blame_size_from_prompt(pr))
                    evidence_rows.append(
                        {
                            "sem": sem,
                            "mode": mode,
                            "seed": sd,
                            "task_id": tid,
                            "template": _template_from_task_id(tid),
                            "ok": bool(r.get("ok")),
                            "max_prompt_verification_json_len": max_ver_json_len,
                            "max_violation_message_len": max_msg_len,
                            "max_blame_size": max_blame_size,
                        }
                    )

    def summarize_numeric(xs: List[float]) -> Dict[str, Any]:
        xs2 = [float(x) for x in xs if x is not None]
        xs2.sort()
        mu, sd = _mean_sd(xs2)
        return {
            "n": len(xs2),
            "mean": mu,
            "sd": sd,
            "p50": _quantile(xs2, 0.5),
            "p90": _quantile(xs2, 0.9),
            "p95": _quantile(xs2, 0.95),
            "p99": _quantile(xs2, 0.99),
        }

    evidence_summary_rows: List[Dict[str, Any]] = []
    for sem in series:
        for mode in ["generic", "raw_witness", "blame_only"]:
            xs_ver = [
                float(r["max_prompt_verification_json_len"])
                for r in evidence_rows
                if r["sem"] == sem and r["mode"] == mode
            ]
            s_ver = summarize_numeric(xs_ver)
            evidence_summary_rows.append(
                {
                    "sem": sem,
                    "metric": f"{mode}_max_prompt_verification_json_len",
                    **s_ver,
                }
            )

            xs_msg = [
                float(r["max_violation_message_len"])
                for r in evidence_rows
                if r["sem"] == sem and r["mode"] == mode
            ]
            s_msg = summarize_numeric(xs_msg)
            evidence_summary_rows.append(
                {
                    "sem": sem,
                    "metric": f"{mode}_max_violation_message_len",
                    **s_msg,
                }
            )

            if mode == "blame_only":
                xs_bl = [
                    float(r["max_blame_size"])
                    for r in evidence_rows
                    if r["sem"] == sem and r["mode"] == mode
                ]
                s_bl = summarize_numeric(xs_bl)
                evidence_summary_rows.append(
                    {
                        "sem": sem,
                        "metric": "blame_only_max_blame_size",
                        **s_bl,
                    }
                )

    ev_csv = out_dir / "evidence_metrics_summary.csv"
    with ev_csv.open("w", encoding="utf-8", newline="") as f:
        cols = ["sem", "metric", "n", "mean", "sd", "p50", "p90", "p95", "p99"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in evidence_summary_rows:
            w.writerow(r)

    template_rows: List[Dict[str, Any]] = []
    for spec in specs:
        for sd in seeds:
            summ = spec.root / f"seed_{sd}" / "summary.json"
            d = _read_json(summ)
            for r in d.get("results", []):
                if not isinstance(r, dict):
                    continue
                tid = str(r.get("task_id") or "").strip()
                if not tid:
                    continue
                template_rows.append(
                    {
                        "sem": spec.sem,
                        "mode": spec.mode,
                        "seed": sd,
                        "task_id": tid,
                        "template": _template_from_task_id(tid),
                        "ok": bool(r.get("ok")),
                        "rounds": int(r.get("rounds") or 0),
                    }
                )

    tpl_agg: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for r in template_rows:
        key = (r["sem"], r["mode"], r["template"])
        a = tpl_agg.get(key)
        if a is None:
            a = {"n": 0, "ok": 0, "rounds_sum": 0}
            tpl_agg[key] = a
        a["n"] += 1
        a["ok"] += int(bool(r["ok"]))
        a["rounds_sum"] += int(r["rounds"])

    tpl_out_rows: List[Dict[str, Any]] = []
    for (sem, mode, tpl), a in tpl_agg.items():
        n = int(a["n"])
        ok = int(a["ok"])
        tpl_out_rows.append(
            {
                "sem": sem,
                "mode": mode,
                "template": tpl,
                "n": n,
                "ok": ok,
                "success_rate": (ok / n) if n else 0.0,
                "avg_rounds": (float(a["rounds_sum"]) / n) if n else 0.0,
            }
        )

    tpl_out_rows.sort(key=lambda r: (r["mode"], r["template"], r["sem"]))
    tpl_csv = out_dir / "per_template_breakdown.csv"
    with tpl_csv.open("w", encoding="utf-8", newline="") as f:
        cols = ["sem", "mode", "template", "n", "ok", "success_rate", "avg_rounds"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in tpl_out_rows:
            w.writerow(r)

    case_studies: Dict[str, List[Dict[str, Any]]] = {}

    if "A" in series and "B" in series:
        for mode in groups:
            counts = {}
            by_seed = {}
            for sd in seeds:
                As = _read_json(_artifact_root("A", mode) / f"seed_{sd}" / "summary.json")
                Bs = _read_json(_artifact_root("B", mode) / f"seed_{sd}" / "summary.json")
                Am = {str(r.get("task_id")): bool(r.get("ok")) for r in As.get("results", []) if isinstance(r, dict)}
                Bm = {str(r.get("task_id")): bool(r.get("ok")) for r in Bs.get("results", []) if isinstance(r, dict)}
                for tid, aok in Am.items():
                    if tid not in Bm:
                        continue
                    bok = Bm[tid]
                    if (not aok) and bok:
                        counts[tid] = counts.get(tid, 0) + 1
                        by_seed.setdefault(tid, []).append(sd)

            ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
            picked = [tid for tid, _ in ranked[: max(0, int(args.case_studies_per_mode))]]

            out_cases: List[Dict[str, Any]] = []
            for tid in picked:
                seeds_for_tid = by_seed.get(tid) or []
                sd = seeds_for_tid[0] if seeds_for_tid else seeds[0]

                t = task_by_id.get(tid) or {}
                intent_text = t.get("intent")
                initial_cfg = t.get("initial_cfg") if isinstance(t.get("initial_cfg"), dict) else {}
                initial_pols = initial_cfg.get("policies")

                a_jsonl = _artifact_root("A", mode) / f"seed_{sd}" / f"{tid}.jsonl"
                b_jsonl = _artifact_root("B", mode) / f"seed_{sd}" / f"{tid}.jsonl"

                a_last: Optional[Dict[str, Any]] = None
                for obj in _iter_jsonl(a_jsonl):
                    a_last = obj

                b_last: Optional[Dict[str, Any]] = None
                for obj in _iter_jsonl(b_jsonl):
                    b_last = obj

                a_ver = a_last.get("verification") if isinstance(a_last, dict) else None
                a_prompt = a_last.get("prompt") if isinstance(a_last, dict) else None
                a_ce = _extract_best_contract_exception_text(a_ver)
                a_msg = _extract_best_violation_message(a_ver)
                a_blame = a_prompt.get("blame_set") if isinstance(a_prompt, dict) else None
                a_feedback = a_prompt.get("feedback") if isinstance(a_prompt, dict) else None

                b_cfg = b_last.get("candidate_cfg") if isinstance(b_last, dict) else None
                b_pols = None
                if isinstance(b_cfg, dict):
                    b_pols = b_cfg.get("policies")

                pol_diff = _policy_diff(initial_pols, b_pols)

                out_cases.append(
                    {
                        "task_id": tid,
                        "seed": sd,
                        "b_only_count": counts.get(tid, 0),
                        "intent": intent_text,
                        "template": _template_from_task_id(tid),
                        "A_failure_contract_exception": a_ce,
                        "A_failure_violation_message": a_msg,
                        "A_failure_feedback": a_feedback,
                        "A_failure_blame_set": a_blame,
                        "B_policy_diff_from_initial": pol_diff,
                    }
                )

            case_studies[mode] = out_cases

    case_json = out_dir / "case_studies_b_only.json"
    case_json.write_text(json.dumps(case_studies, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines: List[str] = []
    md_lines.append(f"# Ports100 Multi-seed Report (seeds {args.seeds})")
    md_lines.append("")
    md_lines.append("## Figures")
    md_lines.append("")
    md_lines.append("- Success rate bar chart: `fig_success_rate_mean_sd.svg`")
    md_lines.append("- Avg rounds bar chart: `fig_avg_rounds_mean_sd.svg`")
    for mode in groups:
        md_lines.append(f"- Rounds CDF (mode={mode}): `fig_cdf_rounds_{mode}.svg`")
    md_lines.append("")

    md_lines.append("## Summary (mean±sd across seeds)")
    md_lines.append("")
    md_lines.append("| Sem | Mode | SR mean±sd | Avg rounds mean±sd |")
    md_lines.append("|---|---|---:|---:|")
    for mode in groups:
        for sem in series:
            r = next((x for x in summary_rows if x["sem"] == sem and x["mode"] == mode), None)
            if r is None:
                continue
            md_lines.append(
                f"| {sem} | {mode} | {r['sr_mean']:.3f}±{r['sr_sd']:.3f} | {r['avg_rounds_mean']:.2f}±{r['avg_rounds_sd']:.2f} |"
            )

    md_lines.append("")
    md_lines.append("## Evidence metrics")
    md_lines.append("")
    md_lines.append("See `evidence_metrics_summary.csv`.")
    md_lines.append("")

    md_lines.append("## Case studies (B-only tasks)")
    md_lines.append("")
    md_lines.append("See `case_studies_b_only.json`.")
    md_lines.append("")

    md_lines.append("## Per-template breakdown")
    md_lines.append("")
    md_lines.append("See `per_template_breakdown.csv`.")

    (out_dir / "README.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
