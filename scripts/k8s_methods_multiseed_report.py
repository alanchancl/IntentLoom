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
    method: str
    root: Path


def _mean_sd(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    if len(xs) == 1:
        return float(xs[0]), 0.0
    return float(st.mean(xs)), float(st.pstdev(xs))


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
    n = int(b + c)
    if n <= 0:
        return 1.0
    k = int(min(b, c))
    log2 = math.log(2.0)

    def _logpmf(i: int) -> float:
        return math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * log2

    lo_logs = [_logpmf(i) for i in range(0, k + 1)]
    hi_logs = [_logpmf(i) for i in range(n - k, n + 1)]
    lo = math.exp(_logsumexp(lo_logs))
    hi = math.exp(_logsumexp(hi_logs))
    p = 2.0 * min(lo, hi)
    return min(1.0, max(0.0, p))


def _exact_sign_permutation_pvalue(diffs: List[float]) -> float:
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


def _write_csv(path: Path, rows: List[Dict[str, Any]], *, fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            if not isinstance(r, dict):
                continue
            w.writerow({k: r.get(k) for k in fieldnames})


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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
    for n in names:
        if n not in b:
            added.append(n)
            continue
        if n not in a:
            removed.append(n)
            continue
        if _json_canon(b.get(n)) == _json_canon(a.get(n)):
            unchanged.append(n)
            continue
        changed.append(n)
    return {"added": added, "removed": removed, "changed": changed, "unchanged": unchanged}


def _policy_edit_distance(before: Any, after: Any) -> int:
    d = _policy_diff(before, after)
    return int(len(d.get("added") or [])) + int(len(d.get("removed") or [])) + int(len(d.get("changed") or []))


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


def _llm_calls_from_jsonl(path: Path) -> int:
    if not path.exists() or not path.is_file():
        return 0
    calls = 0
    for obj in _iter_jsonl(path):
        calls += 1
        if isinstance(obj.get("reflection"), str) and str(obj.get("reflection")).strip():
            calls += 1
    return calls


def _seed_dirs(exp_root: Path) -> List[Path]:
    xs = [p for p in exp_root.iterdir() if p.is_dir() and p.name.startswith("seed_")]
    return sorted(xs, key=lambda p: p.name)


def _load_seed_summary(seed_dir: Path) -> Optional[Dict[str, Any]]:
    p = seed_dir / "summary.json"
    if not p.exists() or not p.is_file():
        return None
    obj = _read_json(p)
    return obj if isinstance(obj, dict) else None


def _load_task_results_for_seed(seed_dir: Path) -> Dict[str, Dict[str, Any]]:
    summary = _load_seed_summary(seed_dir)
    out: Dict[str, Dict[str, Any]] = {}
    if isinstance(summary, dict) and isinstance(summary.get("results"), list):
        for r in summary.get("results"):
            if not isinstance(r, dict):
                continue
            tid = r.get("task_id")
            if not isinstance(tid, str) or not tid:
                continue
            out[tid] = r
    return out


def _method_dir(root: Path, method: str) -> Optional[Path]:
    p = root / method
    if p.exists() and p.is_dir():
        return p
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=str, required=True)
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument(
        "--methods",
        type=str,
        default="m1_vanilla,m2_fewshot,m3_standard_feedback,m4_reflexion,m5_intentloom",
    )
    ap.add_argument(
        "--comparisons",
        type=str,
        default="m5_intentloom:m3_standard_feedback,m5_intentloom:m4_reflexion",
        help="Comma-separated pairs A:B to compare (A vs B).",
    )
    args = ap.parse_args()

    tasks_path = Path(args.tasks)
    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks_list = _read_json(tasks_path)
    if not isinstance(tasks_list, list):
        raise RuntimeError("tasks must be a JSON list")

    id2init: Dict[str, Dict[str, Any]] = {}
    for t in tasks_list:
        if not isinstance(t, dict):
            continue
        tid = t.get("id")
        cfg0 = t.get("initial_cfg")
        if isinstance(tid, str) and tid and isinstance(cfg0, dict):
            id2init[tid] = cfg0

    methods = [m.strip() for m in str(args.methods).split(",") if m.strip()]
    specs: List[ExpSpec] = []
    for m in methods:
        md = _method_dir(root, m)
        if md is None:
            continue
        specs.append(ExpSpec(method=m, root=md))

    per_seed: Dict[str, Dict[str, Dict[str, float]]] = {}
    pooled_ok: Dict[str, Dict[str, int]] = {}

    for spec in specs:
        seed_dirs = _seed_dirs(spec.root)
        if not seed_dirs:
            seed_dirs = [spec.root]

        per_seed.setdefault(spec.method, {})
        pooled_ok.setdefault(spec.method, {})

        for sd in seed_dirs:
            seed_name = sd.name
            summary = _load_seed_summary(sd)
            if summary is None:
                continue

            task_results = _load_task_results_for_seed(sd)

            ok_cnt = 0
            rounds_sum = 0
            llm_calls_sum = 0
            edit_sum = 0
            n = 0

            ok_rounds_sum = 0
            ok_llm_calls_sum = 0
            ok_edit_sum = 0
            ok_n = 0

            for tid, init_cfg in id2init.items():
                r = task_results.get(tid)
                if not isinstance(r, dict):
                    continue
                n += 1

                ok = bool(r.get("ok"))
                rounds = int(r.get("rounds") or 0)
                llm_calls = r.get("llm_calls")
                if not isinstance(llm_calls, int):
                    llm_calls = _llm_calls_from_jsonl(sd / f"{tid}.jsonl") or rounds

                final_cfg = r.get("final_cfg")
                if not isinstance(final_cfg, dict):
                    final_cfg = {}

                init_pols = init_cfg.get("policies") if isinstance(init_cfg, dict) else None
                final_pols = final_cfg.get("policies") if isinstance(final_cfg, dict) else None
                ed = _policy_edit_distance(init_pols, final_pols)

                pooled_ok[spec.method][f"{seed_name}:{tid}"] = 1 if ok else 0

                rounds_sum += rounds
                llm_calls_sum += int(llm_calls)
                edit_sum += ed
                if ok:
                    ok_cnt += 1
                    ok_rounds_sum += rounds
                    ok_llm_calls_sum += int(llm_calls)
                    ok_edit_sum += ed
                    ok_n += 1

            sr = (ok_cnt / n) if n else 0.0
            avg_rounds = (rounds_sum / n) if n else 0.0
            avg_llm_calls = (llm_calls_sum / n) if n else 0.0
            avg_edit = (edit_sum / n) if n else 0.0

            ok_avg_rounds = (ok_rounds_sum / ok_n) if ok_n else 0.0
            ok_avg_llm_calls = (ok_llm_calls_sum / ok_n) if ok_n else 0.0
            ok_avg_edit = (ok_edit_sum / ok_n) if ok_n else 0.0

            per_seed[spec.method][seed_name] = {
                "n": float(n),
                "ok": float(ok_cnt),
                "sr": float(sr),
                "avg_rounds": float(avg_rounds),
                "avg_llm_calls": float(avg_llm_calls),
                "avg_policy_edit": float(avg_edit),
                "ok_avg_rounds": float(ok_avg_rounds),
                "ok_avg_llm_calls": float(ok_avg_llm_calls),
                "ok_avg_policy_edit": float(ok_avg_edit),
            }

    summary_rows: List[Dict[str, Any]] = []
    for method, seeds in per_seed.items():
        ok_x = [v["ok"] for v in seeds.values()]
        sr_x = [v["sr"] for v in seeds.values()]
        rounds_x = [v["avg_rounds"] for v in seeds.values()]
        calls_x = [v["avg_llm_calls"] for v in seeds.values()]
        edit_x = [v["avg_policy_edit"] for v in seeds.values()]

        ok_s_x = [v["ok_avg_rounds"] for v in seeds.values()]
        calls_s_x = [v["ok_avg_llm_calls"] for v in seeds.values()]
        edit_s_x = [v["ok_avg_policy_edit"] for v in seeds.values()]

        ok_mean, ok_sd = _mean_sd(ok_x)
        sr_mean, sr_sd = _mean_sd(sr_x)
        rounds_mean, rounds_sd = _mean_sd(rounds_x)
        calls_mean, calls_sd = _mean_sd(calls_x)
        edit_mean, edit_sd = _mean_sd(edit_x)

        ok_rounds_mean, ok_rounds_sd = _mean_sd(ok_s_x)
        ok_calls_mean, ok_calls_sd = _mean_sd(calls_s_x)
        ok_edit_mean, ok_edit_sd = _mean_sd(edit_s_x)

        summary_rows.append(
            {
                "method": method,
                "seeds": len(seeds),
                "ok_mean": ok_mean,
                "ok_sd": ok_sd,
                "sr_mean": sr_mean,
                "sr_sd": sr_sd,
                "avg_rounds_mean": rounds_mean,
                "avg_rounds_sd": rounds_sd,
                "avg_llm_calls_mean": calls_mean,
                "avg_llm_calls_sd": calls_sd,
                "avg_policy_edit_mean": edit_mean,
                "avg_policy_edit_sd": edit_sd,
                "ok_avg_rounds_mean": ok_rounds_mean,
                "ok_avg_rounds_sd": ok_rounds_sd,
                "ok_avg_llm_calls_mean": ok_calls_mean,
                "ok_avg_llm_calls_sd": ok_calls_sd,
                "ok_avg_policy_edit_mean": ok_edit_mean,
                "ok_avg_policy_edit_sd": ok_edit_sd,
            }
        )

    summary_rows = sorted(summary_rows, key=lambda r: str(r.get("method")))
    _write_csv(
        out_dir / "summary_multiseed.csv",
        summary_rows,
        fieldnames=[
            "method",
            "seeds",
            "ok_mean",
            "ok_sd",
            "sr_mean",
            "sr_sd",
            "avg_rounds_mean",
            "avg_rounds_sd",
            "avg_llm_calls_mean",
            "avg_llm_calls_sd",
            "avg_policy_edit_mean",
            "avg_policy_edit_sd",
            "ok_avg_rounds_mean",
            "ok_avg_rounds_sd",
            "ok_avg_llm_calls_mean",
            "ok_avg_llm_calls_sd",
            "ok_avg_policy_edit_mean",
            "ok_avg_policy_edit_sd",
        ],
    )

    comparisons = [c.strip() for c in str(args.comparisons).split(",") if c.strip() and ":" in c]
    sig_rows: List[Dict[str, Any]] = []
    for comp in comparisons:
        a, b = [x.strip() for x in comp.split(":", 1)]
        if a not in pooled_ok or b not in pooled_ok:
            continue

        keys = sorted(set(pooled_ok[a].keys()) & set(pooled_ok[b].keys()))
        b_only = 0
        c_only = 0
        for k in keys:
            ao = pooled_ok[a].get(k)
            bo = pooled_ok[b].get(k)
            if ao == 1 and bo == 0:
                b_only += 1
            if ao == 0 and bo == 1:
                c_only += 1
        mcnemar_p = _mcnemar_exact_pvalue(b_only, c_only)

        diffs_rounds: List[float] = []
        diffs_calls: List[float] = []
        diffs_edits: List[float] = []
        for seed_name, a_stats in per_seed.get(a, {}).items():
            b_stats = per_seed.get(b, {}).get(seed_name)
            if not isinstance(b_stats, dict):
                continue
            diffs_rounds.append(float(a_stats.get("avg_rounds", 0.0)) - float(b_stats.get("avg_rounds", 0.0)))
            diffs_calls.append(float(a_stats.get("avg_llm_calls", 0.0)) - float(b_stats.get("avg_llm_calls", 0.0)))
            diffs_edits.append(float(a_stats.get("avg_policy_edit", 0.0)) - float(b_stats.get("avg_policy_edit", 0.0)))

        rounds_p = _exact_sign_permutation_pvalue(diffs_rounds)
        calls_p = _exact_sign_permutation_pvalue(diffs_calls)
        edits_p = _exact_sign_permutation_pvalue(diffs_edits)

        sig_rows.append(
            {
                "compare": f"{a}_vs_{b}",
                "discordant_A_only": b_only,
                "discordant_B_only": c_only,
                "mcnemar_p": mcnemar_p,
                "rounds_perm_p": rounds_p,
                "llm_calls_perm_p": calls_p,
                "policy_edit_perm_p": edits_p,
            }
        )

    _write_csv(
        out_dir / "significance_tests.csv",
        sig_rows,
        fieldnames=[
            "compare",
            "discordant_A_only",
            "discordant_B_only",
            "mcnemar_p",
            "rounds_perm_p",
            "llm_calls_perm_p",
            "policy_edit_perm_p",
        ],
    )


if __name__ == "__main__":
    main()
