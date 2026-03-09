import argparse
import csv
import json
import math
import statistics as st
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: List[Dict[str, Any]], *, fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            if not isinstance(r, dict):
                continue
            w.writerow({k: r.get(k) for k in fieldnames})


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
    return float(min(1.0, max(0.0, p)))


def _binom_two_sided_pvalue(k: int, n: int) -> float:
    if n <= 0:
        return 1.0
    k = int(min(k, n - k))
    log2 = math.log(2.0)

    def _logpmf(i: int) -> float:
        return math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * log2

    lo_logs = [_logpmf(i) for i in range(0, k + 1)]
    hi_logs = [_logpmf(i) for i in range(n - k, n + 1)]
    lo = math.exp(_logsumexp(lo_logs))
    hi = math.exp(_logsumexp(hi_logs))
    p = 2.0 * min(lo, hi)
    return float(min(1.0, max(0.0, p)))


def _wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    phat = float(k) / float(n)
    denom = 1.0 + (z * z) / float(n)
    center = (phat + (z * z) / (2.0 * float(n))) / denom
    half = (z * math.sqrt((phat * (1.0 - phat) + (z * z) / (4.0 * float(n))) / float(n))) / denom
    return float(center - half), float(center + half)


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


def _policy_edit_distance(before: Any, after: Any) -> int:
    b = _policy_index(before)
    a = _policy_index(after)
    names = set(b.keys()) | set(a.keys())
    added = 0
    removed = 0
    changed = 0
    for n in names:
        if n not in b:
            added += 1
            continue
        if n not in a:
            removed += 1
            continue
        if _json_canon(b.get(n)) != _json_canon(a.get(n)):
            changed += 1
    return int(added + removed + changed)


def _load_tasks(path: Path) -> Dict[str, Dict[str, Any]]:
    obj = _read_json(path)
    if not isinstance(obj, list):
        raise ValueError(f"Expected JSON list: {path}")
    out: Dict[str, Dict[str, Any]] = {}
    for t in obj:
        if not isinstance(t, dict):
            continue
        tid = str(t.get("id") or "")
        if not tid:
            continue
        out[tid] = t
    return out


def _list_seeds(out_dir: Path) -> List[int]:
    seeds: List[int] = []
    for p in out_dir.glob("seed_*"):
        if not p.is_dir():
            continue
        try:
            seeds.append(int(p.name.split("_", 1)[1]))
        except Exception:
            continue
    return sorted(set(seeds))


def _mode_from_task_id(task_id: str) -> str:
    for m in ("selector_typo", "port_mismatch", "over_permissive", "ipblock_mismatch"):
        if f"_{m}" in task_id:
            return m
    return "unknown"


def _load_seed_results(seed_dir: Path, *, tasks_by_id: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    s = _read_json(seed_dir / "summary.json")
    results = s.get("results")
    if not isinstance(results, list):
        results = []

    out: Dict[str, Dict[str, Any]] = {}
    for r in results:
        if not isinstance(r, dict):
            continue
        tid = str(r.get("task_id") or "")
        if not tid:
            continue
        ok = bool(r.get("ok"))
        rounds = float(r.get("rounds") or 0.0)
        calls = float(r.get("llm_calls") or 0.0)

        final_cfg = r.get("final_cfg") if isinstance(r.get("final_cfg"), dict) else {}
        t = tasks_by_id.get(tid) or {}
        init_cfg = t.get("initial_cfg") if isinstance(t.get("initial_cfg"), dict) else {}
        before = init_cfg.get("policies") if isinstance(init_cfg, dict) else None
        after = final_cfg.get("policies") if isinstance(final_cfg, dict) else None
        ed = float(_policy_edit_distance(before, after))

        out[tid] = {
            "ok": ok,
            "rounds": rounds,
            "llm_calls": calls,
            "policy_edit": ed,
            "mode": _mode_from_task_id(tid),
        }

    return out


def _stable_failures(fail_sets_by_seed: Dict[int, set]) -> Dict[str, Any]:
    seeds = sorted(fail_sets_by_seed)
    if not seeds:
        return {"stable_3of3": [], "fail_2plus": []}

    inter: Optional[set] = None
    counts: Dict[str, int] = {}
    for sd in seeds:
        s = fail_sets_by_seed.get(sd) or set()
        if inter is None:
            inter = set(s)
        else:
            inter = inter.intersection(s)
        for tid in s:
            counts[tid] = counts.get(tid, 0) + 1

    inter = inter or set()
    fail_2plus = sorted([tid for tid, c in counts.items() if c >= 2])
    return {"stable_3of3": sorted(inter), "fail_2plus": fail_2plus}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--m3", type=str, default=str(Path("artifacts") / "exp_realk8s120_optionc_m3"))
    ap.add_argument("--m4", type=str, default=str(Path("artifacts") / "exp_realk8s120_optionc_m4"))
    ap.add_argument("--m5", type=str, default=str(Path("artifacts") / "exp_realk8s120_optionc_blame_m5"))
    ap.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds; if unset, uses common seeds across all methods.",
    )
    args = ap.parse_args()

    tasks_by_id = _load_tasks(Path(args.tasks))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    method_dirs = {
        "M3": Path(args.m3),
        "M4": Path(args.m4),
        "M5": Path(args.m5),
    }

    if args.seeds and str(args.seeds).strip():
        seeds = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]
    else:
        seed_sets: List[set] = []
        for dp in method_dirs.values():
            seed_sets.append(set(_list_seeds(dp)))
        common = set.intersection(*seed_sets) if seed_sets else set()
        seeds = sorted(common)

    if not seeds:
        raise RuntimeError("No seeds found to compare")

    per_method_by_seed: Dict[str, Dict[int, Dict[str, Dict[str, Any]]]] = {"M3": {}, "M4": {}, "M5": {}}

    summary_rows: List[Dict[str, Any]] = []
    mode_rows: List[Dict[str, Any]] = []
    stable_failures: Dict[str, Any] = {}

    for m, dp in method_dirs.items():
        ok_total = 0
        n_total = 0
        rounds_all: List[float] = []
        calls_all: List[float] = []
        edits_all: List[float] = []

        fail_sets_by_seed: Dict[int, set] = {}
        mode_ok: Dict[str, int] = {}
        mode_n: Dict[str, int] = {}

        for sd in seeds:
            seed_dir = dp / f"seed_{sd}"
            by_task = _load_seed_results(seed_dir, tasks_by_id=tasks_by_id)
            per_method_by_seed[m][sd] = by_task

            fails: set = set()
            for tid, r in by_task.items():
                ok = bool(r.get("ok"))
                md = str(r.get("mode") or "unknown")
                mode_n[md] = mode_n.get(md, 0) + 1
                if ok:
                    ok_total += 1
                    mode_ok[md] = mode_ok.get(md, 0) + 1
                else:
                    fails.add(tid)

                n_total += 1
                rounds_all.append(float(r.get("rounds") or 0.0))
                calls_all.append(float(r.get("llm_calls") or 0.0))
                edits_all.append(float(r.get("policy_edit") or 0.0))

            fail_sets_by_seed[sd] = fails

        lo, hi = _wilson_ci(ok_total, n_total)
        summary_rows.append(
            {
                "method": m,
                "seeds": ",".join(str(s) for s in seeds),
                "ok": ok_total,
                "tasks": n_total,
                "success_rate": (float(ok_total) / float(n_total)) if n_total else 0.0,
                "wilson95_lo": lo,
                "wilson95_hi": hi,
                "avg_rounds": float(st.mean(rounds_all)) if rounds_all else 0.0,
                "avg_llm_calls": float(st.mean(calls_all)) if calls_all else 0.0,
                "avg_policy_edit": float(st.mean(edits_all)) if edits_all else 0.0,
                "stable_fail_3of3": len(_stable_failures(fail_sets_by_seed).get("stable_3of3") or []),
                "stable_fail_2plus": len(_stable_failures(fail_sets_by_seed).get("fail_2plus") or []),
            }
        )

        for md in sorted(mode_n.keys()):
            ok = int(mode_ok.get(md, 0))
            n = int(mode_n.get(md, 0))
            lo_m, hi_m = _wilson_ci(ok, n)
            mode_rows.append(
                {
                    "method": m,
                    "mode": md,
                    "ok": ok,
                    "tasks": n,
                    "success_rate": (float(ok) / float(n)) if n else 0.0,
                    "wilson95_lo": lo_m,
                    "wilson95_hi": hi_m,
                }
            )

        stable_failures[m] = _stable_failures(fail_sets_by_seed)

    sig_rows: List[Dict[str, Any]] = []
    sig_mode_rows: List[Dict[str, Any]] = []
    pairs = [("M4", "M3"), ("M5", "M3"), ("M5", "M4")]

    def _pairwise_success_counts(a: str, b: str) -> Tuple[int, int, int, int]:
        a_ok_b_fail = 0
        a_fail_b_ok = 0
        both_ok = 0
        both_fail = 0
        for sd in seeds:
            a_tasks = per_method_by_seed[a].get(sd) or {}
            b_tasks = per_method_by_seed[b].get(sd) or {}
            tids = sorted(set(a_tasks.keys()) & set(b_tasks.keys()))
            for tid in tids:
                ao = bool(a_tasks.get(tid, {}).get("ok"))
                bo = bool(b_tasks.get(tid, {}).get("ok"))
                if ao and (not bo):
                    a_ok_b_fail += 1
                elif (not ao) and bo:
                    a_fail_b_ok += 1
                elif ao and bo:
                    both_ok += 1
                else:
                    both_fail += 1
        return a_ok_b_fail, a_fail_b_ok, both_ok, both_fail

    def _pairwise_success_counts_by_mode(a: str, b: str, md: str) -> Tuple[int, int, int, int]:
        a_ok_b_fail = 0
        a_fail_b_ok = 0
        both_ok = 0
        both_fail = 0
        for sd in seeds:
            a_tasks = per_method_by_seed[a].get(sd) or {}
            b_tasks = per_method_by_seed[b].get(sd) or {}
            tids = sorted(set(a_tasks.keys()) & set(b_tasks.keys()))
            for tid in tids:
                mode_tid = str(a_tasks.get(tid, {}).get("mode") or b_tasks.get(tid, {}).get("mode") or "unknown")
                if mode_tid != md:
                    continue
                ao = bool(a_tasks.get(tid, {}).get("ok"))
                bo = bool(b_tasks.get(tid, {}).get("ok"))
                if ao and (not bo):
                    a_ok_b_fail += 1
                elif (not ao) and bo:
                    a_fail_b_ok += 1
                elif ao and bo:
                    both_ok += 1
                else:
                    both_fail += 1
        return a_ok_b_fail, a_fail_b_ok, both_ok, both_fail

    def _sign_test_metric(a: str, b: str, key: str) -> Dict[str, Any]:
        pos = 0
        neg = 0
        diffs: List[float] = []
        for sd in seeds:
            a_tasks = per_method_by_seed[a].get(sd) or {}
            b_tasks = per_method_by_seed[b].get(sd) or {}
            tids = sorted(set(a_tasks.keys()) & set(b_tasks.keys()))
            for tid in tids:
                da = float(a_tasks.get(tid, {}).get(key) or 0.0)
                db = float(b_tasks.get(tid, {}).get(key) or 0.0)
                d = db - da
                diffs.append(d)
                if d > 0:
                    pos += 1
                elif d < 0:
                    neg += 1
        n = pos + neg
        p = _binom_two_sided_pvalue(min(pos, neg), n)
        return {
            f"{key}_diff_mean_b_minus_a": float(st.mean(diffs)) if diffs else 0.0,
            f"{key}_sign_pos": pos,
            f"{key}_sign_neg": neg,
            f"{key}_sign_n": n,
            f"{key}_sign_p": p,
        }

    def _sign_test_metric_by_mode(a: str, b: str, key: str, md: str) -> Dict[str, Any]:
        pos = 0
        neg = 0
        diffs: List[float] = []
        for sd in seeds:
            a_tasks = per_method_by_seed[a].get(sd) or {}
            b_tasks = per_method_by_seed[b].get(sd) or {}
            tids = sorted(set(a_tasks.keys()) & set(b_tasks.keys()))
            for tid in tids:
                mode_tid = str(a_tasks.get(tid, {}).get("mode") or b_tasks.get(tid, {}).get("mode") or "unknown")
                if mode_tid != md:
                    continue
                da = float(a_tasks.get(tid, {}).get(key) or 0.0)
                db = float(b_tasks.get(tid, {}).get(key) or 0.0)
                d = db - da
                diffs.append(d)
                if d > 0:
                    pos += 1
                elif d < 0:
                    neg += 1
        n = pos + neg
        p = _binom_two_sided_pvalue(min(pos, neg), n)
        return {
            f"{key}_diff_mean_b_minus_a": float(st.mean(diffs)) if diffs else 0.0,
            f"{key}_sign_pos": pos,
            f"{key}_sign_neg": neg,
            f"{key}_sign_n": n,
            f"{key}_sign_p": p,
        }

    sr_by_method = {r["method"]: float(r["success_rate"]) for r in summary_rows if isinstance(r, dict) and r.get("method")}

    for a, b in pairs:
        b_cnt, c_cnt, both_ok, both_fail = _pairwise_success_counts(a, b)
        mcn_p = _mcnemar_exact_pvalue(b_cnt, c_cnt)
        delta_sr = float(sr_by_method.get(a, 0.0) - sr_by_method.get(b, 0.0))

        row: Dict[str, Any] = {
            "method_a": a,
            "method_b": b,
            "paired_b_a_ok_b_fail": b_cnt,
            "paired_c_a_fail_b_ok": c_cnt,
            "paired_both_ok": both_ok,
            "paired_both_fail": both_fail,
            "mcnemar_p_success": mcn_p,
            "delta_success_rate_a_minus_b": delta_sr,
        }
        row.update(_sign_test_metric(a, b, "rounds"))
        row.update(_sign_test_metric(a, b, "llm_calls"))
        row.update(_sign_test_metric(a, b, "policy_edit"))
        sig_rows.append(row)

        for md in ("selector_typo", "port_mismatch", "over_permissive", "ipblock_mismatch", "unknown"):
            b_cnt_m, c_cnt_m, both_ok_m, both_fail_m = _pairwise_success_counts_by_mode(a, b, md)
            if (b_cnt_m + c_cnt_m + both_ok_m + both_fail_m) <= 0:
                continue
            row_m: Dict[str, Any] = {
                "mode": md,
                "method_a": a,
                "method_b": b,
                "paired_b_a_ok_b_fail": b_cnt_m,
                "paired_c_a_fail_b_ok": c_cnt_m,
                "paired_both_ok": both_ok_m,
                "paired_both_fail": both_fail_m,
                "mcnemar_p_success": _mcnemar_exact_pvalue(b_cnt_m, c_cnt_m),
            }
            row_m.update(_sign_test_metric_by_mode(a, b, "rounds", md))
            row_m.update(_sign_test_metric_by_mode(a, b, "llm_calls", md))
            row_m.update(_sign_test_metric_by_mode(a, b, "policy_edit", md))
            sig_mode_rows.append(row_m)

    task_rows: List[Dict[str, Any]] = []
    for sd in seeds:
        all_tids = set()
        for m in ("M3", "M4", "M5"):
            all_tids |= set((per_method_by_seed.get(m, {}).get(sd) or {}).keys())
        for tid in sorted(all_tids):
            m3 = (per_method_by_seed.get("M3", {}).get(sd) or {}).get(tid) or {}
            m4 = (per_method_by_seed.get("M4", {}).get(sd) or {}).get(tid) or {}
            m5 = (per_method_by_seed.get("M5", {}).get(sd) or {}).get(tid) or {}
            md = str(m3.get("mode") or m4.get("mode") or m5.get("mode") or "unknown")
            task_rows.append(
                {
                    "seed": sd,
                    "task_id": tid,
                    "mode": md,
                    "m3_ok": int(bool(m3.get("ok"))),
                    "m3_rounds": float(m3.get("rounds") or 0.0),
                    "m3_llm_calls": float(m3.get("llm_calls") or 0.0),
                    "m3_policy_edit": float(m3.get("policy_edit") or 0.0),
                    "m4_ok": int(bool(m4.get("ok"))),
                    "m4_rounds": float(m4.get("rounds") or 0.0),
                    "m4_llm_calls": float(m4.get("llm_calls") or 0.0),
                    "m4_policy_edit": float(m4.get("policy_edit") or 0.0),
                    "m5_ok": int(bool(m5.get("ok"))),
                    "m5_rounds": float(m5.get("rounds") or 0.0),
                    "m5_llm_calls": float(m5.get("llm_calls") or 0.0),
                    "m5_policy_edit": float(m5.get("policy_edit") or 0.0),
                }
            )

    _write_csv(
        out_dir / "summary_methods.csv",
        summary_rows,
        fieldnames=[
            "method",
            "seeds",
            "ok",
            "tasks",
            "success_rate",
            "wilson95_lo",
            "wilson95_hi",
            "avg_rounds",
            "avg_llm_calls",
            "avg_policy_edit",
            "stable_fail_3of3",
            "stable_fail_2plus",
        ],
    )

    _write_csv(
        out_dir / "significance_tests_by_mode.csv",
        sig_mode_rows,
        fieldnames=[
            "mode",
            "method_a",
            "method_b",
            "paired_b_a_ok_b_fail",
            "paired_c_a_fail_b_ok",
            "paired_both_ok",
            "paired_both_fail",
            "mcnemar_p_success",
            "rounds_diff_mean_b_minus_a",
            "rounds_sign_pos",
            "rounds_sign_neg",
            "rounds_sign_n",
            "rounds_sign_p",
            "llm_calls_diff_mean_b_minus_a",
            "llm_calls_sign_pos",
            "llm_calls_sign_neg",
            "llm_calls_sign_n",
            "llm_calls_sign_p",
            "policy_edit_diff_mean_b_minus_a",
            "policy_edit_sign_pos",
            "policy_edit_sign_neg",
            "policy_edit_sign_n",
            "policy_edit_sign_p",
        ],
    )

    _write_csv(
        out_dir / "task_level.csv",
        task_rows,
        fieldnames=[
            "seed",
            "task_id",
            "mode",
            "m3_ok",
            "m3_rounds",
            "m3_llm_calls",
            "m3_policy_edit",
            "m4_ok",
            "m4_rounds",
            "m4_llm_calls",
            "m4_policy_edit",
            "m5_ok",
            "m5_rounds",
            "m5_llm_calls",
            "m5_policy_edit",
        ],
    )
    _write_csv(
        out_dir / "mode_breakdown.csv",
        mode_rows,
        fieldnames=["method", "mode", "ok", "tasks", "success_rate", "wilson95_lo", "wilson95_hi"],
    )
    _write_csv(
        out_dir / "significance_tests.csv",
        sig_rows,
        fieldnames=[
            "method_a",
            "method_b",
            "paired_b_a_ok_b_fail",
            "paired_c_a_fail_b_ok",
            "paired_both_ok",
            "paired_both_fail",
            "mcnemar_p_success",
            "delta_success_rate_a_minus_b",
            "rounds_diff_mean_b_minus_a",
            "rounds_sign_pos",
            "rounds_sign_neg",
            "rounds_sign_n",
            "rounds_sign_p",
            "llm_calls_diff_mean_b_minus_a",
            "llm_calls_sign_pos",
            "llm_calls_sign_neg",
            "llm_calls_sign_n",
            "llm_calls_sign_p",
            "policy_edit_diff_mean_b_minus_a",
            "policy_edit_sign_pos",
            "policy_edit_sign_neg",
            "policy_edit_sign_n",
            "policy_edit_sign_p",
        ],
    )

    (out_dir / "stable_failures.json").write_text(
        json.dumps(stable_failures, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote {out_dir / 'summary_methods.csv'}")
    print(f"Wrote {out_dir / 'mode_breakdown.csv'}")
    print(f"Wrote {out_dir / 'significance_tests.csv'}")
    print(f"Wrote {out_dir / 'significance_tests_by_mode.csv'}")
    print(f"Wrote {out_dir / 'task_level.csv'}")
    print(f"Wrote {out_dir / 'stable_failures.json'}")


if __name__ == "__main__":
    main()
