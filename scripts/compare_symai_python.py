import argparse
import csv
import json
import math
import statistics as st
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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
    return float(min(1.0, max(0.0, p)))


def _exact_sign_permutation_pvalue(diffs: List[float]) -> float:
    n = len(diffs)
    if n <= 0:
        return 1.0
    obs = float(st.mean(diffs))
    vals: List[float] = []
    for mask in range(1 << n):
        xs: List[float] = []
        for i, d in enumerate(diffs):
            xs.append(d if ((mask >> i) & 1) else -d)
        vals.append(float(st.mean(xs)))
    ge = sum(1 for v in vals if abs(v) >= abs(obs))
    return float(ge) / float(len(vals))


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
        name = p.name
        try:
            seeds.append(int(name.split("_", 1)[1]))
        except Exception:
            continue
    return sorted(set(seeds))


def _load_seed_summary(seed_dir: Path) -> Dict[str, Any]:
    return _read_json(seed_dir / "summary.json")


def _seed_metrics(seed_dir: Path, *, tasks_by_id: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    s = _load_seed_summary(seed_dir)
    results = s.get("results")
    if not isinstance(results, list):
        results = []

    ok_list: List[int] = []
    rounds_list: List[float] = []
    edit_list: List[float] = []
    ok_by_task: Dict[str, bool] = {}

    for r in results:
        if not isinstance(r, dict):
            continue
        tid = str(r.get("task_id") or "")
        if not tid:
            continue
        ok = bool(r.get("ok"))
        rounds = float(r.get("rounds") or 0.0)
        final_cfg = r.get("final_cfg") if isinstance(r.get("final_cfg"), dict) else {}
        t = tasks_by_id.get(tid) or {}
        init_cfg = t.get("initial_cfg") if isinstance(t.get("initial_cfg"), dict) else {}
        before = init_cfg.get("policies") if isinstance(init_cfg, dict) else None
        after = (final_cfg.get("policies") if isinstance(final_cfg, dict) else None)
        ed = float(_policy_edit_distance(before, after))

        ok_list.append(1 if ok else 0)
        rounds_list.append(rounds)
        edit_list.append(ed)
        ok_by_task[tid] = ok

    n = len(ok_list)
    return {
        "tasks": n,
        "ok": int(sum(ok_list)),
        "success_rate": (float(sum(ok_list)) / float(n)) if n else 0.0,
        "avg_rounds": float(st.mean(rounds_list)) if rounds_list else 0.0,
        "avg_policy_edit": float(st.mean(edit_list)) if edit_list else 0.0,
        "ok_by_task": ok_by_task,
        "rounds_list": rounds_list,
        "edit_list": edit_list,
    }


def _load_backend_mode(root: Path, *, prefix: str, mode: str) -> Path:
    d = root / f"{prefix}{mode}"
    if not d.exists() or not d.is_dir():
        raise FileNotFoundError(f"Missing directory: {d}")
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=str, required=True)
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--modes", type=str, default="generic,raw_witness,blame_only")
    ap.add_argument("--py_prefix", type=str, default="py_")
    ap.add_argument("--symai_prefix", type=str, default="symai_")
    args = ap.parse_args()

    tasks_by_id = _load_tasks(Path(args.tasks))
    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    modes = [m.strip() for m in str(args.modes).split(",") if m.strip()]

    summary_rows: List[Dict[str, Any]] = []
    sig_rows: List[Dict[str, Any]] = []

    for mode in modes:
        py_dir = _load_backend_mode(root, prefix=str(args.py_prefix), mode=mode)
        sy_dir = _load_backend_mode(root, prefix=str(args.symai_prefix), mode=mode)

        py_seeds = _list_seeds(py_dir)
        sy_seeds = _list_seeds(sy_dir)
        seeds = sorted(set(py_seeds) & set(sy_seeds))
        if not seeds:
            raise RuntimeError(f"No common seeds for mode={mode}: py={py_seeds}, symai={sy_seeds}")

        py_sr: List[float] = []
        py_rounds: List[float] = []
        py_edits: List[float] = []

        sy_sr: List[float] = []
        sy_rounds: List[float] = []
        sy_edits: List[float] = []

        b = 0
        c = 0

        round_diffs_by_seed: List[float] = []
        edit_diffs_by_seed: List[float] = []

        for sd in seeds:
            py_m = _seed_metrics(py_dir / f"seed_{sd}", tasks_by_id=tasks_by_id)
            sy_m = _seed_metrics(sy_dir / f"seed_{sd}", tasks_by_id=tasks_by_id)

            py_sr.append(float(py_m["success_rate"]))
            py_rounds.append(float(py_m["avg_rounds"]))
            py_edits.append(float(py_m["avg_policy_edit"]))

            sy_sr.append(float(sy_m["success_rate"]))
            sy_rounds.append(float(sy_m["avg_rounds"]))
            sy_edits.append(float(sy_m["avg_policy_edit"]))

            round_diffs_by_seed.append(float(sy_m["avg_rounds"]) - float(py_m["avg_rounds"]))
            edit_diffs_by_seed.append(float(sy_m["avg_policy_edit"]) - float(py_m["avg_policy_edit"]))

            py_ok = py_m.get("ok_by_task") if isinstance(py_m.get("ok_by_task"), dict) else {}
            sy_ok = sy_m.get("ok_by_task") if isinstance(sy_m.get("ok_by_task"), dict) else {}
            tids = sorted(set(py_ok.keys()) & set(sy_ok.keys()))
            for tid in tids:
                py_o = bool(py_ok.get(tid))
                sy_o = bool(sy_ok.get(tid))
                if py_o and (not sy_o):
                    b += 1
                if (not py_o) and sy_o:
                    c += 1

        py_sr_mean, py_sr_sd = _mean_sd(py_sr)
        py_r_mean, py_r_sd = _mean_sd(py_rounds)
        py_e_mean, py_e_sd = _mean_sd(py_edits)

        sy_sr_mean, sy_sr_sd = _mean_sd(sy_sr)
        sy_r_mean, sy_r_sd = _mean_sd(sy_rounds)
        sy_e_mean, sy_e_sd = _mean_sd(sy_edits)

        summary_rows.append(
            {
                "backend": "python",
                "mode": mode,
                "seeds": ",".join(str(s) for s in seeds),
                "ok_mean": py_sr_mean * float(len(tasks_by_id)),
                "ok_sd": py_sr_sd * float(len(tasks_by_id)),
                "sr_mean": py_sr_mean,
                "sr_sd": py_sr_sd,
                "avg_rounds_mean": py_r_mean,
                "avg_rounds_sd": py_r_sd,
                "avg_policy_edit_mean": py_e_mean,
                "avg_policy_edit_sd": py_e_sd,
            }
        )
        summary_rows.append(
            {
                "backend": "symbolicai",
                "mode": mode,
                "seeds": ",".join(str(s) for s in seeds),
                "ok_mean": sy_sr_mean * float(len(tasks_by_id)),
                "ok_sd": sy_sr_sd * float(len(tasks_by_id)),
                "sr_mean": sy_sr_mean,
                "sr_sd": sy_sr_sd,
                "avg_rounds_mean": sy_r_mean,
                "avg_rounds_sd": sy_r_sd,
                "avg_policy_edit_mean": sy_e_mean,
                "avg_policy_edit_sd": sy_e_sd,
            }
        )

        sig_rows.append(
            {
                "mode": mode,
                "mcnemar_b_py_ok_sy_fail": b,
                "mcnemar_c_py_fail_sy_ok": c,
                "mcnemar_p_success": _mcnemar_exact_pvalue(b, c),
                "perm_p_rounds": _exact_sign_permutation_pvalue(round_diffs_by_seed),
                "perm_p_policy_edits": _exact_sign_permutation_pvalue(edit_diffs_by_seed),
                "delta_sr_symai_minus_py": sy_sr_mean - py_sr_mean,
                "delta_rounds_symai_minus_py": sy_r_mean - py_r_mean,
                "delta_policy_edits_symai_minus_py": sy_e_mean - py_e_mean,
            }
        )

    _write_csv(
        out / "summary_multiseed.csv",
        summary_rows,
        fieldnames=[
            "backend",
            "mode",
            "seeds",
            "ok_mean",
            "ok_sd",
            "sr_mean",
            "sr_sd",
            "avg_rounds_mean",
            "avg_rounds_sd",
            "avg_policy_edit_mean",
            "avg_policy_edit_sd",
        ],
    )
    _write_csv(
        out / "significance_tests.csv",
        sig_rows,
        fieldnames=[
            "mode",
            "mcnemar_b_py_ok_sy_fail",
            "mcnemar_c_py_fail_sy_ok",
            "mcnemar_p_success",
            "perm_p_rounds",
            "perm_p_policy_edits",
            "delta_sr_symai_minus_py",
            "delta_rounds_symai_minus_py",
            "delta_policy_edits_symai_minus_py",
        ],
    )

    print(f"Wrote {out / 'summary_multiseed.csv'}")
    print(f"Wrote {out / 'significance_tests.csv'}")


if __name__ == "__main__":
    main()
