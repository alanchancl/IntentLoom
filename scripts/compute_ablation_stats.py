import csv
import json
from math import sqrt
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Iterable, Any

# Paths
ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"

DATASETS = {
    "netconfeval_complex50": {
        "generic": ART / "ablation_netconfeval_complex50_symai_generic",
        "raw": ART / "ablation_netconfeval_complex50_symai_raw",
        "blame": ART / "ablation_netconfeval_complex50_symai_blame",
    },
    "k8s20": {
        "generic": ART / "ablation_k8s20_symai_generic",
        "raw": ART / "ablation_k8s20_symai_raw",
        "blame": ART / "ablation_k8s20_symai_blame",
    },
    "netconfeval_111": {
        "generic": ART / "ablation_net111_symai_generic",
        "raw": ART / "ablation_net111_symai_raw",
        "blame": ART / "ablation_net111_symai_blame",
    },
    "k8s_100": {
        "generic": ART / "ablation_k8s100_symai_generic",
        "raw": ART / "ablation_k8s100_symai_raw",
        "blame": ART / "ablation_k8s100_symai_blame",
    },
}


def _prefer_multiseed_dir(p: Path) -> Path:
    if p.is_dir() and any(p.glob("seed_*/summary.json")):
        return p
    alt = p.parent / f"{p.name}_multiseed"
    if alt.exists() and any(alt.glob("seed_*/summary.json")):
        return alt
    return p


def _pearson_corr(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n < 2 or n != len(ys):
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return 0.0
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / sqrt(vx * vy)


def _quantile(sorted_xs: List[int], q: float) -> int:
    if not sorted_xs:
        return 0
    if q <= 0:
        return sorted_xs[0]
    if q >= 1:
        return sorted_xs[-1]
    idx = int(q * (len(sorted_xs) - 1))
    return sorted_xs[idx]


def _iter_seed_dirs(p: Path) -> Iterable[Tuple[Optional[int], Path]]:
    if p.is_file():
        yield None, p.parent
        return
    if p.is_dir() and any(p.glob("seed_*")):
        seed_dirs: List[Tuple[Optional[int], Path]] = []
        for child in sorted(p.glob("seed_*")):
            if not child.is_dir():
                continue
            seed: Optional[int] = None
            if child.name.startswith("seed_"):
                try:
                    seed = int(child.name[len("seed_") :])
                except Exception:
                    seed = None
            if (child / "summary.json").exists():
                seed_dirs.append((seed, child))
        if seed_dirs:
            for seed, sd in seed_dirs:
                yield seed, sd
            return
    if p.is_dir() and (p / "summary.json").exists():
        yield None, p


def _read_first_jsonl_object(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                return json.loads(line)
    except Exception:
        return None
    return None


def _read_max_verification_len(path: Path) -> Optional[int]:
    max_len: Optional[int] = None
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                pr = obj.get("prompt")
                if not isinstance(pr, dict):
                    continue
                ver = pr.get("verification")
                if ver is None:
                    continue
                try:
                    L = len(json.dumps(ver, ensure_ascii=False))
                except Exception:
                    continue
                if max_len is None or L > max_len:
                    max_len = L
    except Exception:
        return None
    return max_len


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


def _read_max_contract_exception_len(path: Path) -> Optional[int]:
    best: Optional[int] = None
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                pr = obj.get("prompt")
                if not isinstance(pr, dict):
                    continue
                ver = pr.get("verification")
                L = _max_contract_exception_len_from_verification(ver)
                if best is None or L > best:
                    best = L
    except Exception:
        return None
    return best


def compute_evidence_length_rows(p: Path) -> List[Dict[str, Any]]:
    p = _prefer_multiseed_dir(p)
    rows: List[Dict[str, Any]] = []
    for seed, seed_dir in _iter_seed_dirs(p):
        summ = seed_dir / "summary.json"
        if not summ.exists():
            continue
        d = json.loads(summ.read_text(encoding="utf-8"))
        for r in d.get("results", []):
            tid = str(r.get("task_id", ""))
            if not tid:
                continue
            ok = bool(r.get("ok"))
            jl = seed_dir / f"{tid}.jsonl"
            max_ver_len = _read_max_verification_len(jl) if jl.exists() else None
            max_exc_len = _read_max_contract_exception_len(jl) if jl.exists() else None
            rows.append(
                {
                    "seed": seed,
                    "task_id": tid,
                    "ok": ok,
                    "max_verification_len": max_ver_len,
                    "max_contract_exception_len": max_exc_len,
                }
            )
    return rows


def summarize_evidence_length(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    xs_all: List[int] = []
    ys_all: List[float] = []
    xs_pos: List[int] = []
    ys_pos: List[float] = []
    for r in rows:
        L = r.get("max_contract_exception_len")
        if L is None:
            continue
        L = int(L)
        y = 1.0 if r.get("ok") else 0.0
        xs_all.append(L)
        ys_all.append(y)
        if L > 0:
            xs_pos.append(L)
            ys_pos.append(y)

    n = len(xs_all)
    sr = (sum(ys_all) / n) if n else 0.0
    n_pos = len(xs_pos)
    sr_pos = (sum(ys_pos) / n_pos) if n_pos else 0.0
    corr_pos = _pearson_corr([float(v) for v in xs_pos], ys_pos) if n_pos else 0.0

    xs_sorted = sorted(xs_pos)
    qs = {k: _quantile(xs_sorted, q) for k, q in [("p50", 0.50), ("p75", 0.75), ("p90", 0.90), ("p95", 0.95), ("p99", 0.99)]}

    def tail_stats(thr: int) -> Tuple[int, float]:
        idx = [i for i, v in enumerate(xs_pos) if v >= thr]
        if not idx:
            return 0, 0.0
        tail_sr = sum(ys_pos[i] for i in idx) / len(idx)
        return len(idx), tail_sr

    tail = {}
    for key in ["p75", "p90", "p95", "p99"]:
        tn, tsr = tail_stats(int(qs[key]))
        tail[f"tail_n_{key}"] = tn
        tail[f"tail_sr_{key}"] = tsr

    return {"n": n, "sr": sr, "n_pos": n_pos, "sr_pos": sr_pos, "corr_pos": corr_pos, **qs, **tail}


def _read_summary_file(path: Path) -> Dict[str, Tuple[bool, int]]:
    d = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, Tuple[bool, int]] = {}
    for r in d.get("results", []):
        tid = str(r.get("task_id", ""))
        ok = bool(r.get("ok"))
        rounds = int(r.get("rounds", 0))
        out[tid] = (ok, rounds)
    return out


def _iter_seed_summary_files(p: Path) -> Iterable[Tuple[Optional[int], Path]]:
    if p.is_file():
        yield None, p
        return
    if p.is_dir():
        seed_files: List[Tuple[Optional[int], Path]] = []
        for child in sorted(p.glob("seed_*")):
            if not child.is_dir():
                continue
            name = child.name
            seed: Optional[int] = None
            if name.startswith("seed_"):
                try:
                    seed = int(name[len("seed_") :])
                except Exception:
                    seed = None
            f = child / "summary.json"
            if f.exists():
                seed_files.append((seed, f))
        if seed_files:
            for seed, f in seed_files:
                yield seed, f
            return

        direct = p / "summary.json"
        if direct.exists():
            yield None, direct


def read_seeded_summaries(p: Path) -> Dict[Optional[int], Dict[str, Tuple[bool, int]]]:
    out: Dict[Optional[int], Dict[str, Tuple[bool, int]]] = {}
    for seed, f in _iter_seed_summary_files(p):
        out[seed] = _read_summary_file(f)
    return out


def wilson_ci(successes: int, n: int, alpha: float = 0.05) -> Tuple[float, float, float]:
    z = 1.959963984540054 if abs(alpha - 0.05) < 1e-9 else 1.959963984540054  # fixed 95%
    if n == 0:
        return 0.0, 0.0, 0.0
    p = successes / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    half = (z/denom) * sqrt((p*(1 - p) + z*z/(4*n)) / n)
    return p, max(0.0, center - half), min(1.0, center + half)


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    from math import comb
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    p = min(1.0, 2 * tail)
    return p


def _wilcoxon_signed_rank_normal(diffs: List[int]) -> Tuple[float, float, float, int]:
    import math
    xs = [abs(x) for x in diffs if x != 0]
    signs = [1 if x > 0 else -1 for x in diffs if x != 0]
    n = len(xs)
    if n == 0:
        return 0.0, 1.0, 0.0, 0
    idx = list(range(n))
    idx.sort(key=lambda i: xs[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[idx[j + 1]] == xs[idx[i]]:
            j += 1
        r_avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[idx[k]] = r_avg
        i = j + 1
    Wpos = sum(r for r, s in zip(ranks, signs) if s > 0)
    mu = n * (n + 1) / 4.0
    sigma2 = n * (n + 1) * (2 * n + 1) / 24.0
    sigma = math.sqrt(sigma2) if sigma2 > 0 else 1.0
    z = (Wpos - mu - 0.5 * (1 if Wpos > mu else -1)) / sigma
    def _phi(zv: float) -> float:
        return 0.5 * (1.0 + math.erf(zv / math.sqrt(2.0)))
    p = 2.0 * (1.0 - _phi(abs(z)))
    r = abs(z) / math.sqrt(n)
    return z, max(0.0, min(1.0, p)), r, n


def paired_effects(a: Dict[str, Tuple[bool, int]], b: Dict[str, Tuple[bool, int]]):
    tids = sorted(set(a.keys()) & set(b.keys()))
    b_only = 0  # not ok in a, ok in b
    a_only = 0  # ok in a, not ok in b
    d_rounds: List[int] = []  # rounds_a - rounds_b
    for t in tids:
        ok_a, r_a = a[t]
        ok_b, r_b = b[t]
        if ok_a and not ok_b:
            a_only += 1
        elif ok_b and not ok_a:
            b_only += 1
        if ok_a and ok_b:
            d_rounds.append(int(r_a) - int(r_b))
    n = len(d_rounds)
    if n == 0:
        mean_d = 0.0
        sd_d = 0.0
        d_cohen = 0.0
    else:
        mean_d = sum(d_rounds) / n
        var = sum((x - mean_d) ** 2 for x in d_rounds) / (n - 1) if n > 1 else 0.0
        sd_d = sqrt(var)
        d_cohen = (mean_d / sd_d) if sd_d > 0 else 0.0
    pos = sum(1 for x in d_rounds if x > 0)
    neg = sum(1 for x in d_rounds if x < 0)
    ties = n - pos - neg
    cliff_signed = ((pos - neg) / (pos + neg)) if (pos + neg) > 0 else 0.0
    z_w, p_w, r_w, n_w = _wilcoxon_signed_rank_normal(d_rounds)

    return {
        "b_only": b_only,
        "a_only": a_only,
        "diff_pairs": b_only + a_only,
        "mcnemar_p": mcnemar_exact(b_only, a_only),
        "rounds_n": n,
        "rounds_mean_diff": mean_d,
        "rounds_sd_diff": sd_d,
        "paired_cohen_d": d_cohen,
        "wilcoxon_z": z_w,
        "wilcoxon_p": p_w,
        "wilcoxon_r": r_w,
        "sign_pos": pos,
        "sign_neg": neg,
        "sign_ties": ties,
        "sign_effect": cliff_signed,
    }


def _mean_sd(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    mu = sum(xs) / len(xs)
    if len(xs) < 2:
        return mu, 0.0
    var = sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)
    return mu, sqrt(var)


def _t_crit_975(df: int) -> float:
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
    if df in table:
        return table[df]
    if df >= 30:
        return 1.96
    return 2.0


def mean_ci95_t(xs: List[float]) -> Tuple[float, float, float]:
    mu, sd = _mean_sd(xs)
    n = len(xs)
    if n <= 1:
        return mu, mu, mu
    t = _t_crit_975(n - 1)
    half = t * sd / sqrt(n)
    return mu, mu - half, mu + half


def risk_ratio_ci(success_a: int, n_a: int, success_b: int, n_b: int) -> Tuple[float, float, float]:
    a = float(success_a)
    b = float(success_b)
    na = float(n_a)
    nb = float(n_b)
    if na <= 0 or nb <= 0:
        return 0.0, 0.0, 0.0
    if a <= 0 or b <= 0:
        a += 0.5
        b += 0.5
        na += 1.0
        nb += 1.0
    p_a = a / na
    p_b = b / nb
    rr = p_a / p_b if p_b > 0 else 0.0
    from math import log, exp
    var = (1.0 / a - 1.0 / na) + (1.0 / b - 1.0 / nb)
    se = sqrt(max(0.0, var))
    z = 1.959963984540054
    lo = exp(log(rr) - z * se) if rr > 0 else 0.0
    hi = exp(log(rr) + z * se) if rr > 0 else 0.0
    return rr, lo, hi


def required_n_two_proportions(p1: float, p2: float, *, alpha: float = 0.05, power: float = 0.8) -> int:
    import math
    if p1 == p2:
        return 0
    z_alpha = 1.959963984540054
    z_power = 0.8416212335729143
    if abs(power - 0.9) < 1e-9:
        z_power = 1.2815515655446004
    elif abs(power - 0.95) < 1e-9:
        z_power = 1.6448536269514722
    elif abs(power - 0.8) < 1e-9:
        z_power = 0.8416212335729143
    pbar = 0.5 * (p1 + p2)
    num = (z_alpha * math.sqrt(2 * pbar * (1 - pbar)) + z_power * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    den = (p1 - p2) ** 2
    n = int(math.ceil(num / den))
    return max(0, n)


def main() -> None:
    lines: List[str] = []
    ev_summary_rows: List[Dict[str, Any]] = []
    for name, paths in DATASETS.items():
        modes_seeded = {k: read_seeded_summaries(_prefer_multiseed_dir(p)) for k, p in paths.items()}

        lines.append(f"## Dataset: {name}")
        for mk, seeds in modes_seeded.items():
            seed_keys = sorted(seeds.keys(), key=lambda x: (-1 if x is None else int(x)))
            per_seed_sr: List[float] = []
            per_seed_avg_rounds: List[float] = []
            pooled_tasks = 0
            pooled_ok = 0
            pooled_rounds_sum = 0

            for sd in seed_keys:
                md = seeds[sd]
                n = len(md)
                s = sum(1 for v in md.values() if v[0])
                sr = (s / n) if n else 0.0
                avg_r = (sum(v[1] for v in md.values()) / n) if n else 0.0
                per_seed_sr.append(sr)
                per_seed_avg_rounds.append(avg_r)
                pooled_tasks += n
                pooled_ok += s
                pooled_rounds_sum += sum(v[1] for v in md.values())

            p_pool, lo_pool, hi_pool = wilson_ci(pooled_ok, pooled_tasks)
            avg_rounds_pool = (pooled_rounds_sum / pooled_tasks) if pooled_tasks else 0.0

            if len(per_seed_sr) > 1:
                mu_sr, lo_sr, hi_sr = mean_ci95_t(per_seed_sr)
                mu_r, lo_r, hi_r = mean_ci95_t(per_seed_avg_rounds)
                sd_sr = _mean_sd(per_seed_sr)[1]
                sd_r = _mean_sd(per_seed_avg_rounds)[1]
                lines.append(
                    (
                        f"mode={mk} seeds={len(per_seed_sr)} tasks/seed={pooled_tasks//max(1,len(per_seed_sr))} "
                        f"sr_mean={mu_sr:.3f} sd={sd_sr:.3f} 95%tCI=[{lo_sr:.3f},{hi_sr:.3f}] "
                        f"avg_rounds_mean={mu_r:.2f} sd={sd_r:.2f} 95%tCI=[{lo_r:.2f},{hi_r:.2f}] "
                        f"(pooled_sr={p_pool:.3f} Wilson95=[{lo_pool:.3f},{hi_pool:.3f}] pooled_avg_rounds={avg_rounds_pool:.2f})"
                    )
                )
            else:
                lines.append(
                    f"mode={mk} tasks={pooled_tasks} ok={pooled_ok} sr={p_pool:.3f} 95%CI=[{lo_pool:.3f},{hi_pool:.3f}] avg_rounds={avg_rounds_pool:.2f}"
                )

        def _flatten(seeded: Dict[Optional[int], Dict[str, Tuple[bool, int]]]) -> Dict[str, Tuple[bool, int]]:
            out: Dict[str, Tuple[bool, int]] = {}
            for sd, md in seeded.items():
                for tid, v in md.items():
                    key = f"{sd}:{tid}"
                    out[key] = v
            return out

        flat = {k: _flatten(v) for k, v in modes_seeded.items()}
        pooled_counts = {}
        for mk, md in flat.items():
            pooled_counts[mk] = (len(md), sum(1 for v in md.values() if v[0]))

        def cmp(a_name: str, b_name: str):
            r = paired_effects(flat[a_name], flat[b_name])
            na, sa = pooled_counts[a_name]
            nb, sb = pooled_counts[b_name]
            rr, rr_lo, rr_hi = risk_ratio_ci(sb, nb, sa, na)
            lines.append(
                (
                    f"{a_name} vs {b_name} (pooled over seeds): discordant={r['diff_pairs']} (b_only={r['b_only']}, a_only={r['a_only']}), "
                    f"McNemar exact p={r['mcnemar_p']:.4f}; "
                    f"RR(success {b_name}/{a_name})={rr:.3f} 95%CI=[{rr_lo:.3f},{rr_hi:.3f}]; "
                    f"rounds n={r['rounds_n']} mean(a-b)={r['rounds_mean_diff']:.3f} sd={r['rounds_sd_diff']:.3f} paired_d={r['paired_cohen_d']:.3f} "
                    f"Wilcoxon z={r['wilcoxon_z']:.3f} p={r['wilcoxon_p']:.4f} r={r['wilcoxon_r']:.3f}"
                )
            )

            p_a = (sa / na) if na else 0.0
            p_b = (sb / nb) if nb else 0.0
            n_req = required_n_two_proportions(p_a, p_b, alpha=0.05, power=0.8)
            if n_req > 0:
                lines.append(
                    f"  power(0.8, alpha=0.05 two-sided) required N per group for p={p_a:.3f} vs {p_b:.3f}: ~{n_req} tasks"
                )

        cmp("generic", "raw")
        cmp("blame", "raw")
        cmp("generic", "blame")
        # pairwise comparisons
        lines.append("")

        for mk, p in paths.items():
            ev_rows = compute_evidence_length_rows(p)
            s = summarize_evidence_length(ev_rows)
            ev_summary_rows.append({"dataset": name, "mode": mk, **s})

    out = "\n".join(lines)
    print(out)
    (ART / "ablation_stats.txt").write_text(out, encoding="utf-8")

    ev_csv = ART / "evidence_length_stats.csv"
    ev_tex = ART / "evidence_length_stats.tex"
    ev_fig = ART / "evidence_length_figure.tex"

    ev_summary_rows.sort(key=lambda r: (str(r.get("dataset")), str(r.get("mode"))))

    with ev_csv.open("w", encoding="utf-8", newline="") as f:
        cols = [
            "dataset",
            "mode",
            "n",
            "sr",
            "n_pos",
            "sr_pos",
            "corr_pos",
            "p50",
            "p75",
            "p90",
            "p95",
            "p99",
            "tail_n_p75",
            "tail_sr_p75",
            "tail_n_p90",
            "tail_sr_p90",
            "tail_n_p95",
            "tail_sr_p95",
            "tail_n_p99",
            "tail_sr_p99",
        ]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in ev_summary_rows:
            w.writerow({k: r.get(k) for k in cols})

    def _fmt_float(x: Any, nd: int = 3) -> str:
        try:
            return f"{float(x):.{nd}f}"
        except Exception:
            return "0.000"

    def _fmt_int(x: Any) -> str:
        try:
            return str(int(x))
        except Exception:
            return "0"

    def _latex_escape(s: Any) -> str:
        return str(s).replace("\\", "\\\\").replace("_", "\\_")

    tex_lines: List[str] = []
    tex_lines.append("\\begin{tabular}{llrrrrrrrrrr}")
    tex_lines.append("\\toprule")
    tex_lines.append("Dataset & Mode & $n$ & SR & $n_{w}$ & SR$_{w}$ & $r$ & p90 & p95 & p99 & SR@p90 & SR@p95 \\\\")
    tex_lines.append("\\midrule")
    for r in ev_summary_rows:
        n_pos = int(r.get("n_pos") or 0)
        if n_pos <= 0:
            p90 = p95 = p99 = "--"
            sr90 = sr95 = "--"
            corr_pos = "--"
            sr_pos = "--"
        else:
            p90 = _fmt_int(r.get("p90"))
            p95 = _fmt_int(r.get("p95"))
            p99 = _fmt_int(r.get("p99"))
            sr90 = _fmt_float(r.get("tail_sr_p90"), 3)
            sr95 = _fmt_float(r.get("tail_sr_p95"), 3)
            corr_pos = _fmt_float(r.get("corr_pos"), 3)
            sr_pos = _fmt_float(r.get("sr_pos"), 3)

        tex_lines.append(
            " ".join(
                [
                    _latex_escape(r.get("dataset")),
                    "&",
                    _latex_escape(r.get("mode")),
                    "&",
                    _fmt_int(r.get("n")),
                    "&",
                    _fmt_float(r.get("sr"), 3),
                    "&",
                    _fmt_int(r.get("n_pos")),
                    "&",
                    sr_pos,
                    "&",
                    corr_pos,
                    "&",
                    p90,
                    "&",
                    p95,
                    "&",
                    p99,
                    "&",
                    sr90,
                    "&",
                    sr95,
                    "\\\\",
                ]
            )
        )
    tex_lines.append("\\bottomrule")
    tex_lines.append("\\end{tabular}")
    ev_tex.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")

    raw_rows = [r for r in ev_summary_rows if str(r.get("mode")) == "raw" and int(r.get("n_pos") or 0) > 0]
    raw_rows.sort(key=lambda r: str(r.get("dataset")))
    fig_lines: List[str] = []
    fig_lines.append("\\begin{tikzpicture}[x=1.65cm,y=10cm, font=\\scriptsize]")
    fig_lines.append("  \\def\\ymin{0.0}")
    fig_lines.append("  \\def\\ymax{1.02}")
    fig_lines.append("  \\draw[->] (-0.3,\\ymin) -- (%.1f,\\ymin) node[below] {Dataset};" % (float(max(1, len(raw_rows))) + 0.6))
    fig_lines.append("  \\draw[->] (0,\\ymin-0.02) -- (0,\\ymax) node[left] {Success Rate};")
    fig_lines.append("  \\foreach \\y/\\lab in {0.0/0.0,0.2/0.2,0.4/0.4,0.6/0.6,0.8/0.8,1.0/1.0} {\\draw (-0.05,\\y) -- (0.05,\\y) node[left] {\\lab};}")
    fig_lines.append("  \\def\\bw{0.22}")
    for i, r in enumerate(raw_rows, start=1):
        x = float(i)
        ds = _latex_escape(r.get("dataset"))
        sr_w = float(r.get("sr_pos") or 0.0)
        sr90 = float(r.get("tail_sr_p90") or 0.0)
        fig_lines.append(f"  \\node[below] at ({x},\\ymin) {{{ds}}};")
        fig_lines.append(f"  \\draw[fill=blue!50,draw=blue!70] ({x}-\\bw,\\ymin) rectangle ({x}, {sr_w:.4f});")
        fig_lines.append(f"  \\draw[fill=red!45,draw=red!70] ({x},\\ymin) rectangle ({x}+\\bw, {sr90:.4f});")
    fig_lines.append("  \\node[blue!70,anchor=west] at (0.1,1.0) {raw: SR among witness-bearing tasks (SR$_w$)};")
    fig_lines.append("  \\node[red!70,anchor=west] at (0.1,0.94) {raw: SR in top-10\\% witness-length tail (>=p90)};")
    fig_lines.append("\\end{tikzpicture}")
    ev_fig.write_text("\n".join(fig_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
