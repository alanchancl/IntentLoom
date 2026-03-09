from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from intentloom.compiler import Constraint
from intentloom.compiler import compile_intent
from intentloom.parser import parse_intent
from intentloom.symbolicai_verifier import SymbolicAIContractVerifier
from intentloom.verifier import Verifier

from intentloom.adapters.k8s_verifier import K8sPythonVerifier
from intentloom.adapters.k8s_verifier import K8sSymbolicAIContractVerifier


@dataclass(frozen=True)
class TaskCase:
    task_id: str
    intent_text: str
    cfg: Dict[str, Any]


def _load_tasks(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _infer_domain(tasks: List[Dict[str, Any]]) -> str:
    for t in tasks:
        cfg = t.get("initial_cfg")
        if isinstance(cfg, dict) and "k8s" in cfg:
            return "k8s"
    return "netconfeval"


def _extract_task_case(
    t: Dict[str, Any], *, domain: str, cfg_mode: str, final_cfg_by_id: Optional[Dict[str, Any]]
) -> Optional[TaskCase]:
    task_id = str(t.get("id") or t.get("task_id") or "")
    if not task_id:
        return None

    intent_text = t.get("intent")
    if not isinstance(intent_text, str) or not intent_text.strip():
        return None

    cfg: Optional[Dict[str, Any]] = None
    if cfg_mode == "initial":
        if domain == "k8s":
            cfg0 = t.get("initial_cfg")
            if isinstance(cfg0, dict):
                cfg = cfg0
        else:
            llm = t.get("llm")
            if isinstance(llm, dict):
                cfg0 = llm.get("initial_cfg")
                if isinstance(cfg0, dict):
                    cfg = cfg0
            if cfg is None:
                cfg0 = t.get("initial_cfg")
                if isinstance(cfg0, dict):
                    cfg = cfg0
    elif cfg_mode == "final":
        if isinstance(final_cfg_by_id, dict) and task_id in final_cfg_by_id:
            cfg0 = final_cfg_by_id[task_id]
            if isinstance(cfg0, dict):
                cfg = cfg0
    else:
        raise ValueError(f"Unknown cfg_mode: {cfg_mode}")

    if not isinstance(cfg, dict):
        return None

    return TaskCase(task_id=task_id, intent_text=intent_text, cfg=cfg)


def _load_final_cfg_map(summary_path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not summary_path:
        return None
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            s = json.load(f)
    except Exception:
        return None
    out: Dict[str, Any] = {}
    for r in s.get("results", []) if isinstance(s, dict) else []:
        if not isinstance(r, dict):
            continue
        tid = r.get("task_id")
        cfg = r.get("final_cfg")
        if isinstance(tid, str) and tid and isinstance(cfg, dict):
            out[tid] = cfg
    return out


def _percentile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if q <= 0:
        return float(sorted_vals[0])
    if q >= 1:
        return float(sorted_vals[-1])
    idx = int(round(q * (len(sorted_vals) - 1)))
    idx = max(0, min(len(sorted_vals) - 1, idx))
    return float(sorted_vals[idx])


def _summarize_ms(times_ms: List[float]) -> Dict[str, float]:
    if not times_ms:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
    xs = sorted(times_ms)
    return {
        "mean_ms": float(statistics.mean(xs)),
        "p50_ms": _percentile(xs, 0.50),
        "p95_ms": _percentile(xs, 0.95),
    }


def _count_contract_exception(vr_dict: Dict[str, Any]) -> Tuple[bool, int, int]:
    violations = vr_dict.get("violations") if isinstance(vr_dict, dict) else None
    if not isinstance(violations, list):
        return False, 0, 0

    any_exc = False
    with_exc = 0
    total = 0

    for v in violations:
        if not isinstance(v, dict):
            continue
        total += 1
        meta = v.get("meta")
        exc = meta.get("contract_exception") if isinstance(meta, dict) else None
        if isinstance(exc, str) and exc.strip():
            any_exc = True
            with_exc += 1

    return any_exc, with_exc, total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--domain", choices=["auto", "netconfeval", "k8s"], default="auto")
    ap.add_argument("--verifier", choices=["python", "symbolicai"], required=True)
    ap.add_argument("--cfg_mode", choices=["initial", "final"], default="initial")
    ap.add_argument("--final_summary", default=None)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tasks_raw = _load_tasks(args.tasks)
    domain = _infer_domain(tasks_raw) if args.domain == "auto" else args.domain

    final_cfg_by_id = _load_final_cfg_map(args.final_summary) if args.cfg_mode == "final" else None

    cases: List[TaskCase] = []
    for t in tasks_raw:
        if not isinstance(t, dict):
            continue
        tc = _extract_task_case(
            t, domain=domain, cfg_mode=args.cfg_mode, final_cfg_by_id=final_cfg_by_id
        )
        if tc is not None:
            cases.append(tc)

    if domain == "k8s":
        verifier_obj = K8sPythonVerifier() if args.verifier == "python" else K8sSymbolicAIContractVerifier()
    else:
        verifier_obj = Verifier() if args.verifier == "python" else SymbolicAIContractVerifier()

    per_task_ms: List[float] = []
    tasks_with_exc = 0
    violations_with_exc = 0
    violations_total = 0

    for tc in cases:
        intent_node = parse_intent(tc.intent_text)
        compiled = compile_intent(intent_node)
        constraints: List[Constraint] = compiled.constraints

        for _ in range(max(0, int(args.warmup))):
            vr_w = verifier_obj.verify(tc.cfg, constraints)
            _ = vr_w.to_dict()

        start_any = time.perf_counter()
        for _ in range(max(1, int(args.repeats))):
            t0 = time.perf_counter()
            vr = verifier_obj.verify(tc.cfg, constraints)
            t1 = time.perf_counter()
        end_any = time.perf_counter()

        task_ms = ((end_any - start_any) / max(1, int(args.repeats))) * 1000.0
        per_task_ms.append(task_ms)

        vr_dict = vr.to_dict()
        any_exc, with_exc, total = _count_contract_exception(vr_dict)
        if any_exc:
            tasks_with_exc += 1
        violations_with_exc += with_exc
        violations_total += total

    timing = _summarize_ms(per_task_ms)
    out = {
        "tasks_path": args.tasks,
        "domain": domain,
        "verifier": args.verifier,
        "cfg_mode": args.cfg_mode,
        "n_tasks": len(cases),
        "repeats": int(args.repeats),
        "warmup": int(args.warmup),
        "timing_ms_per_verify": timing,
        "task_level_exception_density": (tasks_with_exc / len(cases) if cases else 0.0),
        "violation_level_exception_density": (
            (violations_with_exc / violations_total) if violations_total else 0.0
        ),
        "violations_per_task_mean": (violations_total / len(cases) if cases else 0.0),
    }

    out_text = json.dumps(out, indent=2, ensure_ascii=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out_text)
    else:
        print(out_text)


if __name__ == "__main__":
    main()
