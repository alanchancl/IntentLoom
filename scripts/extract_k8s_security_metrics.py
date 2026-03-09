from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from intentloom.compiler import compile_intent
from intentloom.parser import parse_intent

from intentloom.adapters.k8s_adapter import build_k8s_graph
from intentloom.adapters.k8s_adapter import find_path
from intentloom.adapters.k8s_adapter import k8s_semantics


@dataclass(frozen=True)
class IsolateSpec:
    a: str
    b: str
    port: Optional[int]
    protocol: Optional[str]


def _canon(x: Any) -> str:
    return str(x).strip().lower() if x is not None else ""


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_tasks(tasks_path: Path) -> Dict[str, Dict[str, Any]]:
    obj = _read_json(tasks_path)
    if not isinstance(obj, list):
        raise RuntimeError(f"tasks must be a JSON list: {tasks_path}")
    out: Dict[str, Dict[str, Any]] = {}
    for t in obj:
        if not isinstance(t, dict):
            continue
        tid = t.get("id") or t.get("task_id")
        if isinstance(tid, str) and tid:
            out[tid] = t
    return out


def _parse_semantics_by_label(spec: Optional[str]) -> Dict[str, str]:
    if not isinstance(spec, str) or not spec.strip():
        return {}
    out: Dict[str, str] = {}
    for part in (p.strip() for p in spec.split(",")):
        if not part:
            continue
        if "=" not in part:
            raise ValueError("--semantics_by_label must be comma-separated label=option_a|option_b|option_c")
        label, sem = part.split("=", 1)
        label = label.strip()
        sem = sem.strip()
        if not label or not sem:
            continue
        out[label] = sem
    return out


def _extract_isolate_specs(intent_text: str) -> List[IsolateSpec]:
    node = parse_intent(intent_text)
    compiled = compile_intent(node)
    out: List[IsolateSpec] = []
    for c in compiled.constraints:
        if c.name != "Isolate":
            continue
        intent = c.intent if isinstance(c.intent, dict) else {}
        params = intent.get("params", {}) if isinstance(intent, dict) else {}
        if not isinstance(params, dict):
            params = {}
        a = str(params.get("a") or "")
        b = str(params.get("b") or "")
        port = params.get("port")
        protocol = params.get("protocol")
        port_i: Optional[int] = None
        if isinstance(port, int):
            port_i = int(port)
        elif isinstance(port, str) and port.strip().isdigit():
            port_i = int(port.strip())
        proto_s = str(protocol).strip() if isinstance(protocol, str) and str(protocol).strip() else None
        if a and b:
            out.append(IsolateSpec(a=a, b=b, port=port_i, protocol=proto_s))
    return out


def _match_labels(labels: Any, selector: Any) -> bool:
    if not selector:
        return True
    if not isinstance(labels, dict) or not isinstance(selector, dict):
        return False
    for k, v in selector.items():
        if labels.get(k) != v:
            return False
    return True


def _with_semantics(cfg: Dict[str, Any], semantics: Optional[str]) -> Dict[str, Any]:
    if not semantics:
        return cfg
    if not isinstance(cfg, dict):
        return cfg
    k8s = cfg.get("k8s")
    k8s2: Dict[str, Any] = dict(k8s) if isinstance(k8s, dict) else {}
    k8s2["semantics"] = str(semantics)
    out = dict(cfg)
    out["k8s"] = k8s2
    return out


def _select_pod_nodes_for_entity(cfg: Dict[str, Any], ent: str) -> List[str]:
    ent_c = _canon(ent)
    k8s = cfg.get("k8s") if isinstance(cfg, dict) else None
    pods = k8s.get("pods") if isinstance(k8s, dict) else None
    services = k8s.get("services") if isinstance(k8s, dict) else None
    if not isinstance(pods, list):
        pods = []
    if not isinstance(services, list):
        services = []

    def _pod_node(pod_name: str) -> str:
        n = _canon(pod_name)
        return f"pod:{n}" if n else ""

    if ent_c.startswith("pod:"):
        name = ent_c.split(":", 1)[1]
        if name:
            return [_pod_node(name)]
        return []

    if ent_c.startswith("ns:"):
        ns = ent_c.split(":", 1)[1]
        if not ns:
            return []
        out: List[str] = []
        for p in pods:
            if not isinstance(p, dict):
                continue
            if _canon(p.get("namespace")) != ns:
                continue
            pn = _pod_node(str(p.get("name") or ""))
            if pn:
                out.append(pn)
        return out

    if ent_c.startswith("svc:"):
        svc_name = ent_c.split(":", 1)[1]
        if not svc_name:
            return []
        for s in services:
            if not isinstance(s, dict):
                continue
            if _canon(s.get("name")) != svc_name:
                continue
            ns = _canon(s.get("namespace"))
            selector = s.get("selector") if isinstance(s.get("selector"), dict) else {}
            out: List[str] = []
            for p in pods:
                if not isinstance(p, dict):
                    continue
                if ns and _canon(p.get("namespace")) != ns:
                    continue
                if not _match_labels(p.get("labels") or {}, selector):
                    continue
                pn = _pod_node(str(p.get("name") or ""))
                if pn:
                    out.append(pn)
            return out
        return []

    return []


def _count_forbidden_reachable_pairs(cfg: Dict[str, Any], specs: List[IsolateSpec]) -> int:
    if not specs:
        return 0
    directed = k8s_semantics(cfg) == "option_c"
    g = build_k8s_graph(cfg)
    total = 0
    for sp in specs:
        a_pods = _select_pod_nodes_for_entity(cfg, sp.a)
        b_pods = _select_pod_nodes_for_entity(cfg, sp.b)
        if not a_pods or not b_pods:
            continue
        for u in a_pods:
            for v in b_pods:
                p = find_path(
                    g.links,
                    u,
                    v,
                    edge_reasons=g.edge_reasons,
                    port=sp.port,
                    protocol=sp.protocol,
                    directed=directed,
                )
                if p is None and directed:
                    p = find_path(
                        g.links,
                        v,
                        u,
                        edge_reasons=g.edge_reasons,
                        port=sp.port,
                        protocol=sp.protocol,
                        directed=directed,
                    )
                if p is not None:
                    total += 1
    return int(total)


def _has_over_permissive_ports(cfg: Dict[str, Any]) -> Optional[bool]:
    # Only meaningful under ports-aware semantics.
    if k8s_semantics(cfg) == "option_a":
        return None
    ps = cfg.get("policies") if isinstance(cfg, dict) else None
    if not isinstance(ps, list):
        return False
    for p in ps:
        if not isinstance(p, dict):
            continue
        action = _canon(p.get("action") or "allow")
        if action != "allow":
            continue
        ports = p.get("ports") if "ports" in p else None
        if ports is None:
            return True
        if isinstance(ports, list) and len(ports) == 0:
            return True
    return False


def _seed_dirs(root: Path, seeds: Optional[List[int]]) -> List[Path]:
    xs = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("seed_")]
    if seeds:
        keep = {f"seed_{int(s)}" for s in seeds}
        xs = [p for p in xs if p.name in keep]
    return sorted(xs, key=lambda p: p.name)


def _mean(xs: List[float]) -> float:
    return float(st.mean(xs)) if xs else 0.0


def _load_seed_summary(seed_dir: Path) -> Dict[str, Any]:
    p = seed_dir / "summary.json"
    if not p.exists():
        raise RuntimeError(f"Missing summary.json: {p}")
    obj = _read_json(p)
    if not isinstance(obj, dict):
        raise RuntimeError(f"Invalid summary.json: {p}")
    return obj


def _parse_root_specs(specs: str) -> List[Tuple[str, Path]]:
    out: List[Tuple[str, Path]] = []
    for part in (p.strip() for p in specs.split(",")):
        if not part:
            continue
        if "=" not in part:
            raise ValueError("--roots must be comma-separated label=path pairs")
        label, path_s = part.split("=", 1)
        label = label.strip()
        path = Path(path_s.strip())
        if not label:
            raise ValueError("Empty label in --roots")
        out.append((label, path))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument(
        "--roots",
        required=True,
        help="Comma-separated label=path pairs. Each path must contain seed_*/summary.json",
    )
    ap.add_argument(
        "--seeds",
        default=None,
        help="Optional comma-separated seed numbers to include (e.g., 1,2,3). Default: all seed_* dirs.",
    )
    ap.add_argument(
        "--semantics_by_label",
        default=None,
        help="Optional comma-separated mapping label=option_a|option_b|option_c to override semantics when computing reachability.",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tasks_path = Path(args.tasks)
    out_path = Path(args.out)
    root_specs = _parse_root_specs(str(args.roots))
    semantics_by_label = _parse_semantics_by_label(args.semantics_by_label)

    seeds: Optional[List[int]] = None
    if isinstance(args.seeds, str) and args.seeds.strip():
        seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    tasks_by_id = _load_tasks(tasks_path)

    isolate_specs_by_task: Dict[str, List[IsolateSpec]] = {}
    for tid, t in tasks_by_id.items():
        intent_text = t.get("intent")
        if not isinstance(intent_text, str) or not intent_text.strip():
            isolate_specs_by_task[tid] = []
            continue
        isolate_specs_by_task[tid] = _extract_isolate_specs(intent_text)

    # Cache: (task_id, semantics) -> initial forbidden pairs
    initial_pairs_cache: Dict[Tuple[str, str], int] = {}

    rows: List[Dict[str, Any]] = []

    for label, root in root_specs:
        sem_override = semantics_by_label.get(label)
        seed_dirs = _seed_dirs(root, seeds)
        if not seed_dirs:
            raise RuntimeError(f"No seed_* dirs found under {root}")

        n_runs = 0
        n_ok = 0
        n_isolate_runs = 0
        n_isolate_ok = 0

        init_pairs: List[int] = []
        final_pairs: List[int] = []

        portsnull_flags: List[bool] = []

        for sd in seed_dirs:
            summary = _load_seed_summary(sd)
            results = summary.get("results")
            if not isinstance(results, list):
                continue
            for r in results:
                if not isinstance(r, dict):
                    continue
                tid = r.get("task_id")
                if not isinstance(tid, str) or not tid:
                    continue
                ok = bool(r.get("ok"))
                cfg_f = r.get("final_cfg")
                if not isinstance(cfg_f, dict):
                    continue

                n_runs += 1
                n_ok += 1 if ok else 0

                specs = isolate_specs_by_task.get(tid, [])
                if specs:
                    n_isolate_runs += 1
                    n_isolate_ok += 1 if ok else 0
                    t = tasks_by_id.get(tid, {})
                    cfg0 = t.get("initial_cfg") if isinstance(t, dict) else None
                    if isinstance(cfg0, dict):
                        sem0 = sem_override or k8s_semantics(cfg0)
                        key = (tid, str(sem0))
                        if key not in initial_pairs_cache:
                            initial_pairs_cache[key] = _count_forbidden_reachable_pairs(
                                _with_semantics(cfg0, sem0), specs
                            )
                        init_pairs.append(int(initial_pairs_cache.get(key, 0)))
                    else:
                        init_pairs.append(0)

                    sem_f = sem_override or k8s_semantics(cfg_f)
                    final_pairs.append(
                        int(_count_forbidden_reachable_pairs(_with_semantics(cfg_f, sem_f), specs))
                    )

                pn = _has_over_permissive_ports(_with_semantics(cfg_f, sem_override) if sem_override else cfg_f)
                if pn is not None:
                    portsnull_flags.append(bool(pn))

        row = {
            "label": label,
            "root": str(root),
            "seeds": ",".join(sd.name.replace("seed_", "") for sd in seed_dirs),
            "n_runs": n_runs,
            "success_rate": (n_ok / n_runs if n_runs else 0.0),
            "n_isolate_runs": n_isolate_runs,
            "isolate_success_rate": (n_isolate_ok / n_isolate_runs if n_isolate_runs else 0.0),
            "attack_pairs_initial_mean": _mean([float(x) for x in init_pairs]),
            "attack_pairs_final_mean": _mean([float(x) for x in final_pairs]),
            "attack_pairs_reduction": _mean([float(x) for x in init_pairs]) - _mean([float(x) for x in final_pairs]),
            "final_overperm_portsnull_frac": (
                (sum(1 for x in portsnull_flags if x) / len(portsnull_flags)) if portsnull_flags else ""
            ),
        }
        rows.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
