import os
import sys
import json
import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from intentloom import parse_intent
from intentloom.compiler import canonicalize_node, compile_intent
from intentloom.adapters.k8s_verifier import K8sPythonVerifier


def _canon(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip().lower()


def _rand_choice(rng: random.Random, xs: List[Any]) -> Any:
    return xs[int(rng.random() * len(xs))]


def _mk_pod(*, name: str, namespace: str, labels: Dict[str, str]) -> Dict[str, Any]:
    return {"name": name, "namespace": namespace, "labels": dict(labels)}


def _mk_service(*, name: str, namespace: str, selector: Dict[str, str]) -> Dict[str, Any]:
    return {"name": name, "namespace": namespace, "selector": dict(selector)}


def _mk_policy(
    *,
    name: str,
    src_namespace: Optional[str],
    src_selector: Optional[Dict[str, str]],
    dst_namespace: Optional[str],
    dst_selector: Optional[Dict[str, str]],
    ports: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "name": name,
        "action": "allow",
        "from": {"namespace": src_namespace, "podSelector": src_selector},
        "to": {"namespace": dst_namespace, "podSelector": dst_selector},
    }
    if ports is not None:
        out["ports"] = ports
    return out


def _find_pod(pods: List[Dict[str, Any]], name: str) -> Dict[str, Any]:
    for p in pods:
        if isinstance(p, dict) and _canon(p.get("name")) == _canon(name):
            return p
    return {}


def _find_svc(services: List[Dict[str, Any]], name: str) -> Dict[str, Any]:
    for s in services:
        if isinstance(s, dict) and _canon(s.get("name")) == _canon(name):
            return s
    return {}


def _select_pods_by_selector(
    pods: List[Dict[str, Any]], *, namespace: str, selector: Dict[str, str]
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in pods:
        if not isinstance(p, dict):
            continue
        if _canon(p.get("namespace")) != _canon(namespace):
            continue
        labels = p.get("labels") if isinstance(p.get("labels"), dict) else {}
        ok = True
        for k, v in selector.items():
            if labels.get(k) != v:
                ok = False
                break
        if ok:
            out.append(p)
    return out


def _node_namespace(cfg: Dict[str, Any], node: str) -> Optional[str]:
    s = _canon(node)
    if not s:
        return None
    if s.startswith("ns:"):
        return s.split(":", 1)[1]
    if s.startswith("pod:"):
        name = s.split(":", 1)[1]
        pods = ((cfg.get("k8s") or {}).get("pods") or []) if isinstance(cfg.get("k8s"), dict) else []
        if isinstance(pods, list):
            p = _find_pod(pods, name)
            ns = p.get("namespace") if isinstance(p, dict) else None
            return _canon(ns) if ns is not None else None
        return None
    if s.startswith("svc:"):
        name = s.split(":", 1)[1]
        svcs = ((cfg.get("k8s") or {}).get("services") or []) if isinstance(cfg.get("k8s"), dict) else []
        if isinstance(svcs, list):
            svc = _find_svc(svcs, name)
            ns = svc.get("namespace") if isinstance(svc, dict) else None
            return _canon(ns) if ns is not None else None
        return None
    return None


def _extract_primitives(intent_text: str) -> List[Tuple[str, str, str, Optional[int], Optional[str]]]:
    node = parse_intent(intent_text)
    compiled = compile_intent(node)
    out: List[Tuple[str, str, str, Optional[int], Optional[str]]] = []
    for c in compiled.constraints:
        name = str(c.name)
        params = c.intent.get("params") if isinstance(c.intent, dict) else None
        if not isinstance(params, dict):
            continue
        a = params.get("a")
        b = params.get("b")
        if name in {"Isolate", "Reach"}:
            port = params.get("port")
            protocol = params.get("protocol")
            port_i = int(port) if isinstance(port, int) or (isinstance(port, str) and str(port).strip().isdigit()) else None
            proto_s = str(protocol).strip().lower() if isinstance(protocol, str) and str(protocol).strip() else None
            out.append((name, canonicalize_node(a), canonicalize_node(b), port_i, proto_s))
    return out


def _has_direct_intent_conflict(intent_text: str, cfg_for_ns_lookup: Dict[str, Any]) -> bool:
    prims = _extract_primitives(intent_text)
    reach = {(a, b, p, pr) for (n, a, b, p, pr) in prims if n == "Reach"}
    isolate = {(a, b, p, pr) for (n, a, b, p, pr) in prims if n == "Isolate"}

    def _ports_overlap(p1: Optional[int], pr1: Optional[str], p2: Optional[int], pr2: Optional[str]) -> bool:
        if p1 is None or p2 is None:
            return True
        if int(p1) != int(p2):
            return False
        if pr1 is None or pr2 is None:
            return True
        return str(pr1).strip().lower() == str(pr2).strip().lower()

    for (a, b, p, pr) in list(reach):
        for (x, y, p2, pr2) in list(isolate):
            if not ((a, b) == (x, y) or (a, b) == (y, x)):
                continue
            if _ports_overlap(p, pr, p2, pr2):
                return True

    ns_isolates = []
    for (n, a, b, p, pr) in prims:
        if n != "Isolate":
            continue
        if a.startswith("ns:") and b.startswith("ns:"):
            na = a.split(":", 1)[1]
            nb = b.split(":", 1)[1]
            ns_isolates.append((_canon(na), _canon(nb), p, pr))

    for (n, a, b, p, pr) in prims:
        if n != "Reach":
            continue
        na = _node_namespace(cfg_for_ns_lookup, a)
        nb = _node_namespace(cfg_for_ns_lookup, b)
        if na is None or nb is None:
            continue
        if na == nb:
            continue
        for x, y, p2, pr2 in ns_isolates:
            if {na, nb} == {x, y} and _ports_overlap(p, pr, p2, pr2):
                return True

    return False


def _svc_has_backend(cfg: Dict[str, Any], svc_name: str) -> bool:
    k8s = cfg.get("k8s") if isinstance(cfg, dict) else None
    if not isinstance(k8s, dict):
        return False
    pods = k8s.get("pods")
    svcs = k8s.get("services")
    if not isinstance(pods, list) or not isinstance(svcs, list):
        return False

    svc = _find_svc(svcs, svc_name)
    if not svc:
        return False

    ns = svc.get("namespace")
    sel = svc.get("selector") if isinstance(svc.get("selector"), dict) else {}
    if not isinstance(ns, str) or not sel:
        return False
    matched = _select_pods_by_selector(pods, namespace=ns, selector=sel)
    return len(matched) > 0


@dataclass
class Task:
    id: str
    intent: str
    initial_prompt: Dict[str, Any]
    initial_cfg: Dict[str, Any]
    llm: Dict[str, Any]
    max_rounds: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "intent": self.intent,
            "initial_prompt": self.initial_prompt,
            "initial_cfg": self.initial_cfg,
            "llm": self.llm,
            "max_rounds": self.max_rounds,
        }


def _new_name(rng: random.Random, used: set, prefix: str, pool: List[str]) -> str:
    for _ in range(1000):
        base = _rand_choice(rng, pool)
        suf = int(rng.random() * 10_000)
        name = f"{prefix}{base}{suf}"
        if name not in used:
            used.add(name)
            return name
    raise RuntimeError("could not allocate unique name")


def _make_world(rng: random.Random, namespaces: List[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    pod_pool = [
        "frontend",
        "db",
        "cache",
        "api",
        "worker",
        "debug",
        "metrics",
        "auth",
        "gateway",
        "search",
        "billing",
        "orders",
        "users",
        "profile",
        "inventory",
        "payments",
        "admin",
        "report",
        "cron",
        "backup",
    ]
    svc_pool = [
        "db",
        "cache",
        "api",
        "metrics",
        "auth",
        "gateway",
        "search",
        "orders",
        "users",
    ]

    used_pods: set = set()
    used_svcs: set = set()

    pods: List[Dict[str, Any]] = []
    services: List[Dict[str, Any]] = []

    for ns in namespaces:
        pods_per_ns = 3 + int(rng.random() * 3)
        for _ in range(pods_per_ns):
            name = _new_name(rng, used_pods, "", pod_pool)
            labels: Dict[str, str] = {"app": name}
            if rng.random() < 0.4:
                labels["exposure"] = "public"
            pods.append(_mk_pod(name=name, namespace=ns, labels=labels))

    for ns in namespaces:
        svc_cnt = 1 + int(rng.random() * 2)
        ns_pods = [p for p in pods if _canon(p.get("namespace")) == _canon(ns)]
        if not ns_pods:
            continue
        for _ in range(svc_cnt):
            backend = _rand_choice(rng, ns_pods)
            svc_name = _new_name(rng, used_svcs, "", svc_pool)
            selector = {"app": backend.get("labels", {}).get("app")}
            services.append(_mk_service(name=svc_name, namespace=ns, selector=selector))

    return pods, services


def _mk_cfg(
    pods: List[Dict[str, Any]],
    services: List[Dict[str, Any]],
    policies: List[Dict[str, Any]],
    *,
    semantics: Optional[str] = None,
) -> Dict[str, Any]:
    k8s: Dict[str, Any] = {"pods": pods, "services": services}
    if isinstance(semantics, str) and semantics.strip():
        k8s["semantics"] = semantics.strip()
    return {"k8s": k8s, "policies": policies}


def _port_kv(port: int, protocol: str = "tcp") -> str:
    return f"port={int(port)} protocol={str(protocol).strip().lower()}"


def _ports_spec(port: int, protocol: str = "TCP") -> List[Dict[str, Any]]:
    return [{"port": int(port), "protocol": str(protocol).strip().upper()}]


def _verify(cfg: Dict[str, Any], intent_text: str) -> bool:
    verifier = K8sPythonVerifier()
    intent = parse_intent(intent_text)
    compiled = compile_intent(intent)
    vr = verifier.verify(cfg, compiled.constraints)
    return bool(vr.ok)


def _gen_task(
    rng: random.Random,
    *,
    idx: int,
    llm_spec: Dict[str, Any],
    max_rounds: int,
    include_ports: bool,
) -> Optional[Task]:
    ns_pool = ["prod", "dev", "staging", "qa", "test", "ops", "sandbox"]
    namespaces = list({str(_rand_choice(rng, ns_pool)) for _ in range(3)})
    if len(namespaces) < 2:
        namespaces = ["prod", "dev"]

    pods, services = _make_world(rng, namespaces)

    templates = [
        "isolate_ns_and_reach_service",
        "reach_service_label_mismatch",
        "isolate_ns_allow_all_public",
        "reach_pod_missing_policy",
        "reach_and_isolate_pod_level",
    ]
    tmpl = str(_rand_choice(rng, templates))

    n1 = namespaces[0]
    n2 = namespaces[1]

    pods_n1 = [p for p in pods if _canon(p.get("namespace")) == _canon(n1)]
    pods_n2 = [p for p in pods if _canon(p.get("namespace")) == _canon(n2)]
    if not pods_n1 or not pods_n2:
        return None

    policies_target: List[Dict[str, Any]] = []
    intent_text = ""

    port_i = None
    proto_i = "tcp"
    if include_ports:
        port_pool = [80, 443, 8080, 9090, 3306, 5432, 6379]
        port_i = int(_rand_choice(rng, port_pool))

    if tmpl == "isolate_ns_and_reach_service":
        src = _rand_choice(rng, pods_n1)
        svcs_n1 = [s for s in services if _canon(s.get("namespace")) == _canon(n1)]
        if not svcs_n1:
            return None
        svc = _rand_choice(rng, svcs_n1)
        backend_sel = svc.get("selector") if isinstance(svc.get("selector"), dict) else {}
        backend_pods = _select_pods_by_selector(pods, namespace=n1, selector=backend_sel)
        if not backend_pods:
            return None
        backend = backend_pods[0]

        src_sel = {"app": src.get("labels", {}).get("app")}
        dst_sel = {"app": backend.get("labels", {}).get("app")}
        policies_target.append(
            _mk_policy(
                name=f"allow-{_canon(src.get('name'))}-to-svc-{_canon(svc.get('name'))}",
                src_namespace=n1,
                src_selector=src_sel,
                dst_namespace=n1,
                dst_selector=dst_sel,
                ports=_ports_spec(port_i) if port_i is not None else None,
            )
        )
        if port_i is not None:
            intent_text = (
                f"Isolate ns:{n1} ns:{n2} {_port_kv(port_i, proto_i)} "
                f"AND Reach pod:{src.get('name')} svc:{svc.get('name')} {_port_kv(port_i, proto_i)}"
            )
        else:
            intent_text = f"Isolate ns:{n1} ns:{n2} AND Reach pod:{src.get('name')} svc:{svc.get('name')}"

        policies_init = list(policies_target)
        policies_init.append(
            _mk_policy(
                name=f"allow-{_canon(n1)}-to-{_canon(n2)}-any",
                src_namespace=n1,
                src_selector=None,
                dst_namespace=n2,
                dst_selector=None,
                ports=_ports_spec(port_i) if port_i is not None else None,
            )
        )
        if port_i is not None:
            decoy_port = 80 if port_i != 80 else 443
            policies_init.append(
                _mk_policy(
                    name=f"allow-{_canon(n1)}-to-{_canon(n2)}-any-p{decoy_port}",
                    src_namespace=n1,
                    src_selector=None,
                    dst_namespace=n2,
                    dst_selector=None,
                    ports=_ports_spec(decoy_port),
                )
            )

    elif tmpl == "reach_service_label_mismatch":
        src = _rand_choice(rng, pods_n1)
        svcs_n1 = [s for s in services if _canon(s.get("namespace")) == _canon(n1)]
        if not svcs_n1:
            return None
        svc = _rand_choice(rng, svcs_n1)
        backend_sel = svc.get("selector") if isinstance(svc.get("selector"), dict) else {}
        backend_pods = _select_pods_by_selector(pods, namespace=n1, selector=backend_sel)
        if not backend_pods:
            return None
        backend = backend_pods[0]

        src_sel = {"app": src.get("labels", {}).get("app")}
        dst_sel_ok = {"app": backend.get("labels", {}).get("app")}

        policies_target.append(
            _mk_policy(
                name=f"allow-{_canon(src.get('name'))}-to-{_canon(svc.get('name'))}",
                src_namespace=n1,
                src_selector=src_sel,
                dst_namespace=n1,
                dst_selector=dst_sel_ok,
                ports=_ports_spec(port_i) if port_i is not None else None,
            )
        )
        if port_i is not None:
            intent_text = f"Reach pod:{src.get('name')} svc:{svc.get('name')} {_port_kv(port_i, proto_i)}"
        else:
            intent_text = f"Reach pod:{src.get('name')} svc:{svc.get('name')}"

        dst_sel_bad = {"app": f"typo-{dst_sel_ok.get('app')}"}
        policies_init = [
            _mk_policy(
                name=f"allow-{_canon(src.get('name'))}-to-{_canon(svc.get('name'))}-typo",
                src_namespace=n1,
                src_selector=src_sel,
                dst_namespace=n1,
                dst_selector=dst_sel_bad,
                ports=_ports_spec(port_i) if port_i is not None else None,
            )
        ]

    elif tmpl == "isolate_ns_allow_all_public":
        if port_i is not None:
            intent_text = f"Isolate ns:{n1} ns:{n2} {_port_kv(port_i, proto_i)}"
        else:
            intent_text = f"Isolate ns:{n1} ns:{n2}"
        policies_target = []

        policies_init = [
            _mk_policy(
                name="allow-public-anywhere",
                src_namespace="*",
                src_selector={"exposure": "public"},
                dst_namespace="*",
                dst_selector={"exposure": "public"},
                ports=_ports_spec(port_i) if port_i is not None else None,
            )
        ]
        if port_i is not None:
            decoy_port = 80 if port_i != 80 else 443
            policies_init.append(
                _mk_policy(
                    name=f"allow-public-anywhere-p{decoy_port}",
                    src_namespace="*",
                    src_selector={"exposure": "public"},
                    dst_namespace="*",
                    dst_selector={"exposure": "public"},
                    ports=_ports_spec(decoy_port),
                )
            )

    elif tmpl == "reach_pod_missing_policy":
        a = _rand_choice(rng, pods_n1)
        b = _rand_choice(rng, [p for p in pods_n1 if _canon(p.get("name")) != _canon(a.get("name"))] or pods_n1)
        src_sel = {"app": a.get("labels", {}).get("app")}
        dst_sel = {"app": b.get("labels", {}).get("app")}

        policies_target.append(
            _mk_policy(
                name=f"allow-{_canon(a.get('name'))}-to-{_canon(b.get('name'))}",
                src_namespace=n1,
                src_selector=src_sel,
                dst_namespace=n1,
                dst_selector=dst_sel,
                ports=_ports_spec(port_i) if port_i is not None else None,
            )
        )
        if port_i is not None:
            intent_text = f"Reach pod:{a.get('name')} pod:{b.get('name')} {_port_kv(port_i, proto_i)}"
        else:
            intent_text = f"Reach pod:{a.get('name')} pod:{b.get('name')}"

        policies_init = []

    elif tmpl == "reach_and_isolate_pod_level":
        a = _rand_choice(rng, pods_n1)
        b = _rand_choice(rng, [p for p in pods_n1 if _canon(p.get("name")) != _canon(a.get("name"))] or pods_n1)
        c = _rand_choice(rng, pods_n2)

        src_sel = {"app": a.get("labels", {}).get("app")}
        dst_sel = {"app": b.get("labels", {}).get("app")}
        policies_target.append(
            _mk_policy(
                name=f"allow-{_canon(a.get('name'))}-to-{_canon(b.get('name'))}",
                src_namespace=n1,
                src_selector=src_sel,
                dst_namespace=n1,
                dst_selector=dst_sel,
                ports=_ports_spec(port_i) if port_i is not None else None,
            )
        )

        if port_i is not None:
            intent_text = (
                f"Reach pod:{a.get('name')} pod:{b.get('name')} {_port_kv(port_i, proto_i)} "
                f"AND Isolate pod:{a.get('name')} pod:{c.get('name')} {_port_kv(port_i, proto_i)}"
            )
        else:
            intent_text = f"Reach pod:{a.get('name')} pod:{b.get('name')} AND Isolate pod:{a.get('name')} pod:{c.get('name')}"

        policies_init = list(policies_target)
        policies_init.append(
            _mk_policy(
                name=f"allow-{_canon(n1)}-to-{_canon(n2)}-any",
                src_namespace=n1,
                src_selector=None,
                dst_namespace=n2,
                dst_selector=None,
                ports=_ports_spec(port_i) if port_i is not None else None,
            )
        )

        if port_i is not None:
            decoy_port = 80 if port_i != 80 else 443
            policies_init.append(
                _mk_policy(
                    name=f"allow-{_canon(n1)}-to-{_canon(n2)}-any-p{decoy_port}",
                    src_namespace=n1,
                    src_selector=None,
                    dst_namespace=n2,
                    dst_selector=None,
                    ports=_ports_spec(decoy_port),
                )
            )

    else:
        return None

    semantics = "option_b" if include_ports else None
    cfg_target = _mk_cfg(pods, services, policies_target, semantics=semantics)
    if _has_direct_intent_conflict(intent_text, cfg_target):
        return None

    if "svc:" in _canon(intent_text):
        prims = _extract_primitives(intent_text)
        for (n, a, b, _p, _pr) in prims:
            if n != "Reach":
                continue
            if b.startswith("svc:"):
                svc_name = b.split(":", 1)[1]
                if not _svc_has_backend(cfg_target, svc_name):
                    return None

    if not _verify(cfg_target, intent_text):
        return None

    cfg_init = _mk_cfg(pods, services, policies_init, semantics=semantics)
    if _verify(cfg_init, intent_text):
        return None

    task_id = f"k8sgen_{idx:04d}_{tmpl}"
    initial_prompt = {
        "domain": "k8s_networkpolicy",
        "instruction": "Do not change k8s.pods or k8s.services. Only edit policies. Prefer minimal diffs.",
    }
    return Task(
        id=task_id,
        intent=intent_text,
        initial_prompt=initial_prompt,
        initial_cfg=cfg_init,
        llm=dict(llm_spec),
        max_rounds=max_rounds,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=".\\data\\k8s_policy_100_tasks.json")
    ap.add_argument("--max", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_rounds", type=int, default=6)
    ap.add_argument("--include_ports", action="store_true")
    ap.add_argument("--llm", type=str, default="openai", choices=["openai", "rule_based", "mock"])
    ap.add_argument("--model", type=str, default="gpt-4o-mini")
    ap.add_argument("--base_url", type=str, default=None)
    ap.add_argument("--timeout", type=float, default=None)
    ap.add_argument("--drop_conflicts", action="store_true")
    args = ap.parse_args()

    rng = random.Random(int(args.seed))

    llm_spec: Dict[str, Any]
    if args.llm == "openai":
        llm_spec = {"type": "openai", "model": args.model}
        if args.base_url:
            llm_spec["base_url"] = args.base_url
        if args.timeout is not None:
            llm_spec["timeout"] = args.timeout
    elif args.llm == "rule_based":
        llm_spec = {"type": "rule_based"}
    else:
        llm_spec = {"type": "mock", "candidates": []}

    tasks: List[Dict[str, Any]] = []
    attempts = 0
    max_tasks = int(args.max)
    max_rounds = int(args.max_rounds)

    while len(tasks) < max_tasks:
        attempts += 1
        if attempts > max_tasks * 200:
            break
        t = _gen_task(
            rng,
            idx=len(tasks) + 1,
            llm_spec=llm_spec,
            max_rounds=max_rounds,
            include_ports=bool(args.include_ports),
        )
        if t is None:
            if args.drop_conflicts:
                continue
            else:
                continue
        tasks.append(t.to_dict())

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")

    print({"out": str(out_path), "tasks": len(tasks), "attempts": attempts})


if __name__ == "__main__":
    main()
