import argparse
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from intentloom import parse_intent
from intentloom.compiler import compile_intent
from intentloom.adapters.k8s_verifier import K8sPythonVerifier


@dataclass
class PolicyCandidate:
    name: str
    namespace: str
    direction: str
    src_ns: Optional[str]
    src_selector: Optional[Dict[str, str]]
    src_ipblock: Optional[str]
    dst_ns: Optional[str]
    dst_selector: Optional[Dict[str, str]]
    dst_ipblock: Optional[str]
    ports: Optional[List[Dict[str, Any]]]
    source: Dict[str, Any]


def _canon(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip().lower()


def _safe_labels(sel: Optional[Dict[str, Any]]) -> Optional[Dict[str, str]]:
    if not isinstance(sel, dict):
        return None
    labels = sel.get("matchLabels") if "matchLabels" in sel else sel
    if not isinstance(labels, dict) or not labels:
        return None
    out: Dict[str, str] = {}
    for k, v in labels.items():
        if k is None or v is None:
            continue
        out[str(k)] = str(v)
    return out or None


def _extract_ports(spec: Any) -> Optional[List[Dict[str, Any]]]:
    if spec is None:
        return None
    if not isinstance(spec, list):
        return None
    out: List[Dict[str, Any]] = []
    for it in spec:
        if not isinstance(it, dict):
            continue
        port = it.get("port")
        if port is None:
            continue
        if isinstance(port, str) and not port.strip().isdigit():
            return None
        try:
            p = int(port)
        except Exception:
            continue
        rec: Dict[str, Any] = {"port": p}
        proto = it.get("protocol")
        if isinstance(proto, str) and proto.strip():
            rec["protocol"] = proto.strip().upper()
        out.append(rec)
    return out or None


def _extract_ipblock(spec: Any) -> Optional[str]:
    if not isinstance(spec, dict):
        return None
    cidr = spec.get("cidr")
    if not isinstance(cidr, str) or not cidr.strip():
        return None
    if spec.get("except"):
        return None
    return cidr.strip()


def _load_yaml_docs(path: Path) -> List[Dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    docs: List[Dict[str, Any]] = []
    if re.search(r"^\s*\{\{[-#]", text, flags=re.MULTILINE) or re.search(r"^\s*\{\{\s*[-#]", text, flags=re.MULTILINE):
        return []
    try:
        for doc in yaml.safe_load_all(text):
            if isinstance(doc, dict):
                docs.append(doc)
    except yaml.YAMLError:
        return []
    return docs


def _extract_candidates(docs: List[Dict[str, Any]], source: Dict[str, Any]) -> List[PolicyCandidate]:
    out: List[PolicyCandidate] = []
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        if _canon(doc.get("kind")) != "networkpolicy":
            continue
        meta = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
        spec = doc.get("spec") if isinstance(doc.get("spec"), dict) else {}
        name = str(meta.get("name") or "networkpolicy")
        namespace = str(meta.get("namespace") or "default")

        pod_selector = _safe_labels(spec.get("podSelector"))

        ingress_rules = spec.get("ingress") if isinstance(spec.get("ingress"), list) else []
        egress_rules = spec.get("egress") if isinstance(spec.get("egress"), list) else []

        if ingress_rules:
            for rule in ingress_rules:
                if not isinstance(rule, dict):
                    continue
                ports = _extract_ports(rule.get("ports"))
                if "from" in rule and isinstance(rule.get("from"), list) and len(rule.get("from")) == 0:
                    continue

                froms = rule.get("from") if isinstance(rule.get("from"), list) else None
                if froms is None:
                    out.append(
                        PolicyCandidate(
                            name=name,
                            namespace=namespace,
                            direction="ingress",
                            src_ns="*",
                            src_selector=None,
                            src_ipblock=None,
                            dst_ns=namespace,
                            dst_selector=pod_selector,
                            dst_ipblock=None,
                            ports=ports,
                            source=source,
                        )
                    )
                else:
                    for frm in froms:
                        src_sel = None
                        src_ip = None
                        src_ns: Optional[str] = namespace
                        if isinstance(frm, dict):
                            if frm.get("namespaceSelector") is not None:
                                ns_sel = frm.get("namespaceSelector")
                                if isinstance(ns_sel, dict) and ns_sel.get("matchLabels"):
                                    continue
                                src_ns = "*"
                            src_sel = _safe_labels(frm.get("podSelector"))
                            src_ip = _extract_ipblock(frm.get("ipBlock"))
                        out.append(
                            PolicyCandidate(
                                name=name,
                                namespace=namespace,
                                direction="ingress",
                                src_ns=src_ns,
                                src_selector=src_sel,
                                src_ipblock=src_ip,
                                dst_ns=namespace,
                                dst_selector=pod_selector,
                                dst_ipblock=None,
                                ports=ports,
                                source=source,
                            )
                        )

        if egress_rules:
            for rule in egress_rules:
                if not isinstance(rule, dict):
                    continue
                ports = _extract_ports(rule.get("ports"))
                if "to" in rule and isinstance(rule.get("to"), list) and len(rule.get("to")) == 0:
                    continue

                tos = rule.get("to") if isinstance(rule.get("to"), list) else None
                if tos is None:
                    out.append(
                        PolicyCandidate(
                            name=name,
                            namespace=namespace,
                            direction="egress",
                            src_ns=namespace,
                            src_selector=pod_selector,
                            src_ipblock=None,
                            dst_ns="*",
                            dst_selector=None,
                            dst_ipblock=None,
                            ports=ports,
                            source=source,
                        )
                    )
                else:
                    for to in tos:
                        dst_sel = None
                        dst_ip = None
                        dst_ns: Optional[str] = namespace
                        if isinstance(to, dict):
                            if to.get("namespaceSelector") is not None:
                                ns_sel = to.get("namespaceSelector")
                                if isinstance(ns_sel, dict) and ns_sel.get("matchLabels"):
                                    continue
                                dst_ns = "*"
                            dst_sel = _safe_labels(to.get("podSelector"))
                            dst_ip = _extract_ipblock(to.get("ipBlock"))
                        out.append(
                            PolicyCandidate(
                                name=name,
                                namespace=namespace,
                                direction="egress",
                                src_ns=namespace,
                                src_selector=pod_selector,
                                src_ipblock=None,
                                dst_ns=dst_ns,
                                dst_selector=dst_sel,
                                dst_ipblock=dst_ip,
                                ports=ports,
                                source=source,
                            )
                        )

    return out


def _mk_pod(name: str, namespace: str, labels: Dict[str, str]) -> Dict[str, Any]:
    return {"name": name, "namespace": namespace, "labels": dict(labels)}


def _mk_policy(
    name: str,
    src_ns: Optional[str],
    src_selector: Optional[Dict[str, str]],
    src_ipblock: Optional[str],
    dst_ns: Optional[str],
    dst_selector: Optional[Dict[str, str]],
    dst_ipblock: Optional[str],
    ports: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    pol: Dict[str, Any] = {
        "name": name,
        "action": "allow",
        "from": {"namespace": src_ns, "podSelector": src_selector, "ipBlock": src_ipblock},
        "to": {"namespace": dst_ns, "podSelector": dst_selector, "ipBlock": dst_ipblock},
    }
    if ports is not None:
        pol["ports"] = ports
    return pol


def _mutate_label_value(labels: Dict[str, str]) -> Dict[str, str]:
    out = dict(labels)
    k = next(iter(out.keys()))
    out[k] = f"{out[k]}-typo"
    return out


def _pick_port(ports: Optional[List[Dict[str, Any]]]) -> Tuple[Optional[int], Optional[str]]:
    if not ports:
        return None, None
    p = ports[0]
    port = p.get("port") if isinstance(p, dict) else None
    proto = p.get("protocol") if isinstance(p, dict) else None
    try:
        port_i = int(port) if port is not None else None
    except Exception:
        port_i = None
    proto_s = str(proto).strip().lower() if isinstance(proto, str) and proto.strip() else None
    return port_i, proto_s


def _mutate_port(ports: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    if not ports:
        return None
    out = [dict(p) for p in ports]
    port = out[0].get("port")
    try:
        p = int(port)
    except Exception:
        return None
    out[0]["port"] = 80 if p != 80 else 443
    return out


def _cidr_base(cidr: str) -> Optional[Tuple[int, int]]:
    cidr = cidr.strip()
    if "/" in cidr:
        ip_s, pre_s = cidr.split("/", 1)
    else:
        ip_s, pre_s = cidr, "32"
    parts = ip_s.split(".")
    if len(parts) != 4:
        return None
    try:
        xs = [int(p) for p in parts]
        pre = int(pre_s)
    except Exception:
        return None
    if any(x < 0 or x > 255 for x in xs) or pre < 0 or pre > 32:
        return None
    base = (xs[0] << 24) | (xs[1] << 16) | (xs[2] << 8) | xs[3]
    return base, pre


def _sample_ip_from_cidr(cidr: str) -> str:
    base = _cidr_base(cidr)
    if base is None:
        return "10.0.0.1"
    ipi, _pre = base
    ipi = (ipi & 0xFFFFFF00) + 1
    return ".".join(str((ipi >> (24 - 8 * i)) & 0xFF) for i in range(4))


def _mutate_cidr(cidr: str) -> str:
    base = _cidr_base(cidr)
    if base is None:
        return "192.0.2.0/24"
    ipi, pre = base
    ipi = (ipi & 0xFFFFFF00) + 128
    ip = ".".join(str((ipi >> (24 - 8 * i)) & 0xFF) for i in range(4))
    return f"{ip}/{pre}"


def _intent_reach(a: str, b: str, port: Optional[int], proto: Optional[str]) -> str:
    if port is None:
        return f"Reach {a} {b}"
    proto_s = proto or "tcp"
    return f"Reach {a} {b} port={int(port)} protocol={proto_s}"


def _intent_isolate(a: str, b: str, port: Optional[int], proto: Optional[str]) -> str:
    if port is None:
        return f"Isolate {a} {b}"
    proto_s = proto or "tcp"
    return f"Isolate {a} {b} port={int(port)} protocol={proto_s}"


def _verify(cfg: Dict[str, Any], intent_text: str) -> bool:
    verifier = K8sPythonVerifier()
    intent = parse_intent(intent_text)
    compiled = compile_intent(intent)
    vr = verifier.verify(cfg, compiled.constraints)
    return bool(vr.ok)


def _build_base_cfg(
    cand: PolicyCandidate,
    *,
    src_labels: Dict[str, str],
    dst_labels: Dict[str, str],
    src_name: str,
    dst_name: str,
    extra_pods: Optional[List[Dict[str, Any]]] = None,
    external_ips: Optional[List[str]] = None,
    semantics: str = "option_c",
    policy: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    src_ns = cand.src_ns
    dst_ns = cand.dst_ns
    if src_ns is None or _canon(src_ns) == "*":
        src_ns = cand.namespace
    if dst_ns is None or _canon(dst_ns) == "*":
        dst_ns = cand.namespace
    pods = [
        _mk_pod(src_name, src_ns, src_labels),
        _mk_pod(dst_name, dst_ns, dst_labels),
    ]
    if extra_pods:
        pods.extend(extra_pods)
    k8s: Dict[str, Any] = {"pods": pods, "services": [], "semantics": semantics}
    if external_ips:
        k8s["external_ips"] = list(external_ips)
    policies = [policy] if policy else []
    return {"k8s": k8s, "policies": policies}


def _build_selector_typo_task(
    cand: PolicyCandidate,
    *,
    rng: random.Random,
    semantics: str,
) -> Optional[Dict[str, Any]]:
    side = None
    if cand.src_selector:
        side = "src"
    elif cand.dst_selector:
        side = "dst"
    else:
        return None

    src_labels = cand.src_selector or {"app": "src"}
    dst_labels = cand.dst_selector or {"app": "dst"}

    src_name = f"src-{abs(hash((cand.name, 'src'))) % 10000}"
    dst_name = f"dst-{abs(hash((cand.name, 'dst'))) % 10000}"

    src_sel = dict(src_labels) if cand.src_selector else None
    dst_sel = dict(dst_labels) if cand.dst_selector else None

    if side == "src" and cand.src_selector:
        src_sel = _mutate_label_value(cand.src_selector)
    if side == "dst" and cand.dst_selector:
        dst_sel = _mutate_label_value(cand.dst_selector)

    target_policy = _mk_policy(
        cand.name,
        cand.src_ns,
        cand.src_selector,
        cand.src_ipblock,
        cand.dst_ns,
        cand.dst_selector,
        cand.dst_ipblock,
        cand.ports,
    )
    init_policy = _mk_policy(
        cand.name,
        cand.src_ns,
        src_sel,
        cand.src_ipblock,
        cand.dst_ns,
        dst_sel,
        cand.dst_ipblock,
        cand.ports,
    )

    port, proto = _pick_port(cand.ports)
    src_node = f"ip:{_sample_ip_from_cidr(cand.src_ipblock)}" if cand.src_ipblock else f"pod:{src_name}"
    dst_node = f"ip:{_sample_ip_from_cidr(cand.dst_ipblock)}" if cand.dst_ipblock else f"pod:{dst_name}"
    intent = _intent_reach(src_node, dst_node, port, proto)

    external_ips = []
    if cand.src_ipblock:
        external_ips.append(_sample_ip_from_cidr(cand.src_ipblock))
    if cand.dst_ipblock:
        external_ips.append(_sample_ip_from_cidr(cand.dst_ipblock))

    target_cfg = _build_base_cfg(
        cand,
        src_labels=src_labels,
        dst_labels=dst_labels,
        src_name=src_name,
        dst_name=dst_name,
        external_ips=external_ips or None,
        semantics=semantics,
        policy=target_policy,
    )
    init_cfg = _build_base_cfg(
        cand,
        src_labels=src_labels,
        dst_labels=dst_labels,
        src_name=src_name,
        dst_name=dst_name,
        external_ips=external_ips or None,
        semantics=semantics,
        policy=init_policy,
    )

    if not _verify(target_cfg, intent):
        return None
    if _verify(init_cfg, intent):
        return None

    return {"intent": intent, "initial_cfg": init_cfg, "mode": "selector_typo"}


def _build_port_mismatch_task(
    cand: PolicyCandidate,
    *,
    rng: random.Random,
    semantics: str,
) -> Optional[Dict[str, Any]]:
    if not cand.ports:
        return None

    src_labels = cand.src_selector or {"app": "src"}
    dst_labels = cand.dst_selector or {"app": "dst"}
    src_name = f"src-{abs(hash((cand.name, 'src'))) % 10000}"
    dst_name = f"dst-{abs(hash((cand.name, 'dst'))) % 10000}"

    target_policy = _mk_policy(
        cand.name,
        cand.src_ns,
        cand.src_selector,
        cand.src_ipblock,
        cand.dst_ns,
        cand.dst_selector,
        cand.dst_ipblock,
        cand.ports,
    )
    init_ports = _mutate_port(cand.ports)
    if init_ports is None:
        return None
    init_policy = _mk_policy(
        cand.name,
        cand.src_ns,
        cand.src_selector,
        cand.src_ipblock,
        cand.dst_ns,
        cand.dst_selector,
        cand.dst_ipblock,
        init_ports,
    )

    port, proto = _pick_port(cand.ports)
    src_node = f"ip:{_sample_ip_from_cidr(cand.src_ipblock)}" if cand.src_ipblock else f"pod:{src_name}"
    dst_node = f"ip:{_sample_ip_from_cidr(cand.dst_ipblock)}" if cand.dst_ipblock else f"pod:{dst_name}"
    intent = _intent_reach(src_node, dst_node, port, proto)

    external_ips = []
    if cand.src_ipblock:
        external_ips.append(_sample_ip_from_cidr(cand.src_ipblock))
    if cand.dst_ipblock:
        external_ips.append(_sample_ip_from_cidr(cand.dst_ipblock))

    target_cfg = _build_base_cfg(
        cand,
        src_labels=src_labels,
        dst_labels=dst_labels,
        src_name=src_name,
        dst_name=dst_name,
        external_ips=external_ips or None,
        semantics=semantics,
        policy=target_policy,
    )
    init_cfg = _build_base_cfg(
        cand,
        src_labels=src_labels,
        dst_labels=dst_labels,
        src_name=src_name,
        dst_name=dst_name,
        external_ips=external_ips or None,
        semantics=semantics,
        policy=init_policy,
    )

    if not _verify(target_cfg, intent):
        return None
    if _verify(init_cfg, intent):
        return None

    return {"intent": intent, "initial_cfg": init_cfg, "mode": "port_mismatch"}


def _build_ipblock_mismatch_task(
    cand: PolicyCandidate,
    *,
    rng: random.Random,
    semantics: str,
) -> Optional[Dict[str, Any]]:
    if not (cand.src_ipblock or cand.dst_ipblock):
        return None

    src_labels = cand.src_selector or {"app": "src"}
    dst_labels = cand.dst_selector or {"app": "dst"}
    src_name = f"src-{abs(hash((cand.name, 'src'))) % 10000}"
    dst_name = f"dst-{abs(hash((cand.name, 'dst'))) % 10000}"

    src_ip = cand.src_ipblock
    dst_ip = cand.dst_ipblock

    init_src_ip = _mutate_cidr(src_ip) if src_ip else None
    init_dst_ip = _mutate_cidr(dst_ip) if dst_ip else None

    target_policy = _mk_policy(
        cand.name,
        cand.src_ns,
        cand.src_selector,
        src_ip,
        cand.dst_ns,
        cand.dst_selector,
        dst_ip,
        cand.ports,
    )
    init_policy = _mk_policy(
        cand.name,
        cand.src_ns,
        cand.src_selector,
        init_src_ip,
        cand.dst_ns,
        cand.dst_selector,
        init_dst_ip,
        cand.ports,
    )

    port, proto = _pick_port(cand.ports)
    src_node = f"ip:{_sample_ip_from_cidr(src_ip)}" if src_ip else f"pod:{src_name}"
    dst_node = f"ip:{_sample_ip_from_cidr(dst_ip)}" if dst_ip else f"pod:{dst_name}"
    intent = _intent_reach(src_node, dst_node, port, proto)

    external_ips = []
    if src_ip:
        external_ips.append(_sample_ip_from_cidr(src_ip))
    if dst_ip:
        external_ips.append(_sample_ip_from_cidr(dst_ip))

    target_cfg = _build_base_cfg(
        cand,
        src_labels=src_labels,
        dst_labels=dst_labels,
        src_name=src_name,
        dst_name=dst_name,
        external_ips=external_ips or None,
        semantics=semantics,
        policy=target_policy,
    )
    init_cfg = _build_base_cfg(
        cand,
        src_labels=src_labels,
        dst_labels=dst_labels,
        src_name=src_name,
        dst_name=dst_name,
        external_ips=external_ips or None,
        semantics=semantics,
        policy=init_policy,
    )

    if not _verify(target_cfg, intent):
        return None
    if _verify(init_cfg, intent):
        return None

    return {"intent": intent, "initial_cfg": init_cfg, "mode": "ipblock_mismatch"}


def _build_over_permissive_task(
    cand: PolicyCandidate,
    *,
    rng: random.Random,
    semantics: str,
) -> Optional[Dict[str, Any]]:
    if not cand.src_selector and not cand.dst_selector:
        return None

    src_labels = cand.src_selector or {"app": "src"}
    dst_labels = cand.dst_selector or {"app": "dst"}
    src_name = f"src-{abs(hash((cand.name, 'src'))) % 10000}"
    dst_name = f"dst-{abs(hash((cand.name, 'dst'))) % 10000}"

    port, proto = _pick_port(cand.ports)

    broaden_src = cand.src_selector is not None
    broaden_dst = not broaden_src

    decoy_labels = {"app": f"decoy-{src_labels.get('app', 'src')}"}
    decoy_name = f"decoy-{abs(hash((cand.name, 'decoy'))) % 10000}"

    target_policy = _mk_policy(
        cand.name,
        cand.src_ns,
        cand.src_selector,
        cand.src_ipblock,
        cand.dst_ns,
        cand.dst_selector,
        cand.dst_ipblock,
        cand.ports,
    )

    init_src_sel = None if broaden_src else cand.src_selector
    init_dst_sel = None if broaden_dst else cand.dst_selector
    init_policy = _mk_policy(
        cand.name,
        cand.src_ns,
        init_src_sel,
        cand.src_ipblock,
        cand.dst_ns,
        init_dst_sel,
        cand.dst_ipblock,
        cand.ports,
    )

    src_node = f"ip:{_sample_ip_from_cidr(cand.src_ipblock)}" if cand.src_ipblock else f"pod:{src_name}"
    dst_node = f"ip:{_sample_ip_from_cidr(cand.dst_ipblock)}" if cand.dst_ipblock else f"pod:{dst_name}"
    decoy_node = f"pod:{decoy_name}"

    intent = f"{_intent_reach(src_node, dst_node, port, proto)} AND {_intent_isolate(decoy_node, dst_node, port, proto)}"

    external_ips = []
    if cand.src_ipblock:
        external_ips.append(_sample_ip_from_cidr(cand.src_ipblock))
    if cand.dst_ipblock:
        external_ips.append(_sample_ip_from_cidr(cand.dst_ipblock))

    decoy_ns = cand.src_ns if broaden_src else cand.dst_ns
    decoy_pod = _mk_pod(decoy_name, decoy_ns or cand.namespace, decoy_labels)

    target_cfg = _build_base_cfg(
        cand,
        src_labels=src_labels,
        dst_labels=dst_labels,
        src_name=src_name,
        dst_name=dst_name,
        extra_pods=[decoy_pod],
        external_ips=external_ips or None,
        semantics=semantics,
        policy=target_policy,
    )
    init_cfg = _build_base_cfg(
        cand,
        src_labels=src_labels,
        dst_labels=dst_labels,
        src_name=src_name,
        dst_name=dst_name,
        extra_pods=[decoy_pod],
        external_ips=external_ips or None,
        semantics=semantics,
        policy=init_policy,
    )

    if not _verify(target_cfg, intent):
        return None
    if _verify(init_cfg, intent):
        return None

    return {"intent": intent, "initial_cfg": init_cfg, "mode": "over_permissive"}


def _build_task_from_candidate(
    cand: PolicyCandidate,
    *,
    rng: random.Random,
    semantics: str,
) -> Optional[Dict[str, Any]]:
    builders = [
        _build_selector_typo_task,
        _build_port_mismatch_task,
        _build_ipblock_mismatch_task,
        _build_over_permissive_task,
    ]
    rng.shuffle(builders)
    for fn in builders:
        t = fn(cand, rng=rng, semantics=semantics)
        if t:
            return t
    return None


def _build_tasks_from_candidate(
    cand: PolicyCandidate,
    *,
    rng: random.Random,
    semantics: str,
    max_tasks_per_candidate: int,
) -> List[Dict[str, Any]]:
    builders = [
        _build_selector_typo_task,
        _build_port_mismatch_task,
        _build_ipblock_mismatch_task,
        _build_over_permissive_task,
    ]
    rng.shuffle(builders)
    out: List[Dict[str, Any]] = []
    k = int(max_tasks_per_candidate)
    if k <= 0:
        k = 1
    for fn in builders:
        if len(out) >= k:
            break
        t = fn(cand, rng=rng, semantics=semantics)
        if t:
            out.append(t)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=str, default=".\\data\\realworld_k8s_sources\\index.json")
    ap.add_argument("--out", type=str, default=".\\data\\k8s_realworld_100_tasks.json")
    ap.add_argument("--report_out", type=str, default=".\\data\\k8s_realworld_100_tasks_report.json")
    ap.add_argument("--max_tasks", type=int, default=100)
    ap.add_argument(
        "--max_tasks_per_candidate",
        type=int,
        default=2,
        help="Generate up to K tasks per mined policy candidate by trying multiple perturbation modes.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_rounds", type=int, default=6)
    ap.add_argument("--semantics", type=str, default="option_c")
    ap.add_argument("--model", type=str, default="gpt-4o-mini")
    ap.add_argument("--base_url", type=str, default=None)
    ap.add_argument("--timeout", type=float, default=None)
    args = ap.parse_args()

    rng = random.Random(int(args.seed))
    index_path = Path(args.index)
    out_path = Path(args.out)

    entries = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise RuntimeError("index.json must be a list")

    candidates: List[PolicyCandidate] = []
    for it in entries:
        if not isinstance(it, dict):
            continue
        local_path = it.get("local_path") or it.get("path")
        if not local_path:
            continue
        p = Path(local_path)
        if not p.exists():
            continue
        source = {k: it.get(k) for k in ["owner", "repo", "branch", "path", "raw_url"] if it.get(k) is not None}
        docs = _load_yaml_docs(p)
        candidates.extend(_extract_candidates(docs, source))

    rng.shuffle(candidates)

    llm_spec: Dict[str, Any] = {"type": "openai", "model": args.model}
    if args.base_url:
        llm_spec["base_url"] = args.base_url
    if args.timeout is not None:
        llm_spec["timeout"] = args.timeout

    prompt = {
        "domain": "k8s_networkpolicy",
        "instruction": (
            "Do not change k8s.pods or k8s.services. Only edit policies. Prefer minimal diffs. "
            "Only allow policies are supported (no deny semantics): do not create deny/block policies. "
            "Do not use ports:null; since intents specify port/protocol, keep ports as an explicit list. "
            "For Isolate violations, remove or narrow the allow policy/rule that enables the forbidden path "
            "(prioritize blame_set)."
        ),
    }

    tasks: List[Dict[str, Any]] = []
    max_k = int(args.max_tasks_per_candidate)
    if max_k <= 0:
        max_k = 1
    for cand in candidates:
        if len(tasks) >= int(args.max_tasks):
            break
        built = _build_tasks_from_candidate(
            cand,
            rng=rng,
            semantics=args.semantics,
            max_tasks_per_candidate=max_k,
        )
        for t in built:
            if len(tasks) >= int(args.max_tasks):
                break
            idx = len(tasks) + 1
            task = {
                "id": f"real_k8s100_{idx:04d}_{t['mode']}",
                "intent": t["intent"],
                "initial_prompt": prompt,
                "initial_cfg": t["initial_cfg"],
                "llm": dict(llm_spec),
                "max_rounds": int(args.max_rounds),
                "meta": {"source": cand.source, "mode": t["mode"], "policy": cand.name},
            }
            tasks.append(task)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")

    mode_cnt: Dict[str, int] = {}
    repo_cnt: Dict[str, int] = {}
    ports_cnt = 0
    ipblock_cnt = 0
    for t in tasks:
        meta = t.get("meta") if isinstance(t.get("meta"), dict) else {}
        mode = meta.get("mode") or ""
        if isinstance(mode, str) and mode:
            mode_cnt[mode] = mode_cnt.get(mode, 0) + 1
        src = meta.get("source") if isinstance(meta.get("source"), dict) else {}
        owner = src.get("owner")
        repo = src.get("repo")
        if isinstance(owner, str) and isinstance(repo, str):
            k = f"{owner}/{repo}"
            repo_cnt[k] = repo_cnt.get(k, 0) + 1
        cfg = t.get("initial_cfg") if isinstance(t.get("initial_cfg"), dict) else {}
        pols = cfg.get("policies") if isinstance(cfg.get("policies"), list) else []
        if pols and isinstance(pols[0], dict) and pols[0].get("ports") is not None:
            ports_cnt += 1
        if pols and isinstance(pols[0], dict):
            frm = pols[0].get("from") if isinstance(pols[0].get("from"), dict) else {}
            to = pols[0].get("to") if isinstance(pols[0].get("to"), dict) else {}
            if frm.get("ipBlock") or to.get("ipBlock"):
                ipblock_cnt += 1

    report = {
        "tasks": len(tasks),
        "candidates": len(candidates),
        "modes": dict(sorted(mode_cnt.items(), key=lambda x: (-x[1], x[0]))),
        "unique_repos": len(repo_cnt),
        "top_repos": dict(sorted(repo_cnt.items(), key=lambda x: (-x[1], x[0]))[:20]),
        "ports_explicit_rate": (ports_cnt / len(tasks)) if tasks else 0.0,
        "ipblock_rate": (ipblock_cnt / len(tasks)) if tasks else 0.0,
    }
    report_path = Path(args.report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print({"tasks": len(tasks), "candidates": len(candidates), "out": str(out_path), "report": str(report_path)})


if __name__ == "__main__":
    main()
