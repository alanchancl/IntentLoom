from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from ..compiler import Constraint, Violation
from ..symbolicai_contracts import evaluate_contract
from ..verifier import VerificationResult, VerifierLike

from .k8s_adapter import k8s_semantics, k8s_to_graph_cfg
from .k8s_contracts import build_contract_for_k8s_primitive
from .k8s_adapter import _cidr_contains, build_k8s_graph, find_path, summarize_path_reasons


@dataclass
class K8sPythonVerifier(VerifierLike):
    def verify(self, cfg: Dict[str, Any], constraints: List[Constraint]) -> VerificationResult:
        violations: List[Violation] = []
        directed = k8s_semantics(cfg) == "option_c"
        for c in constraints:
            intent = dict(c.intent or {})
            if c.name in {"Isolate", "Reach"}:
                params = intent.get("params", {}) if isinstance(intent, dict) else {}
                port = params.get("port") if isinstance(params, dict) else None
                protocol = params.get("protocol") if isinstance(params, dict) else None
                port_i = int(port) if isinstance(port, int) or (isinstance(port, str) and str(port).strip().isdigit()) else None
                proto_s = str(protocol).strip() if isinstance(protocol, str) and str(protocol).strip() else None

                g = build_k8s_graph(cfg)
                a = str(params.get("a") or "")
                b = str(params.get("b") or "")
                p = find_path(g.links, a, b, edge_reasons=g.edge_reasons, port=port_i, protocol=proto_s, directed=directed)
                if c.name == "Isolate":
                    if p is None and directed:
                        p = find_path(g.links, b, a, edge_reasons=g.edge_reasons, port=port_i, protocol=proto_s, directed=directed)
                    if p is not None:
                        v = _violation_from_intent(intent)
                        meta = dict(v.meta or {})
                        ev = summarize_path_reasons(p, g.edge_reasons, port=port_i, protocol=proto_s, directed=directed)
                        if ev:
                            meta["policy_evidence_str"] = ev
                            meta["policy_evidence"] = sorted(set(re.findall(r"policy=([A-Za-z0-9_.:-]+)", ev)))
                        violations.append(Violation(code=v.code, message=v.message, path=v.path, meta=meta))
                else:
                    if p is None:
                        v = _violation_from_intent(intent)
                        meta = dict(v.meta or {})
                        ev, ev_pols = _reach_policy_evidence(cfg, a=a, b=b, port=port_i, protocol=proto_s)
                        if ev:
                            meta["policy_evidence_str"] = ev
                        if ev_pols:
                            meta["policy_evidence"] = sorted(set(ev_pols))
                        violations.append(Violation(code=v.code, message=v.message, path=v.path, meta=meta))
                continue

            graph_cfg = k8s_to_graph_cfg(cfg)
            violations.extend(c.check(graph_cfg))
        return VerificationResult(ok=len(violations) == 0, violations=violations)


@dataclass
class K8sSymbolicAIContractVerifier(VerifierLike):
    _cache: Dict[Tuple[str, str], Any] = None

    def __post_init__(self) -> None:
        try:
            import symai  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "SymbolicAI (symai) is not available in the current Python environment. "
                "Ensure you run the experiment with the interpreter where 'symbolicai' is installed. "
                f"python_executable={sys.executable}"
            ) from e
        if self._cache is None:
            self._cache = {}

    def verify(self, cfg: Dict[str, Any], constraints: List[Constraint]) -> VerificationResult:
        violations: List[Violation] = []
        graph_cfg = k8s_to_graph_cfg(cfg)

        for c in constraints:
            intent = dict(c.intent or {})
            if c.name in {"Isolate", "Reach"}:
                key = (c.name, str(intent.get("params", {})) + str(intent.get("path", [])))
                spec = self._cache.get(key)
                if spec is None:
                    spec = build_contract_for_k8s_primitive(intent)
                    self._cache[key] = spec

                ok, exc = evaluate_contract(spec, cfg)
                if not ok:
                    v = _violation_from_intent(intent)
                    meta = dict(v.meta or {})
                    if exc is not None:
                        meta.update(
                            {
                                "contract_exception": str(exc),
                                "contract_type": getattr(spec.contract_cls, "__name__", "<contract>"),
                                "contract_prompt": getattr(spec.contract_cls, "prompt", None),
                            }
                        )
                    violations.append(Violation(code=v.code, message=v.message, path=v.path, meta=meta))
                continue

            violations.extend(c.check(graph_cfg))

        return VerificationResult(ok=len(violations) == 0, violations=violations)


def _violation_from_intent(intent: Dict[str, Any]) -> Violation:
    name = intent.get("name")
    params = intent.get("params", {})
    path = tuple(intent.get("path", []))

    port = params.get("port") if isinstance(params, dict) else None
    protocol = params.get("protocol") if isinstance(params, dict) else None
    extra = ""
    if port is not None:
        extra = f" (port={port}" + (f", protocol={protocol}" if protocol else "") + ")"

    if name == "Isolate":
        a = params.get("a")
        b = params.get("b")
        return Violation(
            code="ISOLATE_VIOLATION",
            message=f"Isolate violated: path exists between {a} and {b}{extra}",
            path=path,
            meta={"a": a, "b": b, "port": port, "protocol": protocol},
        )

    if name == "Reach":
        a = params.get("a")
        b = params.get("b")
        return Violation(
            code="REACH_VIOLATION",
            message=f"Reach violated: no path between {a} and {b}{extra}",
            path=path,
            meta={"a": a, "b": b, "port": port, "protocol": protocol},
        )

    return Violation(
        code="UNKNOWN_PRIMITIVE",
        message=f"Unknown primitive: {name}",
        path=path,
        meta={"name": name, "params": params},
    )


def _canon(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip().lower()


def _k8s_match_labels(labels: Dict[str, Any], selector: Optional[Dict[str, Any]]) -> bool:
    if not selector:
        return True
    if not isinstance(labels, dict):
        return False
    for k, v in selector.items():
        if labels.get(k) != v:
            return False
    return True


def _k8s_match_namespace(ns_name: str, selector: Optional[str]) -> bool:
    if selector is None:
        return True
    s = _canon(selector)
    if s == "*":
        return True
    return _canon(ns_name) == s


def _k8s_policy_ports_allow(ports: Any, *, port: Optional[int], protocol: Optional[str]) -> bool:
    if port is None and protocol is None:
        return True
    if ports is None:
        return True
    if not isinstance(ports, list):
        return False
    proto = _canon(protocol) if isinstance(protocol, str) and protocol.strip() else ""
    for it in ports:
        if not isinstance(it, dict):
            continue
        p = it.get("port")
        try:
            p_i = int(p) if p is not None else None
        except Exception:
            p_i = None
        if port is not None and p_i != port:
            continue
        pr = it.get("protocol")
        pr_s = _canon(pr) if isinstance(pr, str) and pr.strip() else ""
        if proto and pr_s and pr_s != proto:
            continue
        return True
    return False


def _k8s_pods(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    k8s = cfg.get("k8s") if isinstance(cfg, dict) else None
    pods = k8s.get("pods") if isinstance(k8s, dict) else None
    return pods if isinstance(pods, list) else []


def _k8s_services(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    k8s = cfg.get("k8s") if isinstance(cfg, dict) else None
    svcs = k8s.get("services") if isinstance(k8s, dict) else None
    return svcs if isinstance(svcs, list) else []


def _k8s_extract_ip(node: str) -> str:
    s = _canon(node)
    if s.startswith("ip:"):
        return s.split(":", 1)[1]
    return ""


def _k8s_extract_name(node: str, prefix: str) -> str:
    s = _canon(node)
    if s.startswith(prefix + ":"):
        return s.split(":", 1)[1]
    return ""


def _k8s_select_pods(pods: List[Dict[str, Any]], *, namespace: Optional[str], selector: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in pods:
        ns = _canon(p.get("namespace"))
        if not _k8s_match_namespace(ns, namespace):
            continue
        if not _k8s_match_labels(p.get("labels") or {}, selector):
            continue
        out.append(p)
    return out


def _k8s_policy_selects_any_pod(side: Any, pods: List[Dict[str, Any]]) -> bool:
    if not isinstance(side, dict):
        return False
    ns = side.get("namespace") if isinstance(side.get("namespace"), str) else None
    sel = side.get("podSelector") if isinstance(side.get("podSelector"), dict) else None
    for p in pods:
        if not _k8s_match_namespace(_canon(p.get("namespace")), ns):
            continue
        if not _k8s_match_labels(p.get("labels") or {}, sel):
            continue
        return True
    return False


def _reach_policy_evidence(
    cfg: Dict[str, Any],
    *,
    a: str,
    b: str,
    port: Optional[int],
    protocol: Optional[str],
) -> Tuple[str, List[str]]:
    pods = _k8s_pods(cfg)
    svcs = _k8s_services(cfg)

    src_ip = _k8s_extract_ip(a)
    src_pod_name = _k8s_extract_name(a, "pod")
    dst_ip = _k8s_extract_ip(b)
    dst_pod_name = _k8s_extract_name(b, "pod")
    dst_svc_name = _k8s_extract_name(b, "svc")

    dst_pods: List[Dict[str, Any]] = []
    if dst_svc_name:
        svc = None
        for s in svcs:
            if _canon(s.get("name")) == dst_svc_name:
                svc = s
                break
        if isinstance(svc, dict):
            ns = svc.get("namespace") if isinstance(svc.get("namespace"), str) else None
            selector = svc.get("selector") if isinstance(svc.get("selector"), dict) else None
            dst_pods = _k8s_select_pods(pods, namespace=ns, selector=selector)
    elif dst_pod_name:
        for p in pods:
            if _canon(p.get("name")) == dst_pod_name:
                dst_pods = [p]
                break

    policies = cfg.get("policies") if isinstance(cfg, dict) else None
    policies_list = policies if isinstance(policies, list) else []

    ev_pols: Set[str] = set()
    parts: List[str] = []

    for pol in policies_list:
        if not isinstance(pol, dict):
            continue
        if _canon(pol.get("action") or "allow") != "allow":
            continue
        name = str(pol.get("name") or "").strip()
        if not name:
            continue

        if not _k8s_policy_ports_allow(pol.get("ports"), port=port, protocol=protocol):
            continue

        to = pol.get("to") if isinstance(pol.get("to"), dict) else {}
        dst_match = False
        if dst_ip:
            ipb = to.get("ipBlock") if isinstance(to.get("ipBlock"), str) else ""
            if ipb:
                try:
                    dst_match = bool(_cidr_contains(ipb, dst_ip))
                except Exception:
                    dst_match = False
        elif dst_pods:
            dst_match = _k8s_policy_selects_any_pod(to, dst_pods)
        if not dst_match:
            continue

        frm = pol.get("from") if isinstance(pol.get("from"), dict) else {}
        src_match = False
        src_hint = ""
        if src_ip:
            ipb = frm.get("ipBlock") if isinstance(frm.get("ipBlock"), str) else ""
            if not ipb:
                src_match = False
                src_hint = "missing_from_ipBlock"
            else:
                try:
                    src_match = bool(_cidr_contains(ipb, src_ip))
                    if not src_match:
                        src_hint = f"from_ipBlock_mismatch({ipb})"
                except Exception:
                    src_match = False
                    src_hint = f"from_ipBlock_parse_error({ipb})"
        elif src_pod_name:
            src_pods: List[Dict[str, Any]] = []
            for p in pods:
                if _canon(p.get("name")) == src_pod_name:
                    src_pods = [p]
                    break
            if src_pods:
                src_match = _k8s_policy_selects_any_pod(frm, src_pods)
                if not src_match:
                    src_hint = "from_selector_mismatch"

        ev_pols.add(name)
        parts.append(
            "policy="
            + name
            + " dst="
            + _canon(b)
            + " src_match="
            + ("true" if src_match else "false")
            + (" " + src_hint if src_hint else "")
            + " from="
            + str(frm)
            + " to="
            + str(to)
        )

    if not parts and dst_svc_name:
        suggest = f"suggest_add_allow_to_{dst_svc_name}"
        parts.append(
            "policy="
            + suggest
            + " missing_candidate_for_dst="
            + _canon(b)
            + " src="
            + _canon(a)
        )
        ev_pols.add(suggest)

    ev = "; ".join(parts)
    return ev, sorted(ev_pols)
