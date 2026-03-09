from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass(frozen=True)
class K8sGraph:
    links: List[List[str]]
    edge_reasons: Dict[Tuple[str, str], List[Dict[str, Any]]]


def _canon(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip().lower()


def _semantics(cfg: Dict[str, Any]) -> str:
    k8s = cfg.get("k8s") if isinstance(cfg, dict) else None
    if not isinstance(k8s, dict):
        return "option_a"
    s = _canon(k8s.get("semantics") or "option_a")
    return s or "option_a"


def _ip_node(ip: str) -> str:
    s = _canon(ip)
    return f"ip:{s}" if s else ""


def _ip_to_int(ip: str) -> Optional[int]:
    s = _canon(ip)
    if not s:
        return None

    if s.startswith("ip:"):
        s = s.split(":", 1)[1]
    parts = s.split(".")
    if len(parts) != 4:
        return None
    try:
        xs = [int(p) for p in parts]
    except Exception:
        return None
    if any(x < 0 or x > 255 for x in xs):
        return None
    return (xs[0] << 24) | (xs[1] << 16) | (xs[2] << 8) | xs[3]


def _parse_cidr(cidr: str) -> Optional[Tuple[int, int]]:
    s = _canon(cidr)
    if not s:
        return None
    if s.startswith("ip:"):
        s = s.split(":", 1)[1]
    if "/" not in s:
        ipi = _ip_to_int(s)
        if ipi is None:
            return None
        return ipi, 32
    ip_s, pre_s = s.split("/", 1)
    ipi = _ip_to_int(ip_s)
    if ipi is None:
        return None
    try:
        pre = int(pre_s)
    except Exception:
        return None
    if pre < 0 or pre > 32:
        return None
    return ipi, pre


def _cidr_contains(cidr: str, ip: str) -> bool:
    c = _parse_cidr(cidr)
    ipi = _ip_to_int(ip)
    if c is None or ipi is None:
        return False
    base, pre = c
    if pre == 0:
        return True
    mask = (0xFFFFFFFF << (32 - pre)) & 0xFFFFFFFF
    return (base & mask) == (ipi & mask)


def _external_ips(cfg: Dict[str, Any]) -> List[str]:
    k8s = cfg.get("k8s") if isinstance(cfg, dict) else None
    if not isinstance(k8s, dict):
        return []
    xs = k8s.get("external_ips")
    if not isinstance(xs, list):
        return []
    out: List[str] = []
    for x in xs:
        s = _canon(x)
        if s:
            out.append(s)
    return out


def _parse_ports(spec: Any) -> Optional[List[Dict[str, Any]]]:
    if spec is None:
        return None
    if isinstance(spec, (int, str)):
        try:
            p = int(str(spec).strip())
            return [{"port": p}]
        except Exception:
            return None
    if not isinstance(spec, list):
        return None
    out: List[Dict[str, Any]] = []
    for it in spec:
        if isinstance(it, int):
            out.append({"port": int(it)})
            continue
        if isinstance(it, str):
            try:
                out.append({"port": int(it.strip())})
            except Exception:
                continue
            continue
        if isinstance(it, dict):
            prt = it.get("port")
            if prt is None:
                continue
            try:
                p = int(prt)
            except Exception:
                continue
            rec: Dict[str, Any] = {"port": p}
            proto = it.get("protocol")
            if isinstance(proto, str) and proto.strip():
                rec["protocol"] = proto.strip().upper()
            out.append(rec)
            continue
    return out or None


def _reason_allows_port(reason: Dict[str, Any], *, port: Optional[int], protocol: Optional[str]) -> bool:
    if reason.get("type") != "policy":
        return True
    ports = reason.get("ports")
    if port is None:
        return True
    if ports is None:
        return True
    if not isinstance(ports, list):
        return True
    prot_u = protocol.strip().upper() if isinstance(protocol, str) and protocol.strip() else None
    for it in ports:
        if not isinstance(it, dict):
            continue
        p = it.get("port")
        if not isinstance(p, int):
            continue
        if p != int(port):
            continue
        if prot_u is None:
            return True
        pr = it.get("protocol")
        if not isinstance(pr, str) or not pr.strip():
            return True
        if pr.strip().upper() == prot_u:
            return True
    return False


def _match_labels(labels: Dict[str, Any], selector: Optional[Dict[str, Any]]) -> bool:
    if not selector:
        return True
    if not isinstance(labels, dict):
        return False
    for k, v in selector.items():
        if labels.get(k) != v:
            return False
    return True


def _match_namespace(ns_name: str, selector: Optional[str]) -> bool:
    if selector is None:
        return True
    s = _canon(selector)
    if not s:
        return True
    if s == "*":
        return True
    return _canon(ns_name) == s


def _pods(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    k8s = cfg.get("k8s") if isinstance(cfg, dict) else None
    pods = k8s.get("pods") if isinstance(k8s, dict) else None
    return pods if isinstance(pods, list) else []


def _services(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    k8s = cfg.get("k8s") if isinstance(cfg, dict) else None
    svcs = k8s.get("services") if isinstance(k8s, dict) else None
    return svcs if isinstance(svcs, list) else []


def _policies(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    ps = cfg.get("policies") if isinstance(cfg, dict) else None
    return ps if isinstance(ps, list) else []


def _pod_node(pod: Dict[str, Any]) -> str:
    name = _canon(pod.get("name"))
    return f"pod:{name}" if name else ""


def _ns_node(ns_name: Any) -> str:
    n = _canon(ns_name)
    return f"ns:{n}" if n else ""


def _svc_node(svc: Dict[str, Any]) -> str:
    name = _canon(svc.get("name"))
    return f"svc:{name}" if name else ""


def _select_pods(
    pods: List[Dict[str, Any]],
    *,
    namespace: Optional[str],
    pod_selector: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in pods:
        ns = _canon(p.get("namespace"))
        if not _match_namespace(ns, namespace):
            continue
        if not _match_labels(p.get("labels") or {}, pod_selector):
            continue
        out.append(p)
    return out


def build_k8s_graph(cfg: Dict[str, Any]) -> K8sGraph:
    pods = _pods(cfg)
    services = _services(cfg)
    policies = _policies(cfg)
    semantics = _semantics(cfg)
    ext_ips = _external_ips(cfg)

    links: List[List[str]] = []
    reasons: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

    directed_graph = semantics == "option_c"

    def _add_edge(u: str, v: str, reason: Optional[Dict[str, Any]] = None) -> None:
        if not u or not v:
            return
        a = _canon(u)
        b = _canon(v)
        if not a or not b:
            return
        if a == b:
            return
        links.append([a, b])
        key = (a, b) if directed_graph else ((a, b) if a < b else (b, a))
        if reason is not None:
            reasons.setdefault(key, []).append(reason)

    for s in services:
        sn = _svc_node(s)
        ns = _canon(s.get("namespace"))
        selector = s.get("selector") if isinstance(s.get("selector"), dict) else {}
        matched = _select_pods(pods, namespace=ns, pod_selector=selector)
        for p in matched:
            pn = _pod_node(p)
            _add_edge(sn, pn, {"type": "service", "service": _canon(s.get("name")), "namespace": ns})
            if semantics == "option_c":
                _add_edge(pn, sn, {"type": "service", "service": _canon(s.get("name")), "namespace": ns})

    for pol in policies:
        if not isinstance(pol, dict):
            continue
        action = _canon(pol.get("action") or "allow")
        if action not in {"allow"}:
            continue

        name = _canon(pol.get("name") or "policy")
        src = pol.get("from") if isinstance(pol.get("from"), dict) else {}
        dst = pol.get("to") if isinstance(pol.get("to"), dict) else {}

        ports = None if semantics == "option_a" else _parse_ports(pol.get("ports"))

        src_ns = src.get("namespace") if isinstance(src.get("namespace"), str) else None
        dst_ns = dst.get("namespace") if isinstance(dst.get("namespace"), str) else None
        src_sel = src.get("podSelector") if isinstance(src.get("podSelector"), dict) else None
        dst_sel = dst.get("podSelector") if isinstance(dst.get("podSelector"), dict) else None

        src_ipblock = None if semantics == "option_a" else (src.get("ipBlock") if isinstance(src.get("ipBlock"), str) else None)
        dst_ipblock = None if semantics == "option_a" else (dst.get("ipBlock") if isinstance(dst.get("ipBlock"), str) else None)

        src_pods = [] if src_ipblock else _select_pods(pods, namespace=src_ns, pod_selector=src_sel)
        dst_pods = [] if dst_ipblock else _select_pods(pods, namespace=dst_ns, pod_selector=dst_sel)

        src_ips = [ip for ip in ext_ips if src_ipblock and _cidr_contains(src_ipblock, ip)] if src_ipblock else []
        dst_ips = [ip for ip in ext_ips if dst_ipblock and _cidr_contains(dst_ipblock, ip)] if dst_ipblock else []

        reason = {
            "type": "policy",
            "name": name,
            "from": {"namespace": src_ns, "podSelector": src_sel, "ipBlock": src_ipblock},
            "to": {"namespace": dst_ns, "podSelector": dst_sel, "ipBlock": dst_ipblock},
            "ports": ports,
        }

        for a in src_pods:
            for b in dst_pods:
                u = _pod_node(a)
                v = _pod_node(b)
                _add_edge(u, v, dict(reason))

                a_ns = _canon(a.get("namespace"))
                b_ns = _canon(b.get("namespace"))
                if a_ns and b_ns and a_ns != b_ns:
                    _add_edge(_ns_node(a_ns), _ns_node(b_ns), dict(reason))

        for a in src_pods:
            for ip in dst_ips:
                u = _pod_node(a)
                v = _ip_node(ip)
                _add_edge(u, v, dict(reason))

        for ip in src_ips:
            for b in dst_pods:
                u = _ip_node(ip)
                v = _pod_node(b)
                _add_edge(u, v, dict(reason))

    return K8sGraph(links=links, edge_reasons=reasons)


def k8s_to_graph_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    g = build_k8s_graph(cfg)
    return {"links": g.links, "acl": []}


def find_path(
    links: List[List[str]],
    src: str,
    dst: str,
    *,
    edge_reasons: Optional[Dict[Tuple[str, str], List[Dict[str, Any]]]] = None,
    port: Optional[int] = None,
    protocol: Optional[str] = None,
    directed: bool = False,
) -> Optional[List[str]]:
    s = _canon(src)
    t = _canon(dst)
    if not s or not t:
        return None
    if s == t:
        return [s]

    g: Dict[str, List[str]] = {}
    nodes: Set[str] = set()
    for u, v in links:
        a = _canon(u)
        b = _canon(v)
        if not a or not b:
            continue
        nodes.add(a)
        nodes.add(b)
        g.setdefault(a, []).append(b)
        if not directed:
            g.setdefault(b, []).append(a)

    if s not in nodes or t not in nodes:
        return None

    g.setdefault(s, [])

    q: List[str] = [s]
    parent: Dict[str, Optional[str]] = {s: None}

    while q:
        cur = q.pop(0)
        for nxt in g.get(cur, []):
            if nxt in parent:
                continue
            if edge_reasons is not None and (port is not None or protocol is not None):
                a = _canon(cur)
                b = _canon(nxt)
                key = (a, b) if directed else ((a, b) if a < b else (b, a))
                rs = edge_reasons.get(key, [])
                ok_edge = False
                for r in rs:
                    if not isinstance(r, dict):
                        continue
                    if _reason_allows_port(r, port=port, protocol=protocol):
                        ok_edge = True
                        break
                if not ok_edge:
                    continue
            parent[nxt] = cur
            if nxt == t:
                out: List[str] = [t]
                p = cur
                while p is not None:
                    out.append(p)
                    p = parent.get(p)
                out.reverse()
                return out
            q.append(nxt)

    return None


def summarize_path_reasons(
    path: List[str],
    edge_reasons: Dict[Tuple[str, str], List[Dict[str, Any]]],
    *,
    port: Optional[int] = None,
    protocol: Optional[str] = None,
    directed: bool = False,
) -> str:
    if not path or len(path) < 2:
        return ""
    parts: List[str] = []
    for i in range(len(path) - 1):
        a = _canon(path[i])
        b = _canon(path[i + 1])
        key = (a, b) if directed else ((a, b) if a < b else (b, a))
        rs = edge_reasons.get(key, [])
        for r in rs:
            if r.get("type") == "policy":
                if not _reason_allows_port(r, port=port, protocol=protocol):
                    continue
                name = r.get("name")
                frm = r.get("from")
                to = r.get("to")
                parts.append(f"policy={name} from={frm} to={to}")
    return "; ".join(parts)


def k8s_semantics(cfg: Dict[str, Any]) -> str:
    return _semantics(cfg)
