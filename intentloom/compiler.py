from __future__ import annotations

import re

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

from .intent import IntentNode


@dataclass(frozen=True)
class Violation:
    code: str
    message: str
    path: Tuple[str, ...] = ()
    meta: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "path": list(self.path),
            "meta": dict(self.meta or {}),
        }


CheckFn = Callable[[Dict[str, Any]], List[Violation]]


@dataclass(frozen=True)
class Constraint:
    name: str
    intent: Dict[str, Any]
    check: CheckFn


@dataclass(frozen=True)
class CompiledIntent:
    constraints: List[Constraint]
    feedback_schema: Dict[str, Any]


_CIDR_RE = re.compile(r"^(\d{1,3}(?:\.\d{1,3}){3})/(\d{1,2})$")
_IPV4_RE = re.compile(r"^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$")


def canonicalize_node(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        s = x.strip().lower()
    else:
        s = str(x).strip().lower()

    if not s:
        return ""

    s = re.sub(r"\s*/\s*", "/", s)
    m = _CIDR_RE.match(s)
    if m:
        s = m.group(1)

    m2 = _IPV4_RE.match(s)
    if m2:
        try:
            a = int(m2.group(1))
            b = int(m2.group(2))
            c = int(m2.group(3))
            d = int(m2.group(4))
        except ValueError:
            return s

        if 0 <= a <= 255 and 0 <= b <= 255 and 0 <= c <= 255 and 0 <= d <= 255:
            if a == 100 and b == 0:
                return f"100.0.{c}.0"
    return s


def canonicalize_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(cfg, dict):
        return {"links": [], "acl": []}

    out: Dict[str, Any] = dict(cfg)

    links_in = cfg.get("links", [])
    links_out: List[List[str]] = []
    if isinstance(links_in, list):
        for it in links_in:
            u: Any = None
            v: Any = None
            if isinstance(it, dict):
                u = it.get("u")
                v = it.get("v")
            elif isinstance(it, (list, tuple)) and len(it) >= 2:
                u = it[0]
                v = it[1]

            u_s = canonicalize_node(u)
            v_s = canonicalize_node(v)
            if not u_s or not v_s:
                continue
            links_out.append([u_s, v_s])
    out["links"] = links_out

    acl_in = cfg.get("acl", [])
    acl_out: List[Dict[str, str]] = []
    if isinstance(acl_in, list):
        for r in acl_in:
            if not isinstance(r, dict):
                continue
            src = canonicalize_node(r.get("src"))
            dst = canonicalize_node(r.get("dst"))
            action_raw = r.get("action")
            action = action_raw.strip().lower() if isinstance(action_raw, str) else ""
            if action in {"permit", "permitted"}:
                action = "allow"
            if action in {"block", "blocked", "denied"}:
                action = "deny"

            if not src or not dst:
                continue
            if action not in {"allow", "deny"}:
                continue
            acl_out.append({"src": src, "dst": dst, "action": action})
    out["acl"] = acl_out

    return out


def _neighbors(links: Iterable[Sequence[str]]) -> Dict[str, List[str]]:
    g: Dict[str, List[str]] = {}
    for u, v in links:
        g.setdefault(u, []).append(v)
        g.setdefault(v, []).append(u)
    return g


def _has_path(links: Iterable[Sequence[str]], src: str, dst: str) -> bool:
    if src == dst:
        return True
    g = _neighbors(links)
    if src not in g or dst not in g:
        return False
    seen = {src}
    q = [src]
    while q:
        cur = q.pop(0)
        for nxt in g.get(cur, []):
            if nxt == dst:
                return True
            if nxt in seen:
                continue
            seen.add(nxt)
            q.append(nxt)
    return False


def _compile_primitive(prim: Dict[str, Any], path: Tuple[str, ...]) -> List[Constraint]:
    name = prim["name"]
    params = prim.get("params", {})

    if name == "Isolate":
        a = params["a"]
        b = params["b"]

        def check(cfg: Dict[str, Any]) -> List[Violation]:
            c = canonicalize_config(cfg)
            a_n = canonicalize_node(a)
            b_n = canonicalize_node(b)
            links = c.get("links", [])
            if _has_path(links, a_n, b_n):
                return [
                    Violation(
                        code="ISOLATE_VIOLATION",
                        message=f"Isolate violated: path exists between {a} and {b}",
                        path=path,
                        meta={"a": a, "b": b},
                    )
                ]
            return []

        return [
            Constraint(
                name="Isolate",
                intent={"name": name, "params": params, "path": list(path)},
                check=check,
            )
        ]

    if name == "Reach":
        a = params["a"]
        b = params["b"]

        def check(cfg: Dict[str, Any]) -> List[Violation]:
            c = canonicalize_config(cfg)
            a_n = canonicalize_node(a)
            b_n = canonicalize_node(b)
            links = c.get("links", [])
            if not _has_path(links, a_n, b_n):
                return [
                    Violation(
                        code="REACH_VIOLATION",
                        message=f"Reach violated: no path between {a} and {b}",
                        path=path,
                        meta={"a": a, "b": b},
                    )
                ]
            return []

        return [
            Constraint(
                name="Reach",
                intent={"name": name, "params": params, "path": list(path)},
                check=check,
            )
        ]

    if name == "Allow":
        src = params["src"]
        dst = params["dst"]

        def check(cfg: Dict[str, Any]) -> List[Violation]:
            c = canonicalize_config(cfg)
            src_n = canonicalize_node(src)
            dst_n = canonicalize_node(dst)
            acl = c.get("acl", [])
            ok = any(
                r.get("src") == src_n and r.get("dst") == dst_n and r.get("action") == "allow" for r in acl
            )
            if not ok:
                return [
                    Violation(
                        code="ALLOW_VIOLATION",
                        message=f"Allow violated: no allow rule from {src} to {dst}",
                        path=path,
                        meta={"src": src, "dst": dst},
                    )
                ]
            return []

        return [
            Constraint(
                name="Allow",
                intent={"name": name, "params": params, "path": list(path)},
                check=check,
            )
        ]

    if name == "Deny":
        src = params["src"]
        dst = params["dst"]

        def check(cfg: Dict[str, Any]) -> List[Violation]:
            c = canonicalize_config(cfg)
            src_n = canonicalize_node(src)
            dst_n = canonicalize_node(dst)
            acl = c.get("acl", [])
            bad = any(
                r.get("src") == src_n and r.get("dst") == dst_n and r.get("action") == "allow" for r in acl
            )
            if bad:
                return [
                    Violation(
                        code="DENY_VIOLATION",
                        message=f"Deny violated: found allow rule from {src} to {dst}",
                        path=path,
                        meta={"src": src, "dst": dst},
                    )
                ]
            return []

        return [
            Constraint(
                name="Deny",
                intent={"name": name, "params": params, "path": list(path)},
                check=check,
            )
        ]

    if name == "Require":
        key = params["key"]
        equals = params["equals"]

        def check(cfg: Dict[str, Any]) -> List[Violation]:
            c = canonicalize_config(cfg)
            if str(c.get(key)) != str(equals):
                return [
                    Violation(
                        code="REQUIRE_VIOLATION",
                        message=f"Require violated: cfg[{key}] != {equals}",
                        path=path,
                        meta={"key": key, "equals": equals},
                    )
                ]
            return []

        return [
            Constraint(
                name="Require",
                intent={"name": name, "params": params, "path": list(path)},
                check=check,
            )
        ]

    def check_unknown(_: Dict[str, Any]) -> List[Violation]:
        return [
            Violation(
                code="UNKNOWN_PRIMITIVE",
                message=f"Unknown primitive: {name}",
                path=path,
                meta={"name": name, "params": params},
            )
        ]

    return [
        Constraint(
            name=name,
            intent={"name": name, "params": params, "path": list(path)},
            check=check_unknown,
        )
    ]


def _compile(node: IntentNode, path: Tuple[str, ...]) -> List[Constraint]:
    if node.op == "PRIMITIVE":
        return _compile_primitive(node.value, path)

    if node.op == "AND":
        cs: List[Constraint] = []
        for i, ch in enumerate(node.children):
            cs.extend(_compile(ch, path + (f"AND[{i}]",)))
        return cs

    if node.op == "OR":
        compiled_children = [
            _compile(ch, path + (f"OR[{i}]",)) for i, ch in enumerate(node.children)
        ]

        def check(cfg: Dict[str, Any]) -> List[Violation]:
            branches_ok = []
            branch_violations: List[List[Violation]] = []
            for branch in compiled_children:
                v: List[Violation] = []
                for c in branch:
                    v.extend(c.check(cfg))
                branch_violations.append(v)
                branches_ok.append(len(v) == 0)
            if any(branches_ok):
                return []
            flat = [item for sub in branch_violations for item in sub]
            if flat:
                return flat
            return [
                Violation(
                    code="OR_VIOLATION",
                    message="OR violated: no branch satisfied",
                    path=path,
                    meta={},
                )
            ]

        return [Constraint(name="OR", intent=node.to_dict(), check=check)]

    if node.op == "NOT":
        if len(node.children) != 1:
            raise ValueError("NOT expects exactly one child")
        child_constraints = _compile(node.children[0], path + ("NOT[0]",))

        def check(cfg: Dict[str, Any]) -> List[Violation]:
            v: List[Violation] = []
            for c in child_constraints:
                v.extend(c.check(cfg))
            if len(v) == 0:
                return [
                    Violation(
                        code="NOT_VIOLATION",
                        message="NOT violated: child intent is satisfied but expected to be unsatisfied",
                        path=path,
                        meta={},
                    )
                ]
            return []

        return [Constraint(name="NOT", intent=node.to_dict(), check=check)]

    raise ValueError(f"Unknown op: {node.op}")


def compile_intent(intent: IntentNode) -> CompiledIntent:
    constraints = _compile(intent, path=())
    feedback_schema = {
        "intent": intent.to_dict(),
        "constraints": [
            {"name": c.name, "intent": c.intent} for c in constraints
        ],
    }
    return CompiledIntent(constraints=constraints, feedback_schema=feedback_schema)
