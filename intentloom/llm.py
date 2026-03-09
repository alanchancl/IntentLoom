from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

from .compiler import canonicalize_node


class LLM(Protocol):
    def generate(self, prompt: Dict[str, Any]) -> Dict[str, Any]:
        ...


@dataclass
class MockLLM:
    candidates: List[Dict[str, Any]]
    _i: int = 0

    def generate(self, prompt: Dict[str, Any]) -> Dict[str, Any]:
        if not self.candidates:
            return {}
        idx = min(self._i, len(self.candidates) - 1)
        self._i += 1
        return dict(self.candidates[idx])


class RuleBasedLLM:
    def __init__(self, initial_cfg: Dict[str, Any]):
        self._cfg = dict(initial_cfg)

    def _find_path(self, links: List[List[Any]], src: Any, dst: Any) -> Optional[List[str]]:
        s = canonicalize_node(src)
        t = canonicalize_node(dst)
        if not s or not t:
            return None
        if s == t:
            return [s]

        g: Dict[str, List[str]] = {}
        for it in links:
            if not isinstance(it, (list, tuple)) or len(it) < 2:
                continue
            a = canonicalize_node(it[0])
            b = canonicalize_node(it[1])
            if not a or not b:
                continue
            g.setdefault(a, []).append(b)
            g.setdefault(b, []).append(a)
        if s not in g or t not in g:
            return None

        q: List[str] = [s]
        parent: Dict[str, Optional[str]] = {s: None}
        while q:
            cur = q.pop(0)
            for nxt in g.get(cur, []):
                if nxt in parent:
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

    def generate(self, prompt: Dict[str, Any]) -> Dict[str, Any]:
        cfg = {
            "links": [list(x) for x in self._cfg.get("links", [])],
            "acl": [dict(x) for x in self._cfg.get("acl", [])],
        }
        violations = prompt.get("verification", {}).get("violations", [])
        for v in violations:
            if v.get("code") == "ISOLATE_VIOLATION":
                a = v.get("meta", {}).get("a")
                b = v.get("meta", {}).get("b")
                if a and b:
                    p = self._find_path(cfg.get("links", []), a, b)
                    if p and len(p) >= 2:
                        u = p[0]
                        w = p[1]
                        cfg["links"] = [
                            l
                            for l in cfg["links"]
                            if not (
                                isinstance(l, (list, tuple))
                                and len(l) >= 2
                                and ({canonicalize_node(l[0]), canonicalize_node(l[1])} == {u, w})
                            )
                        ]
            if v.get("code") == "REACH_VIOLATION":
                a = v.get("meta", {}).get("a")
                b = v.get("meta", {}).get("b")
                if a and b:
                    a_n = canonicalize_node(a)
                    b_n = canonicalize_node(b)
                    if a_n and b_n and not any(
                        {canonicalize_node(l[0]), canonicalize_node(l[1])} == {a_n, b_n}
                        for l in cfg["links"]
                        if isinstance(l, (list, tuple)) and len(l) >= 2
                    ):
                        cfg["links"].append([a_n, b_n])
            if v.get("code") == "ALLOW_VIOLATION":
                src = v.get("meta", {}).get("src")
                dst = v.get("meta", {}).get("dst")
                if src and dst:
                    src_n = canonicalize_node(src)
                    dst_n = canonicalize_node(dst)
                    if src_n and dst_n:
                        cfg.setdefault("acl", []).append(
                            {"src": src_n, "dst": dst_n, "action": "allow"}
                        )
        self._cfg = cfg
        return cfg
