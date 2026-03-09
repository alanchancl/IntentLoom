from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

from .compiler import Violation, canonicalize_config, canonicalize_node


def symbolicai_available() -> bool:
    try:
        import symai  # noqa: F401

        return True
    except Exception:
        return False


@dataclass(frozen=True)
class ContractSpec:
    name: str
    intent: Dict[str, Any]
    contract_cls: Type[Any]


def build_contract_for_primitive(intent: Dict[str, Any]) -> ContractSpec:
    try:
        import symai
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "SymbolicAI (symai) is not available. Install dependencies first: pip install symbolicai"
        ) from e

    try:
        from symai.strategy import contract as symai_contract
    except Exception:
        try:
            from symai import contract as symai_contract
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "SymbolicAI (symai) is installed but 'contract' could not be imported. "
                "Your symai version may not expose the contract decorator at the expected import path. "
                "Try upgrading/downgrading symbolicai, or verify whether 'from symai.strategy import contract' works. "
                f"symai_version={getattr(symai, '__version__', 'unknown')}"
            ) from e

    try:
        from symai.models import LLMDataModel
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "SymbolicAI (symai) is installed but required API 'symai.models.LLMDataModel' is missing. "
            f"symai_version={getattr(symai, '__version__', 'unknown')}"
        ) from e

    class ContractOk(LLMDataModel):
        value: bool

    name = intent.get("name")
    params = intent.get("params", {})
    path = tuple(intent.get("path", []))

    if name == "Isolate":
        a = params.get("a")
        b = params.get("b")
        prompt = f"Verify that there is no connectivity/path between {a} and {b}."

        @symai_contract(pre_remedy=False, post_remedy=False, verbose=False)
        class IsolateContract:
            prompt: str = ""

            def pre(self, input) -> bool:
                cfg = getattr(input, "value", input)
                c = canonicalize_config(cfg)
                links = c.get("links", [])
                p = _find_path(links, canonicalize_node(a), canonicalize_node(b))
                if p is None:
                    return True
                raise ValueError(
                    f"Isolate violated: found path {p}. Suggested fix: remove/break connectivity on this path."
                )

            def forward(self, cfg: dict) -> ContractOk:
                return ContractOk(value=True)

        IsolateContract.prompt = prompt
        return ContractSpec(name="Isolate", intent=intent, contract_cls=IsolateContract)

    if name == "Reach":
        a = params.get("a")
        b = params.get("b")
        prompt = f"Verify that there is connectivity/path between {a} and {b}."

        @symai_contract(pre_remedy=False, post_remedy=False, verbose=False)
        class ReachContract:
            prompt: str = ""

            def pre(self, input) -> bool:
                cfg = getattr(input, "value", input)
                c = canonicalize_config(cfg)
                links = c.get("links", [])
                p = _find_path(links, canonicalize_node(a), canonicalize_node(b))
                if p is not None:
                    return True
                raise ValueError(
                    f"Reach violated: no path between {a} and {b}. Suggested fix: add connectivity (e.g., a direct link) to establish reachability."
                )

            def forward(self, cfg: dict) -> ContractOk:
                return ContractOk(value=True)

        ReachContract.prompt = prompt
        return ContractSpec(name="Reach", intent=intent, contract_cls=ReachContract)

    if name == "Allow":
        src = params.get("src")
        dst = params.get("dst")
        prompt = f"Verify that an allow rule exists from {src} to {dst}."

        @symai_contract(pre_remedy=False, post_remedy=False, verbose=False)
        class AllowContract:
            prompt: str = ""

            def pre(self, input) -> bool:
                cfg = getattr(input, "value", input)
                c = canonicalize_config(cfg)
                acl = c.get("acl", [])
                src_n = canonicalize_node(src)
                dst_n = canonicalize_node(dst)
                ok = any(
                    r.get("src") == src_n and r.get("dst") == dst_n and r.get("action") == "allow" for r in acl
                )
                if ok:
                    return True
                raise ValueError(
                    f"Allow violated: no allow rule from {src} to {dst}. Suggested fix: add an ACL rule allowing traffic."
                )

            def forward(self, cfg: dict) -> ContractOk:
                return ContractOk(value=True)

        AllowContract.prompt = prompt
        return ContractSpec(name="Allow", intent=intent, contract_cls=AllowContract)

    if name == "Deny":
        src = params.get("src")
        dst = params.get("dst")
        prompt = f"Verify that no allow rule exists from {src} to {dst}."

        @symai_contract(pre_remedy=False, post_remedy=False, verbose=False)
        class DenyContract:
            prompt: str = ""

            def pre(self, input) -> bool:
                cfg = getattr(input, "value", input)
                c = canonicalize_config(cfg)
                acl = c.get("acl", [])
                src_n = canonicalize_node(src)
                dst_n = canonicalize_node(dst)
                has_allow = any(
                    r.get("src") == src_n and r.get("dst") == dst_n and r.get("action") == "allow" for r in acl
                )
                if not has_allow:
                    return True
                raise ValueError(
                    f"Deny violated: found allow rule from {src} to {dst}. Suggested fix: remove/override the allow rule to deny traffic."
                )

            def forward(self, cfg: dict) -> ContractOk:
                return ContractOk(value=True)

        DenyContract.prompt = prompt
        return ContractSpec(name="Deny", intent=intent, contract_cls=DenyContract)

    if name == "Require":
        key = params.get("key")
        equals = params.get("equals")
        prompt = f"Verify that cfg[{key}] == {equals}."

        @symai_contract(pre_remedy=False, post_remedy=False, verbose=False)
        class RequireContract:
            prompt: str = ""

            def pre(self, input) -> bool:
                cfg = getattr(input, "value", input)
                c = canonicalize_config(cfg)
                ok = str(c.get(str(key))) == str(equals)
                if ok:
                    return True
                raise ValueError(
                    f"Require violated: cfg[{key}] != {equals}. Suggested fix: set the required key/value."
                )

            def forward(self, cfg: dict) -> ContractOk:
                return ContractOk(value=True)

        RequireContract.prompt = prompt
        return ContractSpec(name="Require", intent=intent, contract_cls=RequireContract)

    raise ValueError(f"Unsupported primitive for contract backend: {name}")


def evaluate_contract(spec: ContractSpec, cfg: Dict[str, Any]) -> Tuple[bool, Optional[Exception]]:
    inst = spec.contract_cls()
    pre = getattr(inst, "pre", None)
    if not callable(pre):
        return True, None
    try:
        ok = bool(pre(cfg))
        if ok:
            return True, None
        return False, ValueError("Contract pre-condition returned False")
    except Exception as e:
        return False, e


def violation_from_intent(intent: Dict[str, Any]) -> Violation:
    name = intent.get("name")
    params = intent.get("params", {})
    path = tuple(intent.get("path", []))

    if name == "Isolate":
        a = params.get("a")
        b = params.get("b")
        return Violation(
            code="ISOLATE_VIOLATION",
            message=f"Isolate violated: path exists between {a} and {b}",
            path=path,
            meta={"a": a, "b": b},
        )

    if name == "Reach":
        a = params.get("a")
        b = params.get("b")
        return Violation(
            code="REACH_VIOLATION",
            message=f"Reach violated: no path between {a} and {b}",
            path=path,
            meta={"a": a, "b": b},
        )

    if name == "Allow":
        src = params.get("src")
        dst = params.get("dst")
        return Violation(
            code="ALLOW_VIOLATION",
            message=f"Allow violated: no allow rule from {src} to {dst}",
            path=path,
            meta={"src": src, "dst": dst},
        )

    if name == "Deny":
        src = params.get("src")
        dst = params.get("dst")
        return Violation(
            code="DENY_VIOLATION",
            message=f"Deny violated: found allow rule from {src} to {dst}",
            path=path,
            meta={"src": src, "dst": dst},
        )

    if name == "Require":
        key = params.get("key")
        equals = params.get("equals")
        return Violation(
            code="REQUIRE_VIOLATION",
            message=f"Require violated: cfg[{key}] != {equals}",
            path=path,
            meta={"key": key, "equals": equals},
        )

    return Violation(
        code="UNKNOWN_PRIMITIVE",
        message=f"Unknown primitive: {name}",
        path=path,
        meta={"name": name, "params": params},
    )


def _neighbors(links: List[List[str]]) -> Dict[str, List[str]]:
    g: Dict[str, List[str]] = {}
    for u, v in links:
        g.setdefault(u, []).append(v)
        g.setdefault(v, []).append(u)
    return g


def _has_path(links: List[List[str]], src: str, dst: str) -> bool:
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


def _find_path(links: List[List[str]], src: str, dst: str) -> Optional[List[str]]:
    if src == dst:
        return [src]
    g = _neighbors(links)
    if src not in g or dst not in g:
        return None
    prev: Dict[str, Optional[str]] = {src: None}
    q: List[str] = [src]
    while q:
        cur = q.pop(0)
        for nxt in g.get(cur, []):
            if nxt in prev:
                continue
            prev[nxt] = cur
            if nxt == dst:
                # reconstruct
                path: List[str] = [dst]
                p = cur
                while p is not None:
                    path.append(p)
                    p = prev.get(p)
                path.reverse()
                return path
            q.append(nxt)
    return None
