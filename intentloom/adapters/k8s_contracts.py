from __future__ import annotations

import sys
from typing import Any, Dict

from intentloom.compiler import canonicalize_node
from intentloom.symbolicai_contracts import ContractSpec

from .k8s_adapter import build_k8s_graph, find_path, k8s_semantics, summarize_path_reasons


def build_contract_for_k8s_primitive(intent: Dict[str, Any]) -> ContractSpec:
    try:
        import symai
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "SymbolicAI (symai) is not available. Ensure you run with the interpreter where 'symbolicai' is installed. "
            f"python_executable={sys.executable}"
        ) from e

    try:
        from symai.strategy import contract as symai_contract
    except Exception:
        try:
            from symai import contract as symai_contract
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "SymbolicAI (symai) is installed but 'contract' could not be imported. "
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

    if name == "Isolate":
        a = canonicalize_node(params.get("a"))
        b = canonicalize_node(params.get("b"))
        port = params.get("port")
        protocol = params.get("protocol")
        port_i = int(port) if isinstance(port, int) or (isinstance(port, str) and str(port).strip().isdigit()) else None
        proto_s = str(protocol).strip() if isinstance(protocol, str) and str(protocol).strip() else None
        extra = ""
        if port_i is not None:
            extra = f" on port={port_i}" + (f" protocol={proto_s}" if proto_s else "")
        prompt = f"Verify that there is no connectivity/path between {a} and {b}{extra} under Kubernetes NetworkPolicy semantics."

        @symai_contract(pre_remedy=False, post_remedy=False, verbose=False)
        class K8sIsolateContract:
            prompt: str = ""

            def pre(self, input) -> bool:
                cfg = getattr(input, "value", input)
                directed = k8s_semantics(cfg) == "option_c"
                g = build_k8s_graph(cfg)
                p = find_path(g.links, a, b, edge_reasons=g.edge_reasons, port=port_i, protocol=proto_s, directed=directed)
                direction = "a->b"
                if p is None and directed:
                    p = find_path(g.links, b, a, edge_reasons=g.edge_reasons, port=port_i, protocol=proto_s, directed=directed)
                    if p is not None:
                        direction = "b->a"
                if p is None:
                    return True
                evidence = summarize_path_reasons(p, g.edge_reasons, port=port_i, protocol=proto_s, directed=directed)
                msg = f"Isolate violated: found access path {p}."
                if port_i is not None:
                    msg += f" (port={port_i}" + (f", protocol={proto_s}" if proto_s else "") + ")"
                if directed:
                    msg += f" (direction={direction})"
                if evidence:
                    msg += f" Evidence: {evidence}."
                msg += " Suggested fix: narrow or remove the policy/rule(s) enabling this cross-boundary connectivity (e.g., adjust namespace or podSelector labels)."
                raise ValueError(msg)

            def forward(self, cfg: dict) -> ContractOk:
                return ContractOk(value=True)

        K8sIsolateContract.prompt = prompt
        return ContractSpec(name="Isolate", intent=intent, contract_cls=K8sIsolateContract)

    if name == "Reach":
        a = canonicalize_node(params.get("a"))
        b = canonicalize_node(params.get("b"))
        port = params.get("port")
        protocol = params.get("protocol")
        port_i = int(port) if isinstance(port, int) or (isinstance(port, str) and str(port).strip().isdigit()) else None
        proto_s = str(protocol).strip() if isinstance(protocol, str) and str(protocol).strip() else None
        extra = ""
        if port_i is not None:
            extra = f" on port={port_i}" + (f" protocol={proto_s}" if proto_s else "")
        prompt = f"Verify that there is connectivity/path between {a} and {b}{extra} under Kubernetes NetworkPolicy semantics."

        @symai_contract(pre_remedy=False, post_remedy=False, verbose=False)
        class K8sReachContract:
            prompt: str = ""

            def pre(self, input) -> bool:
                cfg = getattr(input, "value", input)
                directed = k8s_semantics(cfg) == "option_c"
                g = build_k8s_graph(cfg)
                p = find_path(g.links, a, b, edge_reasons=g.edge_reasons, port=port_i, protocol=proto_s, directed=directed)
                if p is not None:
                    return True
                raise ValueError(
                    f"Reach violated: no path between {a} and {b}." + (
                        f" (port={port_i}" + (f", protocol={proto_s}" if proto_s else "") + ")" if port_i is not None else ""
                    )
                    + " Suggested fix: add/adjust a policy to allow the intended communication (fix namespace and podSelector labels)."
                )

            def forward(self, cfg: dict) -> ContractOk:
                return ContractOk(value=True)

        K8sReachContract.prompt = prompt
        return ContractSpec(name="Reach", intent=intent, contract_cls=K8sReachContract)

    raise ValueError(f"Unsupported K8s primitive for contract backend: {name}")
