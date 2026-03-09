from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .compiler import Constraint, Violation
from .verifier import VerificationResult, VerifierLike
from .symbolicai_contracts import (
    build_contract_for_primitive,
    evaluate_contract,
    violation_from_intent,
)


@dataclass
class SymbolicAIContractVerifier(VerifierLike):
    """Verifier plugin that evaluates primitive constraints via SymbolicAI @contract.

    Notes:
    - Only primitives (Isolate/Reach/Allow/Deny/Require) are executed through contracts.
    - Composite constraints (OR/NOT) fall back to the existing Python check function.
    """

    _cache: Dict[Tuple[str, str], Any] = None

    def __post_init__(self) -> None:
        try:
            import symai  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "SymbolicAI (symai) is not available in the current Python environment. "
                "Ensure you run the experiment with the same interpreter where 'symbolicai' is installed. "
                f"python_executable={sys.executable}"
            ) from e
        if self._cache is None:
            self._cache = {}

    def verify(self, cfg: Dict[str, Any], constraints: List[Constraint]) -> VerificationResult:
        violations: List[Violation] = []

        for c in constraints:
            intent = dict(c.intent or {})
            # Primitive constraints compiled by our compiler have intent = {name, params, path?}
            if c.name in {"Isolate", "Reach", "Allow", "Deny", "Require"}:
                key = (c.name, str(intent.get("params", {})) + str(intent.get("path", [])))
                spec = self._cache.get(key)
                if spec is None:
                    spec = build_contract_for_primitive(intent)
                    self._cache[key] = spec

                ok, _exc = evaluate_contract(spec, cfg)
                if not ok:
                    v = violation_from_intent(intent)
                    meta = dict(v.meta or {})
                    if _exc is not None:
                        meta.update(
                            {
                                "contract_exception": str(_exc),
                                "contract_type": getattr(spec.contract_cls, "__name__", "<contract>"),
                                "contract_prompt": getattr(spec.contract_cls, "prompt", None),
                            }
                        )
                    violations.append(
                        Violation(code=v.code, message=v.message, path=v.path, meta=meta)
                    )
                continue

            # Fallback to Python checker for composites
            violations.extend(c.check(cfg))

        return VerificationResult(ok=len(violations) == 0, violations=violations)
