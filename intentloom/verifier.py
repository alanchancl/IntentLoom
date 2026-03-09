from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Protocol

from .compiler import Constraint, Violation


@dataclass
class VerificationResult:
    ok: bool
    violations: List[Violation]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "violations": [v.to_dict() for v in self.violations],
        }


class VerifierLike(Protocol):
    def verify(self, cfg: Dict[str, Any], constraints: List[Constraint]) -> VerificationResult:
        ...


class Verifier:
    def verify(self, cfg: Dict[str, Any], constraints: List[Constraint]) -> VerificationResult:
        violations: List[Violation] = []
        for c in constraints:
            violations.extend(c.check(cfg))
        return VerificationResult(ok=len(violations) == 0, violations=violations)
