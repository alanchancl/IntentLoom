from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Tuple


IntentOp = Literal["PRIMITIVE", "AND", "OR", "NOT"]


@dataclass(frozen=True)
class IntentNode:
    op: IntentOp
    value: Any = None
    children: Tuple["IntentNode", ...] = ()

    @staticmethod
    def primitive(name: str, **params: Any) -> "IntentNode":
        return IntentNode(op="PRIMITIVE", value={"name": name, "params": params})

    def __and__(self, other: "IntentNode") -> "IntentNode":
        return IntentNode(op="AND", children=(self, other))

    def __or__(self, other: "IntentNode") -> "IntentNode":
        return IntentNode(op="OR", children=(self, other))

    def __invert__(self) -> "IntentNode":
        return IntentNode(op="NOT", children=(self,))

    def to_dict(self) -> Dict[str, Any]:
        if self.op == "PRIMITIVE":
            return {"op": self.op, **self.value}
        return {"op": self.op, "children": [c.to_dict() for c in self.children]}
