from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .intent import IntentNode
from .openai_llm import OpenAILLM


def intent_from_dict(d: Dict[str, Any]) -> IntentNode:
    op = d.get("op")
    if op == "PRIMITIVE":
        name = d.get("name")
        params = d.get("params") or {}
        if not isinstance(params, dict):
            raise ValueError("Invalid params")
        return IntentNode.primitive(str(name), **params)

    if op in {"AND", "OR"}:
        children = d.get("children") or []
        if not isinstance(children, list) or len(children) < 2:
            raise ValueError(f"{op} expects children")
        node = intent_from_dict(children[0])
        for ch in children[1:]:
            rhs = intent_from_dict(ch)
            node = (node & rhs) if op == "AND" else (node | rhs)
        return node

    if op == "NOT":
        children = d.get("children") or []
        if not isinstance(children, list) or len(children) != 1:
            raise ValueError("NOT expects exactly one child")
        return ~intent_from_dict(children[0])

    raise ValueError(f"Unknown op: {op}")


@dataclass
class OpenAIIntentParser:
    model: str = "gpt-4o-mini"
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        self._llm = OpenAILLM(model=self.model, seed=self.seed)

    def parse(self, text: str) -> IntentNode:
        schema = {
            "type": "object",
            "properties": {
                "op": {"type": "string", "enum": ["PRIMITIVE", "AND", "OR", "NOT"]},
                "name": {"type": "string"},
                "params": {"type": "object"},
                "children": {"type": "array"},
            },
            "required": ["op"],
        }

        prompt = {
            "task": "Parse a natural language network intent into an IntentLoom intent tree.",
            "allowed_primitives": [
                {"name": "Isolate", "params": {"a": "string", "b": "string"}},
                {"name": "Reach", "params": {"a": "string", "b": "string"}},
                {"name": "Allow", "params": {"src": "string", "dst": "string"}},
                {"name": "Deny", "params": {"src": "string", "dst": "string"}},
                {"name": "Require", "params": {"key": "string", "equals": "string"}},
            ],
            "output_schema": schema,
            "input": text,
            "rules": [
                "Use PRIMITIVE with fields: op='PRIMITIVE', name, params.",
                "For AND/OR use op and children list.",
                "For NOT use op='NOT' and exactly one child.",
                "Return ONLY the JSON object.",
            ],
        }

        system = (
            "You convert natural language network intents into an IntentLoom intent tree. "
            "Return ONLY a valid JSON object matching the provided schema. "
            "No markdown, no code fences, no extra text."
        )
        out = self._llm.generate(prompt, system=system)
        return intent_from_dict(out)
