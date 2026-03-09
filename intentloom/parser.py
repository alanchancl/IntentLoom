from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from .intent import IntentNode
from .primitives import Allow, Deny, Isolate, Reach, Require


@dataclass(frozen=True)
class _Token:
    kind: str
    text: str


_OPS = {"AND", "OR", "NOT", "(", ")", ","}


def _parse_kv_args(words: List[str]) -> tuple[List[str], Optional[int], Optional[str]]:
    rest: List[str] = []
    port: Optional[int] = None
    protocol: Optional[str] = None
    for w in words:
        m = re.match(r"^(port|protocol)\s*[:=]\s*(.+)$", str(w), flags=re.IGNORECASE)
        if not m:
            rest.append(w)
            continue
        k = m.group(1).strip().lower()
        v = m.group(2).strip()
        if k == "port":
            try:
                port = int(v)
            except Exception:
                port = None
            continue
        if k == "protocol":
            protocol = v
            continue
    return rest, port, protocol


def _tokenize(s: str) -> List[_Token]:
    raw = re.findall(r"\(|\)|,|\bAND\b|\bOR\b|\bNOT\b|[^\s(),]+", s, flags=re.IGNORECASE)
    toks: List[_Token] = []
    for r in raw:
        u = r.upper()
        if u in _OPS:
            toks.append(_Token(kind=u, text=r))
        else:
            toks.append(_Token(kind="WORD", text=r))
    return toks


class _Parser:
    def __init__(self, tokens: List[_Token]):
        self._toks = tokens
        self._i = 0

    def _peek(self) -> Optional[_Token]:
        if self._i >= len(self._toks):
            return None
        return self._toks[self._i]

    def _pop(self) -> _Token:
        t = self._peek()
        if t is None:
            raise ValueError("Unexpected end of input")
        self._i += 1
        return t

    def _accept(self, kind: str) -> bool:
        t = self._peek()
        if t is None:
            return False
        if t.kind == kind:
            self._i += 1
            return True
        return False

    def _expect(self, kind: str) -> _Token:
        t = self._pop()
        if t.kind != kind:
            raise ValueError(f"Expected {kind}, got {t.kind}:{t.text}")
        return t

    def parse(self) -> IntentNode:
        node = self._parse_or()
        if self._peek() is not None:
            t = self._peek()
            raise ValueError(f"Unexpected token at end: {t.kind}:{t.text}")
        return node

    def _parse_or(self) -> IntentNode:
        node = self._parse_and()
        while self._accept("OR"):
            rhs = self._parse_and()
            node = node | rhs
        return node

    def _parse_and(self) -> IntentNode:
        node = self._parse_not()
        while self._accept("AND"):
            rhs = self._parse_not()
            node = node & rhs
        return node

    def _parse_not(self) -> IntentNode:
        if self._accept("NOT"):
            return ~self._parse_not()
        return self._parse_atom()

    def _parse_atom(self) -> IntentNode:
        if self._accept("("):
            node = self._parse_or()
            self._expect(")")
            return node
        return self._parse_primitive()

    def _read_until_boundary(self) -> List[str]:
        args: List[str] = []
        while True:
            t = self._peek()
            if t is None:
                break
            if t.kind in {"AND", "OR", "NOT", "(", ")"}:
                break
            if t.kind == ",":
                break
            args.append(self._pop().text)
        return args

    def _parse_args_paren(self) -> List[str]:
        self._expect("(")
        parts: List[str] = []
        cur: List[str] = []
        while True:
            t = self._peek()
            if t is None:
                raise ValueError("Unclosed '('")
            if t.kind == ")":
                self._pop()
                if cur:
                    parts.append(" ".join(cur).strip())
                break
            if t.kind == ",":
                self._pop()
                parts.append(" ".join(cur).strip())
                cur = []
                continue
            cur.append(self._pop().text)
        return [p for p in (x.strip() for x in parts) if p]

    def _parse_primitive(self) -> IntentNode:
        t = self._peek()
        if t is None:
            raise ValueError("Expected primitive, got end of input")

        head = self._pop().text
        head_u = head.upper()

        alias = {
            "隔离": "ISOLATE",
            "可达": "REACH",
            "允许": "ALLOW",
            "拒绝": "DENY",
            "要求": "REQUIRE",
        }
        head_u = alias.get(head, head_u)

        nxt = self._peek()
        if nxt is not None and nxt.kind == "(":
            args = self._parse_args_paren()
            return self._build_prim(head_u, args)

        words = self._read_until_boundary()
        return self._build_prim(head_u, words)

    def _build_prim(self, name_u: str, words: List[str]) -> IntentNode:
        def _strip_kw(ws: List[str], kw: str) -> List[str]:
            return [w for w in ws if w.upper() != kw]

        if name_u in {"ISOLATE", "ISOLATE"}:
            ws = _strip_kw(words, "FROM")
            ws, port, protocol = _parse_kv_args(ws)
            if len(ws) < 2:
                raise ValueError("Isolate expects two entities")
            return Isolate(ws[0], ws[1], port=port, protocol=protocol)

        if name_u == "REACH":
            ws = _strip_kw(words, "TO")
            ws, port, protocol = _parse_kv_args(ws)
            if len(ws) < 2:
                raise ValueError("Reach expects two entities")
            return Reach(ws[0], ws[1], port=port, protocol=protocol)

        if name_u == "ALLOW":
            ws = _strip_kw(words, "TO")
            if len(ws) < 2:
                raise ValueError("Allow expects src and dst")
            return Allow(ws[0], ws[1])

        if name_u == "DENY":
            ws = _strip_kw(words, "TO")
            if len(ws) < 2:
                raise ValueError("Deny expects src and dst")
            return Deny(ws[0], ws[1])

        if name_u == "REQUIRE":
            ws = _strip_kw(words, "EQUALS")
            ws = _strip_kw(ws, "=")
            if len(ws) < 2:
                raise ValueError("Require expects key and value")
            return Require(ws[0], ws[1])

        raise ValueError(f"Unknown primitive: {name_u}")


def parse_intent(text: str) -> IntentNode:
    tokens = _tokenize(text)
    if not tokens:
        raise ValueError("Empty intent")
    return _Parser(tokens).parse()
