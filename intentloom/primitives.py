from __future__ import annotations

from .intent import IntentNode


def Isolate(a: str, b: str, *, port: int | None = None, protocol: str | None = None) -> IntentNode:
    params = {"a": a, "b": b}
    if port is not None:
        params["port"] = int(port)
    if isinstance(protocol, str) and protocol.strip():
        params["protocol"] = protocol.strip()
    return IntentNode.primitive("Isolate", **params)


def Reach(a: str, b: str, *, port: int | None = None, protocol: str | None = None) -> IntentNode:
    params = {"a": a, "b": b}
    if port is not None:
        params["port"] = int(port)
    if isinstance(protocol, str) and protocol.strip():
        params["protocol"] = protocol.strip()
    return IntentNode.primitive("Reach", **params)


def Allow(src: str, dst: str) -> IntentNode:
    return IntentNode.primitive("Allow", src=src, dst=dst)


def Deny(src: str, dst: str) -> IntentNode:
    return IntentNode.primitive("Deny", src=src, dst=dst)


def Require(key: str, equals: str) -> IntentNode:
    return IntentNode.primitive("Require", key=key, equals=equals)
