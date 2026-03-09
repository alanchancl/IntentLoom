from __future__ import annotations

from typing import Iterable, List

from .compiler import Violation


def render_feedback(violations: Iterable[Violation]) -> str:
    vs = list(violations)
    if not vs:
        return "All constraints satisfied."

    lines: List[str] = []
    for v in vs:
        if v.code == "ISOLATE_VIOLATION":
            a = (v.meta or {}).get("a")
            b = (v.meta or {}).get("b")
            lines.append(f"Ensure isolation between {a} and {b} (remove connectivity/path).")
            continue
        if v.code == "REACH_VIOLATION":
            a = (v.meta or {}).get("a")
            b = (v.meta or {}).get("b")
            lines.append(f"Ensure reachability from {a} to {b} (add route/link/permission).")
            continue
        if v.code == "ALLOW_VIOLATION":
            src = (v.meta or {}).get("src")
            dst = (v.meta or {}).get("dst")
            lines.append(f"Add an allow rule from {src} to {dst}.")
            continue
        if v.code == "DENY_VIOLATION":
            src = (v.meta or {}).get("src")
            dst = (v.meta or {}).get("dst")
            lines.append(f"Remove/avoid any allow rule from {src} to {dst}.")
            continue
        if v.code == "REQUIRE_VIOLATION":
            key = (v.meta or {}).get("key")
            eq = (v.meta or {}).get("equals")
            lines.append(f"Set `{key}` to `{eq}`.")
            continue
        lines.append(v.message)

    return "\n".join(lines)
