import os
import sys
import json
import argparse
from pathlib import Path
import re
import random
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from intentloom.netconfeval_loader import find_dataset_root, list_dataset_files, load_records


_INTENT_KEYS = [
    "human_language",
    "intent",
    "requirement",
    "requirements",
    "nl_requirement",
    "natural_language",
    "nl",
    "prompt",
    "description",
    "question",
    "input",
]


def _extract_text(rec: Dict[str, Any]) -> Optional[str]:
    for k in _INTENT_KEYS:
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
            return "\n".join(x.strip() for x in v if x.strip())
    return None


def _step1_human_language_to_dsl(text: str) -> Optional[str]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None

    patterns: List[tuple[re.Pattern[str], str]] = [
        (re.compile(r"^Traffic originating from (?P<source>.+?) can reach the subnet (?P<subnet>.+?)\.$"), "Reach"),
        (re.compile(r"^(?P<source>.+?) can reach (?P<subnet>.+?)\.$"), "Reach"),
        (re.compile(r"^The subnet (?P<subnet>.+?) is reachable from (?P<source>.+?)\.$"), "Reach"),
        (re.compile(r"^Connectivity from (?P<source>.+?) to (?P<subnet>.+?) is established\.$"), "Reach"),
        (re.compile(r"^(?P<subnet>.+?) is accessible from (?P<source>.+?)\.$"), "Reach"),
        (re.compile(r"^(?P<source>.+?) cannot reach (?P<subnet>.+?)\.$"), "Isolate"),
    ]

    prims: List[str] = []
    for ln in lines:
        matched = False
        for pat, prim in patterns:
            m = pat.match(ln)
            if not m:
                continue
            src = m.group("source").strip()
            subnet = m.group("subnet").strip()
            if subnet.lower().startswith("the subnet "):
                subnet = subnet[len("the subnet ") :].strip()
            prims.append(f"{prim}({src}, {subnet})")
            matched = True
            break
        if not matched:
            return None

    return " AND ".join(prims)


def _dsl_step1_has_conflict(dsl: str) -> bool:
    if not isinstance(dsl, str):
        return False
    items = re.findall(r"\b(Isolate|Reach)\(([^,]+),\s*([^\)]+)\)", dsl)

    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        p = parent.get(x)
        if p is None:
            parent[x] = x
            return x
        if p == x:
            return x
        r = find(p)
        parent[x] = r
        return r

    def union(a: str, b: str) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    isolates: List[tuple[str, str]] = []

    for prim, a, b in items:
        a_n = a.strip().lower()
        b_n = b.strip().lower()
        if not a_n or not b_n:
            continue
        if prim == "Reach":
            union(a_n, b_n)
        else:
            isolates.append((a_n, b_n))

    for a_n, b_n in isolates:
        if find(a_n) == find(b_n):
            return True
    return False


def _parse_single_step1_prim(dsl: str) -> Optional[Dict[str, str]]:
    if not isinstance(dsl, str):
        return None
    m = re.fullmatch(r"\s*(Reach|Isolate)\(([^,]+),\s*([^\)]+)\)\s*", dsl)
    if not m:
        return None
    prim = m.group(1)
    a = m.group(2).strip()
    b = m.group(3).strip()
    if not a or not b:
        return None
    return {"prim": prim, "a": a, "b": b}


def _parse_step1_prims(dsl: str) -> List[Dict[str, str]]:
    if not isinstance(dsl, str):
        return []
    out: List[Dict[str, str]] = []
    for prim, a, b in re.findall(r"\b(Isolate|Reach)\(([^,]+),\s*([^\)]+)\)", dsl):
        a_s = a.strip()
        b_s = b.strip()
        if not a_s or not b_s:
            continue
        out.append({"prim": prim, "a": a_s, "b": b_s})
    return out


def _make_initial_cfg_for_intent(dsl: str, rng: random.Random) -> Optional[Dict[str, Any]]:
    prims = _parse_step1_prims(dsl)
    if not prims:
        return None

    reach = next((p for p in prims if p["prim"] == "Reach"), None)
    isolate = next((p for p in prims if p["prim"] == "Isolate"), None)

    links: List[List[str]] = []
    acl: List[Dict[str, str]] = []

    if reach is not None:
        if isolate is not None:
            links.append([reach["a"], reach["b"]])
        else:
            b = reach["b"].strip()
            m = re.match(r"^(100\.0\.(\d{1,3})\.)", b)
            if m:
                try:
                    x = int(m.group(2))
                except ValueError:
                    x = None
                if x is not None:
                    other = (x + 1) % 256
                    links.append([reach["a"], f"100.0.{other}.0"])
                else:
                    links.append([reach["a"], "router1"])
            else:
                links.append([reach["a"], "router1"])

        acl.append({"src": reach["a"], "dst": reach["b"], "action": "allow"})
    if isolate is not None:
        links.append([isolate["a"], isolate["b"]])

    if prims:
        p0 = prims[0]
        if rng.random() < 0.3:
            links.append([p0["a"], p0["a"]])

    return {"links": links, "acl": acl}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".\\data\\netconfeval")
    ap.add_argument("--out", type=str, default=".\\data\\netconfeval_tasks.json")
    ap.add_argument("--max", type=int, default=200)
    ap.add_argument("--llm", type=str, default="openai", choices=["openai", "rule_based", "mock"])
    ap.add_argument("--model", type=str, default="gpt-4o-mini")
    ap.add_argument("--base_url", type=str, default=None)
    ap.add_argument("--timeout", type=float, default=None)
    ap.add_argument("--max_rounds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--complex_ratio", type=float, default=0.0)
    ap.add_argument("--initial_cfg_ratio", type=float, default=0.0)
    ap.add_argument(
        "--drop_dsl_conflicts",
        action="store_true",
        help="If set, drop records whose extracted DSL is internally inconsistent under our semantics (e.g., Reach and Isolate on nodes that must be connected).",
    )
    ap.add_argument(
        "--conflict_filter",
        type=str,
        default="all",
        choices=["all", "only_true", "only_false"],
    )
    args = ap.parse_args()

    root = Path(args.root)
    dataset_root = find_dataset_root(root)
    files = list_dataset_files(dataset_root)

    tasks: List[Dict[str, Any]] = []
    skipped = 0

    rng = random.Random(args.seed)
    reach_by_src: Dict[str, List[Dict[str, Any]]] = {}
    isolate_by_src: Dict[str, List[Dict[str, Any]]] = {}
    simple_candidates: List[Dict[str, Any]] = []

    collect_limit = args.max
    if args.complex_ratio and args.max > 0:
        collect_limit = max(args.max * 10, args.max)

    for f in files:
        recs = load_records(f.path)
        for i, r in enumerate(recs):
            if len(tasks) >= collect_limit:
                break
            if not isinstance(r, dict):
                continue

            if args.conflict_filter != "all" and "conflict_exist" in r:
                ce = r.get("conflict_exist")
                if isinstance(ce, bool):
                    if args.conflict_filter == "only_true" and not ce:
                        continue
                    if args.conflict_filter == "only_false" and ce:
                        continue
            text = _extract_text(r)
            if not text:
                skipped += 1
                continue

            dsl = _step1_human_language_to_dsl(text)
            if not dsl:
                skipped += 1
                continue

            dsl_conflict = _dsl_step1_has_conflict(dsl)
            if args.conflict_filter == "only_true" and not dsl_conflict:
                continue
            if args.conflict_filter == "only_false" and dsl_conflict:
                continue
            if args.drop_dsl_conflicts and dsl_conflict:
                continue

            task_id = f"{f.path.stem}_{i}"
            if args.llm == "openai":
                llm_spec: Dict[str, Any] = {"type": "openai", "model": args.model}
                if args.base_url:
                    llm_spec["base_url"] = args.base_url
                if args.timeout is not None:
                    llm_spec["timeout"] = args.timeout
            elif args.llm == "rule_based":
                llm_spec = {"type": "rule_based", "initial_cfg": {"links": [], "acl": []}}
            else:
                llm_spec = {"type": "mock", "candidates": [{"links": [], "acl": []}]}

            item = {
                "id": task_id,
                "source": str(f.path),
                "intent": dsl,
                "raw_intent": text,
                "llm": llm_spec,
                "max_rounds": args.max_rounds,
            }

            tasks.append(item)

            one = _parse_single_step1_prim(dsl)
            if one is not None:
                src = one["a"].strip().lower()
                if one["prim"] == "Reach":
                    reach_by_src.setdefault(src, []).append(item)
                else:
                    isolate_by_src.setdefault(src, []).append(item)
                simple_candidates.append(item)
            else:
                simple_candidates.append(item)
        if len(tasks) >= collect_limit:
            break

    if args.complex_ratio and args.max > 0:
        complex_target = int(args.max * float(args.complex_ratio))
        if complex_target < 0:
            complex_target = 0
        if complex_target > args.max:
            complex_target = args.max

        complex_tasks: List[Dict[str, Any]] = []
        used_ids: set[str] = set()

        common_srcs = [s for s in reach_by_src.keys() if s in isolate_by_src]
        rng.shuffle(common_srcs)

        for src in common_srcs:
            if len(complex_tasks) >= complex_target:
                break
            r_item = rng.choice(reach_by_src.get(src, []))
            i_item = rng.choice(isolate_by_src.get(src, []))

            r_one = _parse_single_step1_prim(r_item["intent"])
            i_one = _parse_single_step1_prim(i_item["intent"])
            if r_one is None or i_one is None:
                continue
            if r_one["b"].strip().lower() == i_one["b"].strip().lower():
                continue

            dsl = f"Reach({r_one['a']}, {r_one['b']}) AND Isolate({i_one['a']}, {i_one['b']})"
            if _dsl_step1_has_conflict(dsl):
                continue

            cid = f"complex_{r_item['id']}__{i_item['id']}"
            if cid in used_ids:
                continue
            used_ids.add(cid)

            complex_tasks.append(
                {
                    "id": cid,
                    "source": f"{r_item['source']};{i_item['source']}",
                    "intent": dsl,
                    "raw_intent": f"{r_item.get('raw_intent','')}\\n{i_item.get('raw_intent','')}",
                    "llm": dict(r_item["llm"]),
                    "max_rounds": r_item["max_rounds"],
                }
            )

        if len(complex_tasks) < complex_target:
            reach_srcs = [s for s, xs in reach_by_src.items() if len(xs) >= 2]
            rng.shuffle(reach_srcs)
            for src in reach_srcs:
                if len(complex_tasks) >= complex_target:
                    break
                xs = reach_by_src.get(src, [])
                if len(xs) < 2:
                    continue
                r_item = rng.choice(xs)
                i_item = rng.choice([z for z in xs if z["id"] != r_item["id"]])
                r_one = _parse_single_step1_prim(r_item["intent"])
                i_one = _parse_single_step1_prim(i_item["intent"])
                if r_one is None or i_one is None:
                    continue
                if r_one["b"].strip().lower() == i_one["b"].strip().lower():
                    continue
                dsl = f"Reach({r_one['a']}, {r_one['b']}) AND Isolate({r_one['a']}, {i_one['b']})"
                if _dsl_step1_has_conflict(dsl):
                    continue
                cid = f"complex_{r_item['id']}__synthiso__{i_item['id']}"
                if cid in used_ids:
                    continue
                used_ids.add(cid)
                complex_tasks.append(
                    {
                        "id": cid,
                        "source": f"{r_item['source']};{i_item['source']}",
                        "intent": dsl,
                        "raw_intent": f"{r_item.get('raw_intent','')}\\n{i_item.get('raw_intent','')}",
                        "llm": dict(r_item["llm"]),
                        "max_rounds": r_item["max_rounds"],
                    }
                )

        remaining = args.max - len(complex_tasks)
        rng.shuffle(simple_candidates)
        seen_simple: set[str] = set()
        simple_out: List[Dict[str, Any]] = []
        for it in simple_candidates:
            if len(simple_out) >= remaining:
                break
            if it["id"] in used_ids or it["id"] in seen_simple:
                continue
            seen_simple.add(it["id"])
            simple_out.append(dict(it))

        tasks = complex_tasks + simple_out

    if args.initial_cfg_ratio and tasks:
        ratio = float(args.initial_cfg_ratio)
        if ratio < 0.0:
            ratio = 0.0
        if ratio > 1.0:
            ratio = 1.0
        inject_n = int(round(len(tasks) * ratio))
        idxs = list(range(len(tasks)))
        rng.shuffle(idxs)
        chosen = set(idxs[:inject_n])

        for j, t in enumerate(tasks):
            if j not in chosen:
                continue
            cfg0 = _make_initial_cfg_for_intent(t.get("intent", ""), rng)
            if not isinstance(cfg0, dict):
                continue
            t["initial_cfg"] = cfg0
            t["initial_prompt"] = {
                "guidance": "Start from the given candidate_cfg and apply minimal edits to satisfy all constraints.",
            }

    if len(tasks) > args.max:
        tasks = tasks[: args.max]

    Path(args.out).write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    print({"out": args.out, "tasks": len(tasks), "skipped": skipped})


if __name__ == "__main__":
    main()
