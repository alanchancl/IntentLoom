import os
import sys
import json
import argparse
import time
import re
from json import JSONDecodeError
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from intentloom import IntentLoomRunner, parse_intent
from intentloom.llm import MockLLM
from intentloom.compiler import compile_intent
from intentloom.adapters.k8s_llm import make_k8s_openai_llm
from intentloom.adapters.k8s_verifier import K8sPythonVerifier, K8sSymbolicAIContractVerifier
from intentloom.adapters.k8s_adapter import build_k8s_graph, find_path


@dataclass
class TaskResult:
    task_id: str
    ok: bool
    rounds: int
    llm_calls: int
    final_cfg: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "ok": self.ok,
            "rounds": self.rounds,
            "llm_calls": self.llm_calls,
            "final_cfg": self.final_cfg,
        }


def _few_shot_examples_k8s() -> List[Dict[str, Any]]:
    return [
        {
            "intent": "Reach pod:demo-client svc:demo-svc port=80 protocol=tcp",
            "before": {
                "k8s": {
                    "pods": [
                        {"name": "demo-client", "namespace": "demo", "labels": {"app": "demo-client"}},
                        {"name": "demo-backend", "namespace": "demo", "labels": {"app": "demo-backend"}},
                    ],
                    "services": [
                        {"name": "demo-svc", "namespace": "demo", "selector": {"app": "demo-backend"}},
                    ],
                },
                "policies": [],
            },
            "after": {
                "k8s": {
                    "pods": [
                        {"name": "demo-client", "namespace": "demo", "labels": {"app": "demo-client"}},
                        {"name": "demo-backend", "namespace": "demo", "labels": {"app": "demo-backend"}},
                    ],
                    "services": [
                        {"name": "demo-svc", "namespace": "demo", "selector": {"app": "demo-backend"}},
                    ],
                },
                "policies": [
                    {
                        "name": "allow-demo-client-to-demo-svc",
                        "action": "allow",
                        "from": {"namespace": "demo", "podSelector": {"app": "demo-client"}},
                        "to": {"namespace": "demo", "podSelector": {"app": "demo-backend"}},
                        "ports": [{"port": 80, "protocol": "TCP"}],
                    }
                ],
            },
        },
        {
            "intent": "Isolate ns:alpha ns:beta port=443 protocol=tcp",
            "before": {
                "k8s": {
                    "pods": [
                        {"name": "a", "namespace": "alpha", "labels": {"app": "a"}},
                        {"name": "b", "namespace": "beta", "labels": {"app": "b"}},
                    ],
                    "services": [],
                },
                "policies": [
                    {
                        "name": "allow-alpha-to-beta-any",
                        "action": "allow",
                        "from": {"namespace": "alpha", "podSelector": None},
                        "to": {"namespace": "beta", "podSelector": None},
                        "ports": None,
                    }
                ],
            },
            "after": {
                "k8s": {
                    "pods": [
                        {"name": "a", "namespace": "alpha", "labels": {"app": "a"}},
                        {"name": "b", "namespace": "beta", "labels": {"app": "b"}},
                    ],
                    "services": [],
                },
                "policies": [],
            },
        },
        {
            "intent": "Reach pod:p1 pod:p2 port=8080 protocol=tcp",
            "before": {
                "k8s": {
                    "pods": [
                        {"name": "p1", "namespace": "n1", "labels": {"app": "p1"}},
                        {"name": "p2", "namespace": "n2", "labels": {"app": "p2"}},
                    ],
                    "services": [],
                },
                "policies": [],
            },
            "after": {
                "k8s": {
                    "pods": [
                        {"name": "p1", "namespace": "n1", "labels": {"app": "p1"}},
                        {"name": "p2", "namespace": "n2", "labels": {"app": "p2"}},
                    ],
                    "services": [],
                },
                "policies": [
                    {
                        "name": "allow-p1-to-p2-8080",
                        "action": "allow",
                        "from": {"namespace": "n1", "podSelector": {"app": "p1"}},
                        "to": {"namespace": "n2", "podSelector": {"app": "p2"}},
                        "ports": [{"port": 8080, "protocol": "TCP"}],
                    }
                ],
            },
        },
    ]


def _load_task_result_from_jsonl(
    path: Path,
    *,
    task_id: str,
    force_llm_calls: Optional[int] = None,
) -> Optional[TaskResult]:
    if not path.exists() or not path.is_file():
        return None
    rounds = 0
    llm_calls = 0
    last_obj: Optional[Dict[str, Any]] = None
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                rounds += 1
                llm_calls += 1
                if isinstance(obj.get("reflection"), str) and str(obj.get("reflection")).strip():
                    llm_calls += 1
                last_obj = obj
    except OSError:
        return None

    if rounds <= 0 or last_obj is None:
        return None

    verification = last_obj.get("verification")
    ok = bool(verification.get("ok")) if isinstance(verification, dict) else False
    final_cfg = last_obj.get("candidate_cfg")
    if not isinstance(final_cfg, dict):
        final_cfg = {}
    if isinstance(force_llm_calls, int) and force_llm_calls >= 0:
        llm_calls = int(force_llm_calls)
    return TaskResult(task_id=task_id, ok=ok, rounds=rounds, llm_calls=llm_calls, final_cfg=final_cfg)


def _canon(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip().lower()


def _split_node(x: Any):
    s = _canon(x)
    if ":" not in s:
        return "", s
    t, v = s.split(":", 1)
    return t, v


def _find_pod(cfg: Dict[str, Any], name: str) -> Dict[str, Any]:
    k8s = cfg.get("k8s") if isinstance(cfg, dict) else None
    pods = k8s.get("pods") if isinstance(k8s, dict) else None
    if not isinstance(pods, list):
        return {}
    for p in pods:
        if not isinstance(p, dict):
            continue
        if _canon(p.get("name")) == _canon(name):
            return p
    return {}


def _find_svc(cfg: Dict[str, Any], name: str) -> Dict[str, Any]:
    k8s = cfg.get("k8s") if isinstance(cfg, dict) else None
    svcs = k8s.get("services") if isinstance(k8s, dict) else None
    if not isinstance(svcs, list):
        return {}
    for s in svcs:
        if not isinstance(s, dict):
            continue
        if _canon(s.get("name")) == _canon(name):
            return s
    return {}


def _policies_on_path(path: List[str], edge_reasons):
    if not path or len(path) < 2:
        return set()
    out = set()
    for i in range(len(path) - 1):
        a = _canon(path[i])
        b = _canon(path[i + 1])
        key = (a, b) if a < b else (b, a)
        rs = edge_reasons.get(key, []) if isinstance(edge_reasons, dict) else []
        for r in rs:
            if isinstance(r, dict) and r.get("type") == "policy" and r.get("name"):
                out.add(_canon(r.get("name")))
    return out


def _override_semantics(cfg: Dict[str, Any], semantics_override: str | None) -> Dict[str, Any]:
    if not isinstance(cfg, dict):
        return {}
    if semantics_override is None:
        return cfg
    s = _canon(semantics_override)
    if s not in {"option_a", "option_b", "option_c"}:
        return cfg
    out = dict(cfg)
    k8s = out.get("k8s")
    k8s2 = dict(k8s) if isinstance(k8s, dict) else {}
    k8s2["semantics"] = s
    out["k8s"] = k8s2
    return out


class K8sRuleBasedLLM:
    def __init__(self, initial_cfg: Dict[str, Any]):
        self._cfg = dict(initial_cfg or {})

    def generate(self, prompt: Dict[str, Any]) -> Dict[str, Any]:
        base = prompt.get("candidate_cfg")
        if isinstance(base, dict):
            cfg = dict(base)
        else:
            cfg = dict(self._cfg)

        if "k8s" not in cfg:
            cfg["k8s"] = {}
        if "policies" not in cfg or not isinstance(cfg.get("policies"), list):
            cfg["policies"] = []

        violations = (prompt.get("verification") or {}).get("violations") or []

        for v in violations:
            if not isinstance(v, dict):
                continue
            if v.get("code") != "ISOLATE_VIOLATION":
                continue
            meta = v.get("meta") or {}
            a = meta.get("a")
            b = meta.get("b")
            port = meta.get("port")
            protocol = meta.get("protocol")
            port_i = int(port) if isinstance(port, int) or (isinstance(port, str) and str(port).strip().isdigit()) else None
            proto_s = str(protocol).strip() if isinstance(protocol, str) and str(protocol).strip() else None
            g = build_k8s_graph(cfg)
            p = find_path(
                g.links,
                str(a or ""),
                str(b or ""),
                edge_reasons=g.edge_reasons,
                port=port_i,
                protocol=proto_s,
            )
            pols = _policies_on_path(p or [], g.edge_reasons)
            if pols:
                cfg["policies"] = [
                    pol
                    for pol in cfg.get("policies", [])
                    if not (
                        isinstance(pol, dict)
                        and _canon(pol.get("name"))
                        and _canon(pol.get("name")) in pols
                    )
                ]

        for v in violations:
            if not isinstance(v, dict):
                continue
            if v.get("code") != "REACH_VIOLATION":
                continue
            meta = v.get("meta") or {}
            a = meta.get("a")
            b = meta.get("b")
            port = meta.get("port")
            protocol = meta.get("protocol")
            port_i = int(port) if isinstance(port, int) or (isinstance(port, str) and str(port).strip().isdigit()) else None
            proto_s = str(protocol).strip().upper() if isinstance(protocol, str) and str(protocol).strip() else None
            a_t, a_v = _split_node(a)
            b_t, b_v = _split_node(b)
            if a_t != "pod":
                continue

            a_pod = _find_pod(cfg, a_v)
            if not a_pod:
                continue

            src_ns = a_pod.get("namespace")
            src_sel = a_pod.get("labels") if isinstance(a_pod.get("labels"), dict) else {}

            dst_ns = None
            dst_sel = None
            if b_t == "pod":
                b_pod = _find_pod(cfg, b_v)
                if b_pod:
                    dst_ns = b_pod.get("namespace")
                    dst_sel = b_pod.get("labels") if isinstance(b_pod.get("labels"), dict) else {}
            elif b_t == "svc":
                svc = _find_svc(cfg, b_v)
                if svc:
                    dst_ns = svc.get("namespace")
                    dst_sel = svc.get("selector") if isinstance(svc.get("selector"), dict) else {}
            elif b_t == "ns":
                dst_ns = b_v
                dst_sel = None

            if dst_ns is None:
                continue

            pol_name = f"auto-allow-{_canon(a_v)}-to-{_canon(b_t)}-{_canon(b_v)}"
            new_pol = {
                "name": pol_name,
                "action": "allow",
                "from": {"namespace": str(src_ns) if src_ns is not None else None, "podSelector": src_sel or None},
                "to": {"namespace": str(dst_ns) if dst_ns is not None else None, "podSelector": dst_sel or None},
            }
            if port_i is not None:
                rec = {"port": int(port_i)}
                if proto_s:
                    rec["protocol"] = proto_s
                new_pol["ports"] = [rec]
            existing = [p for p in cfg.get("policies", []) if isinstance(p, dict) and _canon(p.get("name")) == _canon(pol_name)]
            if not existing:
                cfg["policies"].append(new_pol)

        self._cfg = cfg
        return cfg


class K8sDeleteBlameLLM:
    def __init__(self, initial_cfg: Dict[str, Any]):
        self._cfg = dict(initial_cfg or {})

    def generate(self, prompt: Dict[str, Any]) -> Dict[str, Any]:
        base = prompt.get("candidate_cfg")
        if isinstance(base, dict):
            cfg = dict(base)
        else:
            cfg = dict(self._cfg)

        if "k8s" not in cfg:
            cfg["k8s"] = {}
        if "policies" not in cfg or not isinstance(cfg.get("policies"), list):
            cfg["policies"] = []

        verification = prompt.get("verification") if isinstance(prompt, dict) else None
        violations = (verification or {}).get("violations") if isinstance(verification, dict) else None
        if not isinstance(violations, list):
            violations = []

        has_isolate = any(isinstance(v, dict) and v.get("code") == "ISOLATE_VIOLATION" for v in violations)
        blame_set = prompt.get("blame_set")
        blamed_policies = blame_set.get("blamed_policies") if isinstance(blame_set, dict) else None
        if has_isolate and isinstance(blamed_policies, list) and blamed_policies:
            banned = {_canon(x) for x in blamed_policies if isinstance(x, str) and _canon(x)}
            if banned:
                cfg["policies"] = [
                    p
                    for p in cfg.get("policies", [])
                    if not (isinstance(p, dict) and _canon(p.get("name")) in banned)
                ]

        self._cfg = cfg
        return cfg


def _make_llm(spec: Dict[str, Any]):
    return _make_llm_seeded(spec, seed=None)


def _make_llm_seeded(spec: Dict[str, Any], *, seed: int | None = None):
    t = (spec.get("type") or "").lower()
    if t == "rule_based":
        return K8sRuleBasedLLM(spec.get("initial_cfg") or {})
    if t == "delete_blame":
        return K8sDeleteBlameLLM(spec.get("initial_cfg") or {})
    if t == "mock":
        return MockLLM(spec.get("candidates") or [])
    if t == "openai":
        model = spec.get("model") or "gpt-4o-mini"
        base_url = spec.get("base_url")
        timeout = spec.get("timeout")
        default_headers = spec.get("default_headers")
        temperature = spec.get("temperature")
        return make_k8s_openai_llm(
            model=model,
            base_url=base_url,
            timeout=timeout,
            default_headers=default_headers,
            seed=seed,
            temperature=temperature,
        )
    raise ValueError(f"Unknown llm type: {t}")


def _parse_seeds(seed: int | None, seeds: str | None) -> List[int | None]:
    if isinstance(seeds, str) and seeds.strip():
        out: List[int | None] = []
        for part in seeds.split(","):
            p = part.strip()
            if not p:
                continue
            out.append(int(p))
        return out
    if seed is None:
        return [None]
    return [int(seed)]


def _read_task_ids_file(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    ids: List[str] = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        s = str(raw).strip()
        if not s or s.startswith("#"):
            continue
        ids.append(s)
    return ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=str, default=".\\data\\k8s_policy_5_tasks.json")
    ap.add_argument("--out", type=str, default=".\\artifacts\\experiment_k8s")
    ap.add_argument("--verifier", type=str, default="python", choices=["python", "symbolicai"])
    ap.add_argument(
        "--semantics_override",
        type=str,
        default=None,
        choices=["option_a", "option_b", "option_c"],
        help="Override cfg.k8s.semantics for all tasks (useful to compare Option A vs Option B on the same ports task set).",
    )
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds; if set, runs multiple times and writes to out/seed_<seed>/.",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="If set, skips tasks that already have out_dir/<task_id>.jsonl and continues the remaining tasks.",
    )
    ap.add_argument(
        "--task_retries",
        type=int,
        default=3,
        help="Number of retries per task if the LLM call fails (e.g., timeout). After retries are exhausted, aborts the seed (no summary.json).",
    )
    ap.add_argument(
        "--task_retry_backoff_s",
        type=float,
        default=2.0,
        help="Base backoff seconds between per-task retries (exponential: base * 2^(attempt-1)).",
    )
    ap.add_argument(
        "--skip_completed_seeds",
        action="store_true",
        help="If set, skips any seed_<seed> that already contains summary.json.",
    )
    ap.add_argument(
        "--ablation_mode",
        type=str,
        default="raw_witness",
        choices=["generic", "standard_feedback", "raw_witness", "blame_only"],
    )
    ap.add_argument(
        "--method",
        type=str,
        default=None,
        choices=[
            "m0_deterministic_delete",
            "m1_vanilla",
            "m2_fewshot",
            "m3_standard_feedback",
            "m4_reflexion",
            "m5_intentloom",
        ],
        help="Convenience shortcut that configures open-loop/closed-loop and evidence style for M0-M5 baselines.",
    )
    ap.add_argument(
        "--reflexion",
        action="store_true",
        help="If set, performs an extra reflection step before each candidate generation.",
    )
    ap.add_argument(
        "--no_initial_verification_in_prompt",
        action="store_true",
        help="If set, does not include the initial verifier feedback in the first prompt (useful for open-loop one-shot baselines).",
    )
    ap.add_argument(
        "--max_rounds_override",
        type=int,
        default=None,
        help="If set, overrides task max_rounds for all tasks.",
    )
    ap.add_argument(
        "--only_verify_initial",
        action="store_true",
        help="Only verify each task's initial_cfg (no LLM calls). Writes verification JSON artifacts.",
    )
    ap.add_argument(
        "--llm_override",
        type=str,
        default=None,
        choices=["openai", "mock", "rule_based", "delete_blame"],
        help="Override llm.type for all tasks (useful for debugging).",
    )
    ap.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override llm.model for all tasks (e.g., openai/gpt-4o-mini for OpenRouter).",
    )
    ap.add_argument(
        "--base_url",
        type=str,
        default=None,
        help="Override llm.base_url for all tasks (e.g., https://openrouter.ai/api/v1).",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Override llm.timeout (seconds) for all tasks.",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override llm.temperature for all tasks (sampling; default is 0.0).",
    )
    ap.add_argument(
        "--max_tasks",
        type=int,
        default=None,
        help="If set, only run the first N tasks.",
    )
    ap.add_argument(
        "--task_mode",
        type=str,
        default=None,
        help="If set, only run tasks whose meta.mode equals this value (e.g., over_permissive).",
    )
    ap.add_argument(
        "--task_modes",
        type=str,
        default=None,
        help="Comma-separated meta.mode values to run (e.g., over_permissive,selector_typo).",
    )
    ap.add_argument(
        "--task_ids",
        type=str,
        default=None,
        help="Comma-separated task IDs to run; if unset, runs all tasks.",
    )
    ap.add_argument(
        "--task_ids_file",
        type=str,
        default=None,
        help="Path to a text file of task IDs to run (one per line; blank lines and # comments ignored).",
    )
    args = ap.parse_args()

    tasks_path = Path(args.tasks)

    tasks: List[Dict[str, Any]] = json.loads(tasks_path.read_text(encoding="utf-8"))

    if args.verifier == "symbolicai":
        verifier = K8sSymbolicAIContractVerifier()
    else:
        verifier = K8sPythonVerifier()
    runner = IntentLoomRunner(verifier=verifier)

    wanted: set[str] = set()
    if isinstance(args.task_ids, str) and args.task_ids.strip():
        wanted |= {s.strip() for s in str(args.task_ids).split(",") if s.strip()}
    if isinstance(args.task_ids_file, str) and args.task_ids_file.strip():
        wanted |= {s for s in _read_task_ids_file(args.task_ids_file) if s}
    if wanted:
        tasks = [t for t in tasks if isinstance(t, dict) and (t.get("id") in wanted)]

    mode_wanted = None
    if isinstance(args.task_modes, str) and args.task_modes.strip():
        mode_wanted = {s.strip() for s in str(args.task_modes).split(",") if s.strip()}
    elif isinstance(args.task_mode, str) and args.task_mode.strip():
        mode_wanted = {str(args.task_mode).strip()}
    if mode_wanted:
        tasks = [
            t
            for t in tasks
            if isinstance(t, dict)
            and isinstance(t.get("meta"), dict)
            and (t.get("meta") or {}).get("mode") in mode_wanted
        ]

    if isinstance(args.max_tasks, int) and args.max_tasks > 0:
        tasks = tasks[: args.max_tasks]

    seed_list = _parse_seeds(args.seed, args.seeds)
    base_out = Path(args.out)

    for sd in seed_list:
        out_dir = base_out if sd is None else (base_out / f"seed_{sd}")
        out_dir.mkdir(parents=True, exist_ok=True)

        if args.skip_completed_seeds and (out_dir / "summary.json").exists():
            print(f"[skip] seed={sd} already has summary.json")
            continue

        try:
            results: List[TaskResult] = []
            ok_cnt = 0
            rounds_sum = 0
            llm_calls_sum = 0

            for t in tasks:
                task_id = t.get("id") or ""
                log_path = out_dir / f"{task_id}.jsonl"
                if args.resume and (not args.only_verify_initial) and log_path.exists():
                    force_calls: Optional[int] = None
                    if args.method and str(args.method).strip().lower() == "m0_deterministic_delete":
                        force_calls = 0
                    if args.llm_override and str(args.llm_override).strip().lower() == "delete_blame":
                        force_calls = 0
                    existing = _load_task_result_from_jsonl(
                        log_path,
                        task_id=task_id,
                        force_llm_calls=force_calls,
                    )
                    if existing is not None:
                        results.append(existing)
                        if existing.ok:
                            ok_cnt += 1
                        rounds_sum += existing.rounds
                        llm_calls_sum += existing.llm_calls
                        continue

                intent_text = t.get("intent") or ""
                intent = parse_intent(intent_text)
                llm_spec = dict(t.get("llm") or {})
                if args.llm_override:
                    llm_spec["type"] = args.llm_override
                if args.model:
                    llm_spec["model"] = args.model
                if args.base_url is not None:
                    llm_spec["base_url"] = str(args.base_url)
                if args.timeout is not None:
                    llm_spec["timeout"] = args.timeout
                if args.temperature is not None:
                    llm_spec["temperature"] = args.temperature
                max_rounds_task = int(t.get("max_rounds") or 3)
                if args.max_rounds_override is not None and int(args.max_rounds_override) > 0:
                    max_rounds_task = int(args.max_rounds_override)

                initial_prompt = dict(t.get("initial_prompt") or {})
                initial_cfg = t.get("initial_cfg")
                cfg0 = _override_semantics(initial_cfg if isinstance(initial_cfg, dict) else {}, args.semantics_override)

                ablation_mode = args.ablation_mode
                include_initial_verification_in_prompt = not bool(args.no_initial_verification_in_prompt)
                reflexion = bool(args.reflexion)
                max_rounds = max_rounds_task

                if args.method:
                    m = str(args.method).strip().lower()
                    if m == "m0_deterministic_delete":
                        llm_spec["type"] = "delete_blame"
                        include_initial_verification_in_prompt = True
                        reflexion = False
                        ablation_mode = "blame_only"
                    elif m == "m1_vanilla":
                        include_initial_verification_in_prompt = False
                        reflexion = False
                        ablation_mode = "standard_feedback"
                        max_rounds = 1
                    elif m == "m2_fewshot":
                        include_initial_verification_in_prompt = False
                        reflexion = False
                        ablation_mode = "standard_feedback"
                        max_rounds = 1
                        initial_prompt["few_shot_examples"] = _few_shot_examples_k8s()
                    elif m == "m3_standard_feedback":
                        include_initial_verification_in_prompt = True
                        reflexion = False
                        ablation_mode = "standard_feedback"
                    elif m == "m4_reflexion":
                        include_initial_verification_in_prompt = True
                        reflexion = True
                        ablation_mode = "standard_feedback"
                    elif m == "m5_intentloom":
                        include_initial_verification_in_prompt = True
                        reflexion = False
                        ablation_mode = "blame_only"

                err_path = out_dir / f"{task_id}__error.json"
                if args.only_verify_initial:
                    compiled = compile_intent(intent)
                    vr0 = verifier.verify(cfg0, compiled.constraints)
                    out_obj = {
                        "seed": sd,
                        "task_id": task_id,
                        "intent": intent_text,
                        "ok": vr0.ok,
                        "initial_cfg": cfg0,
                        "verification": vr0.to_dict(),
                    }
                    (out_dir / f"{task_id}__initial_verify.json").write_text(
                        json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                    results.append(
                        TaskResult(task_id=task_id, ok=vr0.ok, rounds=0, llm_calls=0, final_cfg=cfg0)
                    )
                    if vr0.ok:
                        ok_cnt += 1
                    continue

                llm_type = (llm_spec.get("type") or "").lower()
                if llm_type in {"rule_based", "delete_blame"}:
                    llm_spec["initial_cfg"] = cfg0
                llm = _make_llm_seeded(llm_spec, seed=sd)

                loop_result = None
                last_exc: Exception | None = None
                max_attempts = max(1, int(args.task_retries) + 1)
                for attempt in range(1, max_attempts + 1):
                    try:
                        loop_result = runner.run(
                            intent=intent,
                            llm=llm,
                            initial_prompt=initial_prompt,
                            max_rounds=max_rounds,
                            initial_cfg=cfg0,
                            ablation_mode=ablation_mode,
                            include_initial_verification_in_prompt=include_initial_verification_in_prompt,
                            reflexion=reflexion,
                        )
                        last_exc = None
                        break
                    except Exception as e:
                        last_exc = e
                        if attempt >= max_attempts:
                            break
                        backoff = float(args.task_retry_backoff_s) * (2 ** (attempt - 1))
                        time.sleep(max(0.0, backoff))

                if loop_result is None:
                    e = last_exc if last_exc is not None else RuntimeError("Task failed")
                    err_obj = {
                        "seed": sd,
                        "task_id": task_id,
                        "intent": intent_text,
                        "error_type": type(e).__name__,
                        "error": str(e),
                    }
                    err_path.write_text(
                        json.dumps(err_obj, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                    raise e

                results.append(
                    TaskResult(
                        task_id=task_id,
                        ok=loop_result.ok,
                        rounds=len(loop_result.steps),
                        llm_calls=(
                            0
                            if (llm_spec.get("type") or "").lower() == "delete_blame"
                            else (
                                len(loop_result.steps)
                                + sum(
                                    1
                                    for s in loop_result.steps
                                    if isinstance(getattr(s, "reflection", None), str) and getattr(s, "reflection")
                                )
                            )
                        ),
                        final_cfg=loop_result.final_cfg,
                    )
                )

                loop_result.write_jsonl(log_path)

                if err_path.exists():
                    try:
                        err_path.unlink()
                    except OSError:
                        pass

                if loop_result.ok:
                    ok_cnt += 1
                rounds_sum += len(loop_result.steps)
                if (llm_spec.get("type") or "").lower() == "delete_blame":
                    llm_calls_sum += 0
                else:
                    llm_calls_sum += len(loop_result.steps) + sum(
                        1
                        for s in loop_result.steps
                        if isinstance(getattr(s, "reflection", None), str) and getattr(s, "reflection")
                    )

            summary = {
                "seed": sd,
                "tasks": len(results),
                "ok": ok_cnt,
                "success_rate": (ok_cnt / len(results)) if results else 0.0,
                "avg_rounds": (rounds_sum / len(results)) if results else 0.0,
                "avg_llm_calls": (llm_calls_sum / len(results)) if results else 0.0,
                "results": [r.to_dict() for r in results],
            }

            (out_dir / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(summary)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(
                f"[error] seed={sd} aborted (no summary.json written): {type(e).__name__}: {e}",
                file=sys.stderr,
            )
            continue


if __name__ == "__main__":
    main()
