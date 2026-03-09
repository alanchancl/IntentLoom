import os
import sys
import json
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from intentloom import IntentLoomRunner, parse_intent
from intentloom.llm import MockLLM, RuleBasedLLM
from intentloom.openai_llm import OpenAILLM


def _parse_intent_with_fallback(text: str, parser: str, *, seed: int | None = None):
    try:
        return parse_intent(text)
    except Exception:
        if parser != "openai":
            raise
        from intentloom.openai_intent_parser import OpenAIIntentParser

        return OpenAIIntentParser(seed=seed).parse(text)


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


def _make_llm(spec: Dict[str, Any], *, seed: int | None = None):
    t = (spec.get("type") or "").lower()
    if t == "rule_based":
        return RuleBasedLLM(spec.get("initial_cfg") or {})
    if t == "mock":
        return MockLLM(spec.get("candidates") or [])
    if t == "openai":
        model = spec.get("model") or "gpt-4o-mini"
        base_url = spec.get("base_url")
        timeout = spec.get("timeout")
        default_headers = spec.get("default_headers")
        return OpenAILLM(
            model=model,
            base_url=base_url,
            timeout=timeout,
            default_headers=default_headers,
            seed=seed,
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=str, default=".\\data\\sample_tasks.json")
    ap.add_argument("--out", type=str, default=".\\artifacts\\experiment")
    ap.add_argument("--verifier", type=str, default="python", choices=["python", "symbolicai"])
    ap.add_argument("--intent_parser", type=str, default="rule", choices=["rule", "openai"])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds; if set, runs multiple times and writes to out/seed_<seed>/.",
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
            "m3_standard_feedback",
            "m4_reflexion",
            "m5_intentloom",
        ],
        help="Convenience shortcut that configures evidence style for M3-M5 baselines.",
    )
    ap.add_argument(
        "--reflexion",
        action="store_true",
        help="If set, performs an extra reflection step before each candidate generation.",
    )
    ap.add_argument(
        "--no_initial_verification_in_prompt",
        action="store_true",
        help="If set, does not include the initial verifier feedback in the first prompt.",
    )
    ap.add_argument(
        "--max_rounds_override",
        type=int,
        default=None,
        help="If set, overrides task max_rounds for all tasks.",
    )
    ap.add_argument(
        "--llm_override",
        type=str,
        default=None,
        choices=["openai", "mock", "rule_based"],
        help="Override llm.type for all tasks (useful for running baselines).",
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
        help="Override llm.base_url for all tasks.",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Override llm.timeout (seconds) for all tasks.",
    )
    ap.add_argument(
        "--max_tasks",
        type=int,
        default=None,
        help="If set, only run the first N tasks.",
    )
    args = ap.parse_args()

    tasks_path = Path(args.tasks)

    tasks: List[Dict[str, Any]] = json.loads(tasks_path.read_text(encoding="utf-8"))

    if isinstance(args.max_tasks, int) and args.max_tasks > 0:
        tasks = tasks[: args.max_tasks]

    if args.verifier == "symbolicai":
        from intentloom.symbolicai_verifier import SymbolicAIContractVerifier

        runner = IntentLoomRunner(verifier=SymbolicAIContractVerifier())
    else:
        runner = IntentLoomRunner()

    seed_list = _parse_seeds(args.seed, args.seeds)
    base_out = Path(args.out)

    for sd in seed_list:
        out_dir = base_out if sd is None else (base_out / f"seed_{sd}")
        out_dir.mkdir(parents=True, exist_ok=True)
        results: List[TaskResult] = []
        ok_cnt = 0
        rounds_sum = 0
        llm_calls_sum = 0

        for t in tasks:
            task_id = t.get("id") or ""
            intent_text = t.get("intent") or ""
            intent = _parse_intent_with_fallback(intent_text, parser=args.intent_parser, seed=sd)
            llm_spec = dict(t.get("llm") or {})
            if args.llm_override:
                llm_spec["type"] = args.llm_override
            if args.model:
                llm_spec["model"] = args.model
            if args.base_url:
                llm_spec["base_url"] = args.base_url
            if args.timeout is not None:
                llm_spec["timeout"] = args.timeout
            llm = _make_llm(llm_spec, seed=sd)
            max_rounds_task = int(t.get("max_rounds") or 3)
            if args.max_rounds_override is not None and int(args.max_rounds_override) > 0:
                max_rounds_task = int(args.max_rounds_override)

            ablation_mode = args.ablation_mode
            include_initial_verification_in_prompt = not bool(args.no_initial_verification_in_prompt)
            reflexion = bool(args.reflexion)
            max_rounds = max_rounds_task

            if args.method:
                m = str(args.method).strip().lower()
                if m == "m3_standard_feedback":
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

            initial_prompt = t.get("initial_prompt") or {}
            initial_cfg = t.get("initial_cfg")
            if not isinstance(initial_cfg, dict):
                initial_cfg = llm_spec.get("initial_cfg")
            if not isinstance(initial_cfg, dict):
                initial_cfg = {"links": [], "acl": []}
            loop_result = runner.run(
                intent=intent,
                llm=llm,
                initial_prompt=initial_prompt,
                max_rounds=max_rounds,
                initial_cfg=initial_cfg,
                ablation_mode=ablation_mode,
                include_initial_verification_in_prompt=include_initial_verification_in_prompt,
                reflexion=reflexion,
            )
            results.append(
                TaskResult(
                    task_id=task_id,
                    ok=loop_result.ok,
                    rounds=len(loop_result.steps),
                    llm_calls=len(loop_result.steps)
                    + sum(1 for s in loop_result.steps if isinstance(getattr(s, "reflection", None), str) and getattr(s, "reflection")),
                    final_cfg=loop_result.final_cfg,
                )
            )

            log_path = out_dir / f"{task_id}.jsonl"
            loop_result.write_jsonl(log_path)

            if loop_result.ok:
                ok_cnt += 1
            rounds_sum += len(loop_result.steps)
            llm_calls_sum += len(loop_result.steps) + sum(
                1 for s in loop_result.steps if isinstance(getattr(s, "reflection", None), str) and getattr(s, "reflection")
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


if __name__ == "__main__":
    main()
