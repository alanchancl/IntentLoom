import os
import sys
import json
import csv
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from intentloom import parse_intent
from intentloom.compiler import compile_intent
from intentloom.adapters.k8s_verifier import K8sPythonVerifier, K8sSymbolicAIContractVerifier


def _canon(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip().lower()


def _template_from_task_id(task_id: str) -> str:
    s = str(task_id)
    parts = s.split("_", 2)
    if len(parts) >= 3:
        return parts[2]
    return ""


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def _get_candidate_cfg(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    cand = obj.get("candidate_cfg")
    if isinstance(cand, dict):
        return cand
    prompt = obj.get("prompt")
    if isinstance(prompt, dict) and isinstance(prompt.get("candidate_cfg"), dict):
        return prompt.get("candidate_cfg")
    return None


def _get_old_ok_and_codes(obj: Dict[str, Any]) -> Tuple[Optional[bool], List[str]]:
    v = obj.get("verification")
    if not isinstance(v, dict):
        return None, []
    ok = v.get("ok")
    ok_b = bool(ok) if isinstance(ok, bool) else None
    viols = v.get("violations")
    codes: List[str] = []
    if isinstance(viols, list):
        for it in viols:
            if isinstance(it, dict) and it.get("code"):
                codes.append(str(it.get("code")))
    return ok_b, codes


@dataclass
class ReplayRow:
    task_id: str
    template: str
    seed: int
    old_rounds: int
    old_first_ok_round: Optional[int]
    old_final_ok: Optional[bool]
    old_final_violation_codes: str
    replay_first_ok_round: Optional[int]
    replay_final_ok: bool
    replay_final_violation_codes: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "template": self.template,
            "seed": self.seed,
            "old_rounds": self.old_rounds,
            "old_first_ok_round": self.old_first_ok_round,
            "old_final_ok": self.old_final_ok,
            "old_final_violation_codes": self.old_final_violation_codes,
            "replay_first_ok_round": self.replay_first_ok_round,
            "replay_final_ok": self.replay_final_ok,
            "replay_final_violation_codes": self.replay_final_violation_codes,
            "rounds_saved_if_early_stop": (
                (self.old_rounds - self.replay_first_ok_round)
                if isinstance(self.replay_first_ok_round, int) and self.replay_first_ok_round > 0
                else None
            ),
        }


def _write_csv(path: Path, rows: List[Dict[str, Any]], *, fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            if not isinstance(r, dict):
                continue
            w.writerow({k: r.get(k) for k in fieldnames})


def _count_codes(rows: List[ReplayRow], *, which: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in rows:
        s = ""
        if which == "old":
            s = str(r.old_final_violation_codes or "")
        elif which == "replay":
            s = str(r.replay_final_violation_codes or "")
        parts = [p.strip() for p in s.split(",") if p.strip()]
        if not parts:
            continue
        for p in parts:
            out[p] = int(out.get(p, 0)) + 1
    return dict(sorted(out.items(), key=lambda kv: (-int(kv[1]), str(kv[0]))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--verifier",
        type=str,
        default="python",
        choices=["python", "symbolicai"],
        help="Verifier backend used for replay verification (no LLM calls).",
    )
    ap.add_argument(
        "--artifact_root",
        type=str,
        default="artifacts/ports100_option_c_generic",
        help="Root directory containing seed_<seed>/<task_id>.jsonl artifacts.",
    )
    ap.add_argument(
        "--tasks",
        type=str,
        default="data/k8s_policy_100_ports_tasks.json",
        help="Task list JSON used to map task_id -> intent text.",
    )
    ap.add_argument(
        "--seeds",
        type=str,
        default="1,2,3",
        help="Comma-separated seeds to replay (matches seed_<seed> directories).",
    )
    ap.add_argument(
        "--task_ids",
        type=str,
        default=None,
        help="Comma-separated task IDs to replay; if unset, replays all tasks in tasks file.",
    )
    ap.add_argument(
        "--template",
        type=str,
        default=None,
        help="If set, only replay tasks whose template equals this string (e.g., reach_pod_missing_policy).",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory for replay_summary.{json,csv}; default is <artifact_root>/replay_verify.",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    artifact_root = (repo_root / str(args.artifact_root)).resolve()
    tasks_path = (repo_root / str(args.tasks)).resolve()

    out_dir = (Path(args.out).resolve() if args.out else (artifact_root / "replay_verify")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks_list = json.loads(tasks_path.read_text(encoding="utf-8"))
    tasks: List[Dict[str, Any]] = [t for t in tasks_list if isinstance(t, dict)] if isinstance(tasks_list, list) else []
    intent_by_id: Dict[str, str] = {str(t.get("id")): str(t.get("intent")) for t in tasks if t.get("id") and t.get("intent")}

    all_task_ids = list(intent_by_id.keys())

    if isinstance(args.task_ids, str) and args.task_ids.strip():
        wanted = [s.strip() for s in str(args.task_ids).split(",") if s.strip()]
        all_task_ids = [tid for tid in all_task_ids if tid in set(wanted)]

    if isinstance(args.template, str) and args.template.strip():
        tmpl = str(args.template).strip()
        all_task_ids = [tid for tid in all_task_ids if _template_from_task_id(tid) == tmpl]

    seed_list = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]

    if args.verifier == "symbolicai":
        verifier = K8sSymbolicAIContractVerifier()
    else:
        verifier = K8sPythonVerifier()

    rows: List[ReplayRow] = []
    missing_jsonl: List[Tuple[int, str]] = []

    for sd in seed_list:
        seed_dir = artifact_root / f"seed_{sd}"
        for task_id in all_task_ids:
            jsonl_path = seed_dir / f"{task_id}.jsonl"
            if not jsonl_path.exists():
                missing_jsonl.append((sd, task_id))
                continue

            intent_text = intent_by_id.get(task_id)
            if not intent_text:
                continue
            compiled = compile_intent(parse_intent(intent_text))

            old_rounds = 0
            old_first_ok_round: Optional[int] = None
            old_final_ok: Optional[bool] = None
            old_final_codes: List[str] = []

            replay_first_ok_round: Optional[int] = None
            replay_final_ok = False
            replay_final_codes: List[str] = []

            last_candidate_cfg: Optional[Dict[str, Any]] = None

            for obj in _iter_jsonl(jsonl_path):
                r = obj.get("round")
                try:
                    rnd = int(r)
                except Exception:
                    rnd = old_rounds + 1
                old_rounds = max(old_rounds, rnd)

                old_ok, old_codes = _get_old_ok_and_codes(obj)
                if old_ok is True and old_first_ok_round is None:
                    old_first_ok_round = rnd
                old_final_ok = old_ok
                old_final_codes = old_codes

                cand = _get_candidate_cfg(obj)
                if isinstance(cand, dict):
                    last_candidate_cfg = cand
                    vr = verifier.verify(cand, compiled.constraints)
                    if vr.ok and replay_first_ok_round is None:
                        replay_first_ok_round = rnd
                    replay_final_ok = vr.ok
                    replay_final_codes = [v.code for v in vr.violations]

            if last_candidate_cfg is None:
                continue

            rows.append(
                ReplayRow(
                    task_id=task_id,
                    template=_template_from_task_id(task_id),
                    seed=sd,
                    old_rounds=old_rounds,
                    old_first_ok_round=old_first_ok_round,
                    old_final_ok=old_final_ok,
                    old_final_violation_codes=",".join(old_final_codes),
                    replay_first_ok_round=replay_first_ok_round,
                    replay_final_ok=bool(replay_final_ok),
                    replay_final_violation_codes=",".join(replay_final_codes),
                )
            )

    row_dicts = [r.to_dict() for r in rows]

    counterfactual_ok = [r for r in rows if isinstance(r.replay_first_ok_round, int)]
    old_ok = sum(1 for r in rows if r.old_final_ok is True)
    old_fail = sum(1 for r in rows if r.old_final_ok is False)
    old_unknown = len(rows) - old_ok - old_fail

    fail_to_ok = sum(
        1
        for r in rows
        if r.old_final_ok is False and isinstance(r.replay_first_ok_round, int)
    )
    ok_to_no_ok = sum(
        1
        for r in rows
        if r.old_final_ok is True and not isinstance(r.replay_first_ok_round, int)
    )
    earlier_ok = sum(
        1
        for r in rows
        if isinstance(r.old_first_ok_round, int)
        and isinstance(r.replay_first_ok_round, int)
        and r.replay_first_ok_round < r.old_first_ok_round
    )

    per_template: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        t = str(r.template or "")
        st = per_template.get(t)
        if st is None:
            st = {
                "template": t,
                "n": 0,
                "old_ok": 0,
                "old_fail": 0,
                "old_unknown": 0,
                "replay_any_ok": 0,
                "fail_to_ok": 0,
                "ok_to_no_ok": 0,
                "earlier_ok": 0,
                "rounds_saved_sum": 0,
                "rounds_saved_n": 0,
            }
            per_template[t] = st

        st["n"] += 1
        if r.old_final_ok is True:
            st["old_ok"] += 1
        elif r.old_final_ok is False:
            st["old_fail"] += 1
        else:
            st["old_unknown"] += 1

        if isinstance(r.replay_first_ok_round, int):
            st["replay_any_ok"] += 1

        if r.old_final_ok is False and isinstance(r.replay_first_ok_round, int):
            st["fail_to_ok"] += 1
        if r.old_final_ok is True and not isinstance(r.replay_first_ok_round, int):
            st["ok_to_no_ok"] += 1
        if (
            isinstance(r.old_first_ok_round, int)
            and isinstance(r.replay_first_ok_round, int)
            and r.replay_first_ok_round < r.old_first_ok_round
        ):
            st["earlier_ok"] += 1

        if isinstance(r.replay_first_ok_round, int) and r.replay_first_ok_round > 0:
            st["rounds_saved_sum"] += int(r.old_rounds - r.replay_first_ok_round)
            st["rounds_saved_n"] += 1

    per_template_rows: List[Dict[str, Any]] = []
    for st in per_template.values():
        n = int(st.get("n") or 0)
        ok_old = int(st.get("old_ok") or 0)
        ok_cf = int(st.get("replay_any_ok") or 0)
        saved_n = int(st.get("rounds_saved_n") or 0)
        saved_sum = int(st.get("rounds_saved_sum") or 0)
        per_template_rows.append(
            {
                **{k: st.get(k) for k in st.keys() if k not in {"rounds_saved_sum", "rounds_saved_n"}},
                "old_success_rate": (ok_old / n) if n else 0.0,
                "counterfactual_success_rate": (ok_cf / n) if n else 0.0,
                "avg_rounds_saved_if_early_stop": (saved_sum / saved_n) if saved_n else 0.0,
            }
        )

    per_template_rows.sort(key=lambda x: str(x.get("template") or ""))
    rounds_saved = [
        (r.old_rounds - r.replay_first_ok_round)
        for r in rows
        if isinstance(r.replay_first_ok_round, int) and r.replay_first_ok_round > 0
    ]

    summary = {
        "artifact_root": str(artifact_root),
        "tasks_path": str(tasks_path),
        "seeds": seed_list,
        "n_rows": len(rows),
        "n_missing_jsonl": len(missing_jsonl),
        "old_final_ok_true": old_ok,
        "old_final_ok_false": old_fail,
        "old_final_ok_none": old_unknown,
        "fail_to_ok": fail_to_ok,
        "ok_to_no_ok": ok_to_no_ok,
        "earlier_ok": earlier_ok,
        "counterfactual_success_rate": (len(counterfactual_ok) / len(rows)) if rows else 0.0,
        "avg_rounds_saved_if_early_stop": (sum(rounds_saved) / len(rounds_saved)) if rounds_saved else 0.0,
        "old_final_violation_code_counts": _count_codes(rows, which="old"),
        "replay_final_violation_code_counts": _count_codes(rows, which="replay"),
        "missing_jsonl_examples": [{"seed": sd, "task_id": tid} for sd, tid in missing_jsonl[:20]],
        "per_template": per_template_rows,
        "rows": row_dicts,
    }

    (out_dir / "replay_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = list(row_dicts[0].keys()) if row_dicts else [
        "task_id",
        "template",
        "seed",
        "old_rounds",
        "old_first_ok_round",
        "old_final_ok",
        "old_final_violation_codes",
        "replay_first_ok_round",
        "replay_final_ok",
        "replay_final_violation_codes",
        "rounds_saved_if_early_stop",
    ]
    _write_csv(out_dir / "replay_summary.csv", row_dicts, fieldnames=fieldnames)

    print(json.dumps({k: summary[k] for k in summary.keys() if k != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
