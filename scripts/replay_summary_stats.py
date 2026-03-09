from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class TemplateStats:
    template: str
    n: int = 0
    old_ok: int = 0
    old_fail: int = 0
    old_unknown: int = 0
    replay_any_ok: int = 0
    fail_to_ok: int = 0
    ok_to_no_ok: int = 0
    earlier_ok: int = 0
    rounds_saved_sum: int = 0
    rounds_saved_n: int = 0

    def add_row(self, r: Dict[str, Any]) -> None:
        self.n += 1

        old_final_ok = r.get("old_final_ok")
        if old_final_ok is True:
            self.old_ok += 1
        elif old_final_ok is False:
            self.old_fail += 1
        else:
            self.old_unknown += 1

        rf = r.get("replay_first_ok_round")
        of = r.get("old_first_ok_round")

        replay_has_ok = _is_int(rf)
        old_has_ok = _is_int(of)

        if replay_has_ok:
            self.replay_any_ok += 1

        if old_final_ok is False and replay_has_ok:
            self.fail_to_ok += 1

        if old_final_ok is True and not replay_has_ok:
            self.ok_to_no_ok += 1

        if old_has_ok and replay_has_ok and int(rf) < int(of):
            self.earlier_ok += 1

        saved = r.get("rounds_saved_if_early_stop")
        if _is_int(saved):
            self.rounds_saved_sum += int(saved)
            self.rounds_saved_n += 1

    def old_success_rate(self) -> float:
        return (self.old_ok / self.n) if self.n else 0.0

    def counterfactual_success_rate(self) -> float:
        return (self.replay_any_ok / self.n) if self.n else 0.0

    def avg_rounds_saved(self) -> float:
        return (self.rounds_saved_sum / self.rounds_saved_n) if self.rounds_saved_n else 0.0


def _summarize(path: Path, *, top_k: int) -> Dict[str, Any]:
    d = _read_json(path)
    rows = d.get("rows") if isinstance(d, dict) else None
    if not isinstance(rows, list):
        rows = []

    old_ok = sum(1 for r in rows if isinstance(r, dict) and r.get("old_final_ok") is True)
    old_fail = sum(1 for r in rows if isinstance(r, dict) and r.get("old_final_ok") is False)
    old_unknown = len(rows) - old_ok - old_fail

    counterfactual_ok = sum(
        1 for r in rows if isinstance(r, dict) and _is_int(r.get("replay_first_ok_round"))
    )
    fail_to_ok = sum(
        1
        for r in rows
        if isinstance(r, dict)
        and r.get("old_final_ok") is False
        and _is_int(r.get("replay_first_ok_round"))
    )
    ok_to_no_ok = sum(
        1
        for r in rows
        if isinstance(r, dict)
        and r.get("old_final_ok") is True
        and not _is_int(r.get("replay_first_ok_round"))
    )
    earlier_ok = sum(
        1
        for r in rows
        if isinstance(r, dict)
        and _is_int(r.get("old_first_ok_round"))
        and _is_int(r.get("replay_first_ok_round"))
        and int(r.get("replay_first_ok_round")) < int(r.get("old_first_ok_round"))
    )

    saved = [
        int(r.get("rounds_saved_if_early_stop"))
        for r in rows
        if isinstance(r, dict) and _is_int(r.get("rounds_saved_if_early_stop"))
    ]

    by_t: Dict[str, TemplateStats] = defaultdict(lambda: TemplateStats(template=""))
    for r in rows:
        if not isinstance(r, dict):
            continue
        t = str(r.get("template") or "")
        st = by_t.get(t)
        if st is None or st.template == "":
            st = TemplateStats(template=t)
            by_t[t] = st
        st.add_row(r)

    tpl_list = list(by_t.values())
    tpl_list.sort(key=lambda x: x.template)

    top_fail_to_ok = sorted(tpl_list, key=lambda s: s.fail_to_ok, reverse=True)[:top_k]
    top_ok_to_no_ok = sorted(tpl_list, key=lambda s: s.ok_to_no_ok, reverse=True)[:top_k]
    top_earlier_ok = sorted(tpl_list, key=lambda s: s.earlier_ok, reverse=True)[:top_k]

    return {
        "path": str(path),
        "n_rows": len(rows),
        "old_ok": old_ok,
        "old_fail": old_fail,
        "old_unknown": old_unknown,
        "counterfactual_ok": counterfactual_ok,
        "counterfactual_success_rate": (counterfactual_ok / len(rows)) if rows else 0.0,
        "fail_to_ok": fail_to_ok,
        "ok_to_no_ok": ok_to_no_ok,
        "earlier_ok": earlier_ok,
        "avg_rounds_saved_if_early_stop": (sum(saved) / len(saved)) if saved else 0.0,
        "top_templates_fail_to_ok": [
            {
                "template": s.template,
                "fail_to_ok": s.fail_to_ok,
                "n": s.n,
                "old_sr": s.old_success_rate(),
                "cf_sr": s.counterfactual_success_rate(),
                "avg_saved": s.avg_rounds_saved(),
            }
            for s in top_fail_to_ok
            if s.fail_to_ok > 0
        ],
        "top_templates_ok_to_no_ok": [
            {
                "template": s.template,
                "ok_to_no_ok": s.ok_to_no_ok,
                "n": s.n,
                "old_sr": s.old_success_rate(),
                "cf_sr": s.counterfactual_success_rate(),
                "avg_saved": s.avg_rounds_saved(),
            }
            for s in top_ok_to_no_ok
            if s.ok_to_no_ok > 0
        ],
        "top_templates_earlier_ok": [
            {
                "template": s.template,
                "earlier_ok": s.earlier_ok,
                "n": s.n,
                "old_sr": s.old_success_rate(),
                "cf_sr": s.counterfactual_success_rate(),
                "avg_saved": s.avg_rounds_saved(),
            }
            for s in top_earlier_ok
            if s.earlier_ok > 0
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument(
        "--paths",
        type=str,
        default=",".join(
            [
                "artifacts/ports100_option_c_generic/replay_verify_symbolicai/replay_summary.json",
                "artifacts/ports100_option_c_raw_witness/replay_verify_symbolicai/replay_summary.json",
                "artifacts/ports100_option_c_blame_only/replay_verify_symbolicai/replay_summary.json",
            ]
        ),
        help="Comma-separated replay_summary.json paths.",
    )
    ap.add_argument("--json", type=str, default=None, help="If set, write full stats JSON to this path.")
    args = ap.parse_args()

    paths = [Path(p.strip()) for p in str(args.paths).split(",") if p.strip()]

    out: Dict[str, Any] = {
        "top_k": int(args.top_k),
        "summaries": [],
    }

    for p in paths:
        out["summaries"].append(_summarize(p, top_k=int(args.top_k)))

    print(json.dumps(out, ensure_ascii=False, indent=2))

    if args.json:
        Path(args.json).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
