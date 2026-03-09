import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from intentloom import IntentLoomRunner, Isolate, Reach
from intentloom.llm import RuleBasedLLM


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=str, default="")
    args = ap.parse_args()

    intent = Isolate("finance", "office") & Reach("finance", "internet")

    initial_cfg = {
        "links": [
            ["finance", "office"],
            ["finance", "internet"],
        ],
        "acl": [],
    }

    llm = RuleBasedLLM(initial_cfg=initial_cfg)
    runner = IntentLoomRunner()
    result = runner.run(intent=intent, llm=llm, max_rounds=3)

    if args.jsonl:
        result.write_jsonl(args.jsonl)

    print({"ok": result.ok, "final_cfg": result.final_cfg})
    for s in result.steps:
        print({"round": s.round, "verification": s.verification})


if __name__ == "__main__":
    main()
