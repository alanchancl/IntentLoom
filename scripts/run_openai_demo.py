import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from intentloom import IntentLoomRunner, parse_intent
from intentloom.openai_llm import OpenAILLM


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--intent", type=str, default="Isolate finance from office AND Reach finance internet")
    ap.add_argument("--model", type=str, default="gpt-4o-mini")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--jsonl", type=str, default=".\\artifacts\\openai_demo.jsonl")
    args = ap.parse_args()

    intent = parse_intent(args.intent)

    llm = OpenAILLM(model=args.model)
    runner = IntentLoomRunner()

    result = runner.run(intent=intent, llm=llm, max_rounds=args.rounds)
    if args.jsonl:
        result.write_jsonl(args.jsonl)

    print({"ok": result.ok, "rounds": len(result.steps), "final_cfg": result.final_cfg})


if __name__ == "__main__":
    main()
