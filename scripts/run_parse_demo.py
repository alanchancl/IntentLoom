import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from intentloom import parse_intent


def main() -> None:
    samples = [
        "Isolate finance from office AND Reach finance internet",
        "NOT (Deny finance internet) AND Allow finance internet",
        "隔离 finance office AND 可达 finance internet",
        "Require mode production",
    ]

    for s in samples:
        node = parse_intent(s)
        print({"input": s, "intent": node.to_dict()})


if __name__ == "__main__":
    main()
