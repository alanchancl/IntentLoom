import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from intentloom import IntentLoomRunner, Isolate, Reach


def main() -> None:
    try:
        from intentloom.symbolicai_verifier import SymbolicAIContractVerifier
    except Exception as e:
        print(f"SymbolicAI verifier is not available: {e}")
        print("Install dependencies first: pip install -r requirements.txt")
        return

    intent = Isolate("finance", "office") & Reach("finance", "internet")

    cfg = {
        "links": [["finance", "office"], ["finance", "internet"]],
        "acl": [],
    }

    runner = IntentLoomRunner(verifier=SymbolicAIContractVerifier())

    class OneShotLLM:
        def generate(self, prompt):
            return cfg

    result = runner.run(intent=intent, llm=OneShotLLM(), max_rounds=1)
    print(result.to_dict())


if __name__ == "__main__":
    main()
