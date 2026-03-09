# IntentLoom

IntentLoom is a prototype for intent-guided iterative repair of network policy configurations.

This repository contains:

- `intentloom/`: core library code
- `scripts/`: experiment runners, dataset builders, and analysis utilities
- `data/`: released task files used for experiments, plus the real-world source index used to reconstruct the RealK8s dataset

## Installation

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

## Reproducibility Contents

Released task files include:

- NetConfEval-derived task sets
- synthetic K8s task sets
- RealK8s task sets

The repository also includes the scripts used to:

- build tasks from NetConfEval
- mine real-world Kubernetes NetworkPolicy sources
- build RealK8s tasks from the mined source index
- run experiments and aggregate results

## Notes

- OpenAI-compatible runs require environment variables such as `OPENAI_API_KEY` or `OPENROUTER_API_KEY`.
- Real-world raw source files are not committed here; `data/realworld_k8s_sources/index.json` preserves source provenance and reconstruction metadata.
