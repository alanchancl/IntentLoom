from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..openai_llm import OpenAILLM


_K8S_SYSTEM_PROMPT = """You are a configuration generator in a closed-loop verification system for Kubernetes NetworkPolicy.
Return ONLY a valid JSON object. No markdown, no code fences.

The output must be a JSON object representing a candidate Kubernetes policy configuration.
Use this schema (keys are required):
{
  "k8s": {
    "pods": [ {"name": str, "namespace": str, "labels": {str: str}} ],
    "services": [ {"name": str, "namespace": str, "selector": {str: str}} ]
  },
  "policies": [
    {
      "name": str,
      "action": "allow",
      "from": {"namespace": str|"*"|null, "podSelector": {str: str}|null, "ipBlock": str|null},
      "to":   {"namespace": str|"*"|null, "podSelector": {str: str}|null},
      "ports": null | [ {"port": int, "protocol": "TCP"|"UDP"|null} ]
    }
  ]
}

Interpret connectivity as allowed iff there exists at least one policy selecting the source pod and destination pod.
If a policy includes `ports`, it only allows those ports (and optionally protocol). If `ports` is null/missing, it allows all ports.
If `k8s.semantics` is `option_c`, treat connectivity as directional: a policy allows traffic from its `from` selector to its `to` selector (not necessarily the reverse).
Try to make minimal edits to fix violations.

`podSelector` is a LABEL selector (matchLabels semantics). Do NOT put pod names in `podSelector`.
Use the `labels` shown under `k8s.pods[*].labels` when selecting pods.

If a policy uses `from.ipBlock`, it is a CIDR string. For a single host IP like `ip:10.0.0.1`, prefer `10.0.0.1/32` (or an equivalent CIDR that contains it).

Do NOT change `k8s.pods` or `k8s.services` (treat them as fixed ground truth). Only edit `policies`.

IMPORTANT: only `action="allow"` is supported by the verifier in this experiment. Do NOT output `deny`/`block` policies.
If you need to enforce isolation, you must REMOVE or NARROW the existing allow policy/rule that enables the forbidden path.

IMPORTANT: There is NO deny semantics in this simplified policy language.
Do NOT invent a "deny" effect by creating policies whose names look like `deny-*`/`block-*` while using `action="allow"`.
Those policies still ALLOW traffic and will not help isolation.

IMPORTANT: If the intent specifies `port=...` (and optionally `protocol=...`), you must NOT output `ports: null` or omit `ports`.
`ports: null` means allow ALL ports and is almost always over-permissive for these tasks.
Instead, explicitly set `ports` to a list of concrete port records that match the intent.

If the prompt includes `blame_set`, treat it as precise evidence of which policy/rule(s) are responsible; prioritize minimal edits that change only those blamed policies.

When fixing `ISOLATE_VIOLATION`, prefer deleting or narrowing the blamed allow policy (e.g., restrict namespace/podSelector/ports) rather than adding new allow policies.

If the blamed allow policy directly enables the forbidden connectivity (e.g., it is an allow from the isolated source to the isolated destination on the forbidden port), the minimal correct fix is usually to DELETE that policy entirely.
Do NOT try to "fix" isolation by setting `ports: null` (that allows MORE traffic and will not help).

If the prompt includes `reflection`, treat it as your own short self-analysis and follow its minimal-change repair plan.

If the prompt includes `few_shot_examples`, treat it as a list of (intent, before, after) examples; follow the style of the `after` policies.

If a violation contains `meta.contract_exception`, treat it as precise symbolic evidence (e.g., witness path) and adjust the YAML/JSON to break that path (for Isolate) or create a missing path (for Reach)."""


@dataclass
class K8sOpenAILLM:
    inner: OpenAILLM
    system: str = _K8S_SYSTEM_PROMPT

    def generate(self, prompt: Dict[str, Any]) -> Dict[str, Any]:
        out = self.inner.generate(prompt, system=self.system)
        if not isinstance(out, dict):
            return {}

        pols = out.get("policies")
        if isinstance(pols, list):
            cleaned = []
            for i, p in enumerate(pols):
                if not isinstance(p, dict):
                    continue
                action_raw = p.get("action", "allow")
                action = action_raw.strip().lower() if isinstance(action_raw, str) else "allow"
                if action != "allow":
                    continue
                p2 = dict(p)
                p2["action"] = "allow"
                name = p2.get("name")
                if not isinstance(name, str) or not name.strip():
                    p2["name"] = f"policy-{i+1}"
                frm = p2.get("from")
                to = p2.get("to")
                p2["from"] = frm if isinstance(frm, dict) else {"namespace": None, "podSelector": None}
                p2["to"] = to if isinstance(to, dict) else {"namespace": None, "podSelector": None}
                cleaned.append(p2)
            out["policies"] = cleaned

        prev = prompt.get("candidate_cfg")
        if not isinstance(prev, dict):
            return out

        prev_k8s = prev.get("k8s")
        if not isinstance(prev_k8s, dict):
            return out

        out_k8s = out.get("k8s")
        if not isinstance(out_k8s, dict):
            out_k8s = {}
            out["k8s"] = out_k8s

        if isinstance(prev_k8s.get("pods"), list):
            out_k8s["pods"] = prev_k8s.get("pods")
        if isinstance(prev_k8s.get("services"), list):
            out_k8s["services"] = prev_k8s.get("services")

        if isinstance(prev_k8s.get("semantics"), str) and prev_k8s.get("semantics"):
            out_k8s["semantics"] = prev_k8s.get("semantics")

        if isinstance(prev_k8s.get("external_ips"), list):
            out_k8s["external_ips"] = prev_k8s.get("external_ips")

        return out

    def generate_text(
        self,
        prompt: Dict[str, Any],
        *,
        system: Optional[str] = None,
        max_tokens: int = 512,
    ) -> str:
        return self.inner.generate_text(prompt, system=system, max_tokens=max_tokens)


def make_k8s_openai_llm(
    *,
    model: str = "gpt-4o-mini",
    base_url: Optional[str] = None,
    timeout: Optional[float] = None,
    default_headers: Optional[Dict[str, str]] = None,
    seed: Optional[int] = None,
    temperature: Optional[float] = 0.0,
) -> K8sOpenAILLM:
    return K8sOpenAILLM(
        inner=OpenAILLM(
            model=model,
            base_url=base_url,
            timeout=timeout,
            default_headers=default_headers,
            seed=seed,
            temperature=temperature,
        )
    )
