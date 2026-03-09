from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from .compiler import CompiledIntent, compile_intent
from .feedback import render_feedback
from .llm import LLM
from .verifier import Verifier, VerifierLike, VerificationResult


@dataclass
class LoopStep:
    round: int
    prompt: Dict[str, Any]
    llm_input: Optional[Dict[str, Any]]
    reflection: Optional[str]
    candidate_cfg: Dict[str, Any]
    verification: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round": self.round,
            "prompt": self.prompt,
            "llm_input": self.llm_input,
            "reflection": self.reflection,
            "candidate_cfg": self.candidate_cfg,
            "verification": self.verification,
        }


@dataclass
class LoopResult:
    ok: bool
    final_cfg: Dict[str, Any]
    steps: List[LoopStep]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "final_cfg": self.final_cfg,
            "steps": [s.to_dict() for s in self.steps],
        }

    def write_jsonl(self, path: Union[str, Path]) -> None:
        p = Path(path)
        if p.parent:
            p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for s in self.steps:
                f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")


AblationMode = Literal["generic", "standard_feedback", "raw_witness", "blame_only"]


def _normalize_ablation_mode(mode: str) -> str:
    m = str(mode or "").strip().lower()
    if m == "standard_feedback":
        return "generic"
    if m in {"generic", "raw_witness", "blame_only"}:
        return m
    return "raw_witness"


def _project_verification_codes_only(verification: Dict[str, Any]) -> Dict[str, Any]:
    violations_in = verification.get("violations")
    violations_out: List[Dict[str, Any]] = []
    if isinstance(violations_in, list):
        for v in violations_in:
            if not isinstance(v, dict):
                continue
            violations_out.append(
                {
                    "code": v.get("code"),
                    "path": v.get("path"),
                }
            )
    return {
        "ok": bool(verification.get("ok")),
        "violations": violations_out,
    }


def _extract_first_bracket_list(text: str) -> Optional[List[Any]]:
    m = re.search(r"\[[^\[\]]*\]", text)
    if not m:
        return None
    try:
        val = ast.literal_eval(m.group(0))
    except Exception:
        return None
    return val if isinstance(val, list) else None


def _compute_blame_set(cfg: Dict[str, Any], vr: VerificationResult) -> Dict[str, Any]:
    blamed_links: List[List[str]] = []
    blamed_policies: List[str] = []

    domain = "netconfeval"
    if isinstance(cfg, dict) and "k8s" in cfg:
        domain = "k8s"

    for v in vr.violations:
        meta = v.meta or {}
        exc = meta.get("contract_exception") if isinstance(meta, dict) else None
        if domain == "k8s":
            meta_pols = meta.get("policy_evidence") if isinstance(meta, dict) else None
            if isinstance(meta_pols, list):
                for p in meta_pols:
                    p2 = str(p).strip()
                    if p2 and p2 not in blamed_policies:
                        blamed_policies.append(p2)

            text_sources: List[str] = []
            if isinstance(exc, str) and exc.strip():
                text_sources.append(exc)
            pol_str = meta.get("policy_evidence_str") if isinstance(meta, dict) else None
            if isinstance(pol_str, str) and pol_str.strip():
                text_sources.append(pol_str)
            if isinstance(v.message, str) and v.message.strip():
                text_sources.append(v.message)

            pols: List[str] = []
            for t in text_sources:
                pols.extend(re.findall(r"policy=([A-Za-z0-9_.:-]+)", t))
            for p in pols:
                p2 = str(p).strip()
                if p2 and p2 not in blamed_policies:
                    blamed_policies.append(p2)
            continue

        if not isinstance(exc, str) or not exc.strip():
            continue

        # netconfeval: parse a concrete counterexample path list and blame its edges
        path_list = _extract_first_bracket_list(exc)
        if not path_list or len(path_list) < 2:
            continue
        path_nodes = [str(x) for x in path_list]
        for i in range(len(path_nodes) - 1):
            edge = [path_nodes[i], path_nodes[i + 1]]
            if edge not in blamed_links:
                blamed_links.append(edge)

    return {
        "blamed_links": blamed_links,
        "blamed_policies": blamed_policies,
    }


def _generic_feedback_from_violations(vr: VerificationResult) -> str:
    if vr.ok:
        return "All constraints satisfied."
    codes = []
    for v in vr.violations:
        if v.code and v.code not in codes:
            codes.append(v.code)
    if not codes:
        return "Verification failed. Revise the configuration to satisfy the intent."
    return "Verification failed. Fix violations: " + ", ".join(codes) + "."


class IntentLoomRunner:
    def __init__(self, verifier: Optional[VerifierLike] = None):
        self._verifier: VerifierLike = verifier or Verifier()

    def run(
        self,
        intent,
        llm: LLM,
        initial_prompt: Optional[Dict[str, Any]] = None,
        max_rounds: int = 3,
        *,
        initial_cfg: Optional[Dict[str, Any]] = None,
        ablation_mode: AblationMode = "raw_witness",
        include_initial_verification_in_prompt: bool = True,
        reflexion: bool = False,
        reflexion_max_tokens: int = 256,
    ) -> LoopResult:
        compiled: CompiledIntent = compile_intent(intent)
        steps: List[LoopStep] = []

        ablation_mode_norm = _normalize_ablation_mode(str(ablation_mode))
        prompt: Dict[str, Any] = dict(initial_prompt or {})
        if ablation_mode_norm != "blame_only":
            prompt.pop("blame_set", None)
        prompt["intent"] = compiled.feedback_schema["intent"]

        last_cfg: Dict[str, Any] = {}
        if isinstance(initial_cfg, dict):
            last_cfg = dict(initial_cfg)
            prompt["candidate_cfg"] = initial_cfg
            if include_initial_verification_in_prompt:
                vr0 = self._verifier.verify(initial_cfg, compiled.constraints)
                verification_full0 = vr0.to_dict()
                if ablation_mode_norm == "raw_witness":
                    prompt_verification0 = verification_full0
                    prompt_feedback0 = render_feedback(vr0.violations)
                elif ablation_mode_norm == "blame_only":
                    prompt_verification0 = _project_verification_codes_only(verification_full0)
                    blame0 = _compute_blame_set(initial_cfg, vr0)
                    prompt["blame_set"] = blame0
                    prompt_feedback0 = render_feedback(vr0.violations)
                else:
                    prompt_verification0 = _project_verification_codes_only(verification_full0)
                    prompt_feedback0 = render_feedback(vr0.violations)

                prompt["verification"] = prompt_verification0
                prompt["feedback"] = prompt_feedback0
        for r in range(1, max_rounds + 1):
            llm_input = dict(prompt)
            reflection_text: Optional[str] = None
            if reflexion:
                if not hasattr(llm, "generate_text"):
                    raise RuntimeError(
                        "Reflexion mode requires an LLM with a generate_text(prompt, system=..., max_tokens=...) method."
                    )
                reflection_prompt = {
                    "intent": compiled.feedback_schema["intent"],
                    "candidate_cfg": llm_input.get("candidate_cfg"),
                    "verification": llm_input.get("verification"),
                    "feedback": llm_input.get("feedback"),
                }
                reflection_sys = (
                    "You are debugging a configuration repair attempt in a closed-loop verifier. "
                    "Given the intent and the verifier feedback, produce a short reflection with: "
                    "(1) suspected root cause, (2) minimal-change repair plan. "
                    "Do NOT propose large rewrites. Output plain text only."
                )
                reflection_text = llm.generate_text(
                    reflection_prompt,
                    system=reflection_sys,
                    max_tokens=int(reflexion_max_tokens) if isinstance(reflexion_max_tokens, int) else 256,
                )
                if isinstance(reflection_text, str) and len(reflection_text) > 1200:
                    reflection_text = reflection_text[:1200]
                llm_input["reflection"] = reflection_text

            candidate = llm.generate(llm_input)
            last_cfg = candidate
            vr = self._verifier.verify(candidate, compiled.constraints)
            verification_full = vr.to_dict()

            if ablation_mode_norm == "raw_witness":
                prompt_verification = verification_full
                prompt_feedback = render_feedback(vr.violations)
                blame_set = None
            elif ablation_mode_norm == "blame_only":
                prompt_verification = _project_verification_codes_only(verification_full)
                blame_set = _compute_blame_set(candidate, vr)
                prompt_feedback = render_feedback(vr.violations)
            else:
                prompt_verification = _project_verification_codes_only(verification_full)
                prompt_feedback = render_feedback(vr.violations)
                blame_set = None

            prompt = {
                "intent": compiled.feedback_schema["intent"],
                "candidate_cfg": candidate,
                "verification": prompt_verification,
                "feedback": prompt_feedback,
            }
            if isinstance(blame_set, dict):
                prompt["blame_set"] = blame_set
            steps.append(
                LoopStep(
                    round=r,
                    prompt=prompt,
                    llm_input=llm_input,
                    reflection=reflection_text,
                    candidate_cfg=candidate,
                    verification=verification_full,
                )
            )
            if vr.ok:
                return LoopResult(ok=True, final_cfg=candidate, steps=steps)

        return LoopResult(ok=False, final_cfg=last_cfg, steps=steps)
