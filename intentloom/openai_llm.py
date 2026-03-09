from __future__ import annotations

import json
import os
import re
import time
from json import JSONDecodeError
from typing import Any, Dict, Optional

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


class OpenAILLM:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        default_headers: Optional[Dict[str, str]] = None,
        seed: Optional[int] = None,
        temperature: Optional[float] = 0.0,
    ):
        if OpenAI is None:
            raise RuntimeError("Missing dependency: openai. Install with: pip install openai")

        ignore_env_base_url = False
        if isinstance(base_url, str):
            norm = base_url.strip()
            if norm == "" or norm.lower() in {"default", "none", "null"}:
                base_url = None
                ignore_env_base_url = True
            else:
                base_url = norm

        env_base_url = (
            os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("OPENAI_API_BASE")
            or os.environ.get("OPENROUTER_BASE_URL")
            or os.environ.get("OPENROUTER_API_BASE")
            or os.environ.get("OPENROUTER_API_URL")
            or os.environ.get("OPENROUTER_URL")
        )
        if base_url is None and not ignore_env_base_url:
            base_url = env_base_url
        if isinstance(base_url, str):
            base_url = base_url.strip() or None

        base_url_l = str(base_url).lower() if base_url is not None else ""
        is_openrouter = bool(base_url) and ("openrouter" in base_url_l)

        if api_key:
            key = api_key
            key_source = "api_key"
        else:
            if is_openrouter:
                if os.environ.get("OPENROUTER_API_KEY"):
                    key = os.environ.get("OPENROUTER_API_KEY")
                    key_source = "OPENROUTER_API_KEY"
                else:
                    key = os.environ.get("OPENAI_API_KEY")
                    key_source = "OPENAI_API_KEY"
            else:
                if os.environ.get("OPENAI_API_KEY"):
                    key = os.environ.get("OPENAI_API_KEY")
                    key_source = "OPENAI_API_KEY"
                else:
                    key = os.environ.get("OPENROUTER_API_KEY")
                    key_source = "OPENROUTER_API_KEY"

        if isinstance(key, str):
            key = key.strip()

        if not key:
            raise RuntimeError(
                "Missing API key. Set OPENAI_API_KEY (OpenAI) or OPENROUTER_API_KEY (OpenRouter)."
            )

        self._key_source = key_source
        self._key_len = len(key) if isinstance(key, str) else None

        if timeout is None:
            env_timeout = os.environ.get("OPENAI_TIMEOUT")
            if env_timeout:
                try:
                    timeout = float(env_timeout)
                except ValueError:
                    timeout = None

        env_headers: Dict[str, str] = {}
        referer = os.environ.get("OPENROUTER_HTTP_REFERER")
        title = os.environ.get("OPENROUTER_APP_TITLE")
        if referer:
            env_headers["HTTP-Referer"] = referer
        if title:
            env_headers["X-Title"] = title

        headers = dict(env_headers)
        if isinstance(default_headers, dict):
            for k, v in default_headers.items():
                if v is None:
                    continue
                headers[str(k)] = str(v)

        self._base_url = base_url
        self._timeout = timeout
        self._default_headers = headers
        self._seed = seed

        try:
            tval = 0.0 if temperature is None else float(temperature)
        except Exception:
            tval = 0.0
        if tval < 0.0:
            tval = 0.0
        if tval > 2.0:
            tval = 2.0
        self._temperature = tval

        self._client = OpenAI(
            api_key=key,
            base_url=base_url,
            timeout=timeout,
            default_headers=(headers or None),
        )
        self._model = model

    def generate(self, prompt: Dict[str, Any], *, system: Optional[str] = None) -> Dict[str, Any]:
        if system is None:
            system = (
                "You are a configuration generator in a closed-loop verification system. "
                "Return ONLY a valid JSON object. No markdown, no code fences. "
                "The output must be a JSON object representing a candidate configuration. "
                "Prefer keys: links (list of [u,v]) and acl (list of {src,dst,action})."
                " If the prompt includes a 'blame_set' field, treat it as structured evidence of which configuration elements are responsible for the current violation(s), and prioritize minimal edits that change only those blamed elements."
                " If the prompt includes a 'reflection' field, treat it as your own short self-analysis of the failure and follow its minimal-change plan."
                " Do not include newline characters in any string values."
                " Output must be minified JSON on a single line."
            )

        def _call_once(user_content: str) -> str:
            kwargs = {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_content},
                ],
                "temperature": self._temperature,
                "max_tokens": 1024,
            }
            if self._seed is not None:
                try:
                    kwargs["seed"] = int(self._seed)
                except Exception:
                    pass
            try:
                try:
                    resp = self._client.chat.completions.create(
                        **kwargs,
                        response_format={"type": "json_object"},
                    )
                    return resp.choices[0].message.content or "{}"
                except TypeError as e:
                    try:
                        resp = self._client.chat.completions.create(**kwargs)
                        return resp.choices[0].message.content or "{}"
                    except TypeError as e2:
                        if "seed" in str(e) or "seed" in str(e2):
                            kwargs2 = dict(kwargs)
                            kwargs2.pop("seed", None)
                            resp = self._client.chat.completions.create(**kwargs2)
                            return resp.choices[0].message.content or "{}"
                        raise
                except Exception as e:
                    msg = str(e)
                    if "response_format" in msg:
                        resp = self._client.chat.completions.create(**kwargs)
                        return resp.choices[0].message.content or "{}"
                    if "seed" in msg and "seed" in kwargs:
                        kwargs2 = dict(kwargs)
                        kwargs2.pop("seed", None)
                        try:
                            resp = self._client.chat.completions.create(
                                **kwargs2,
                                response_format={"type": "json_object"},
                            )
                            return resp.choices[0].message.content or "{}"
                        except Exception:
                            resp = self._client.chat.completions.create(**kwargs2)
                            return resp.choices[0].message.content or "{}"
                    status = getattr(e, "status_code", None)
                    if (
                        status == 401
                        and self._base_url
                        and "openrouter.ai" in str(self._base_url).lower()
                        and self._key_source == "OPENROUTER_API_KEY"
                    ):
                        alt = os.environ.get("OPENAI_API_KEY")
                        if isinstance(alt, str):
                            alt = alt.strip()
                        if alt:
                            client2 = OpenAI(
                                api_key=alt,
                                base_url=self._base_url,
                                timeout=self._timeout,
                                default_headers=(self._default_headers or None),
                            )
                            try:
                                resp2 = client2.chat.completions.create(
                                    **kwargs,
                                    response_format={"type": "json_object"},
                                )
                                return resp2.choices[0].message.content or "{}"
                            except Exception:
                                pass
                    raise
            except Exception as e:
                raise RuntimeError(
                    "OpenAI request failed. "
                    "Check network connectivity and firewall/proxy settings. "
                    "If the error mentions unsupported_country_region_territory, the upstream provider is blocked in your region; switch to a different OpenRouter model/provider or run from a supported region. "
                    "If you need a gateway/proxy, set OPENAI_BASE_URL/OPENAI_API_BASE or OPENROUTER_BASE_URL and optionally OPENAI_TIMEOUT. "
                    f"base_url={self._base_url!r} timeout={self._timeout!r} "
                    f"default_headers_keys={sorted(self._default_headers.keys())!r} "
                    f"key_source={self._key_source!r} key_len={self._key_len!r} "
                    f"seed={self._seed!r} "
                    f"error={type(e).__name__}: {e}"
                ) from e

        def _call(user_content: str) -> str:
            last_err: Optional[Exception] = None
            for backoff_s in (0.0, 1.0, 2.0, 4.0):
                if backoff_s:
                    time.sleep(backoff_s)
                try:
                    return _call_once(user_content)
                except Exception as e:
                    last_err = e
                    msg = str(e).lower()
                    if (
                        "unsupported_country_region_territory" in msg
                        or "request_forbidden" in msg
                        or ("permissiondeniederror" in msg and "403" in msg)
                    ):
                        break
            if last_err is not None:
                raise last_err
            return "{}"

        user = json.dumps(prompt, ensure_ascii=False)
        content = _call(user)
        try:
            return _strip_strings(_parse_json_object(content))
        except Exception:
            pass

        repair = {
            "_error": "Previous output was not valid JSON. Return ONLY a valid JSON object.",
            "_bad_output": content[:2000],
            "prompt": prompt,
            "_requirements": "Return a SINGLE-LINE minified JSON object only. No newlines. No extra text.",
        }
        content2 = _call(json.dumps(repair, ensure_ascii=False))
        try:
            return _strip_strings(_parse_json_object(content2))
        except Exception:
            pass

        repair2 = {
            "_error": "Still invalid JSON. Return ONLY a JSON object.",
            "_bad_output": (content2[:2000] if isinstance(content2, str) else ""),
            "intent": prompt.get("intent"),
            "candidate_cfg": prompt.get("candidate_cfg"),
            "_requirements": "Return a SINGLE-LINE minified JSON object only. No newlines. No extra text.",
        }
        content3 = _call(json.dumps(repair2, ensure_ascii=False))
        try:
            return _strip_strings(_parse_json_object(content3))
        except Exception:
            base = prompt.get("candidate_cfg")
            return _strip_strings(base) if isinstance(base, dict) else {}

    def generate_text(
        self,
        prompt: Dict[str, Any],
        *,
        system: Optional[str] = None,
        max_tokens: int = 512,
    ) -> str:
        if system is None:
            system = "You are a helpful assistant."

        def _call_once(user_content: str) -> str:
            kwargs = {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_content},
                ],
                "temperature": self._temperature,
                "max_tokens": int(max_tokens) if isinstance(max_tokens, int) else 512,
            }
            if self._seed is not None:
                try:
                    kwargs["seed"] = int(self._seed)
                except Exception:
                    pass
            try:
                try:
                    resp = self._client.chat.completions.create(**kwargs)
                    return resp.choices[0].message.content or ""
                except TypeError as e:
                    if "seed" in str(e) and "seed" in kwargs:
                        kwargs2 = dict(kwargs)
                        kwargs2.pop("seed", None)
                        resp = self._client.chat.completions.create(**kwargs2)
                        return resp.choices[0].message.content or ""
                    raise
                except Exception as e:
                    msg = str(e)
                    if "seed" in msg and "seed" in kwargs:
                        kwargs2 = dict(kwargs)
                        kwargs2.pop("seed", None)
                        resp = self._client.chat.completions.create(**kwargs2)
                        return resp.choices[0].message.content or ""
                    raise
            except Exception as e:
                raise RuntimeError(
                    "OpenAI request failed. "
                    "Check network connectivity and firewall/proxy settings. "
                    "If the error mentions unsupported_country_region_territory, the upstream provider is blocked in your region; switch to a different OpenRouter model/provider or run from a supported region. "
                    "If you need a gateway/proxy, set OPENAI_BASE_URL/OPENAI_API_BASE or OPENROUTER_BASE_URL and optionally OPENAI_TIMEOUT. "
                    f"base_url={self._base_url!r} timeout={self._timeout!r} "
                    f"default_headers_keys={sorted(self._default_headers.keys())!r} "
                    f"key_source={self._key_source!r} key_len={self._key_len!r} "
                    f"seed={self._seed!r} "
                    f"error={type(e).__name__}: {e}"
                ) from e

        def _call(user_content: str) -> str:
            last_err: Optional[Exception] = None
            for backoff_s in (0.0, 1.0, 2.0, 4.0):
                if backoff_s:
                    time.sleep(backoff_s)
                try:
                    return _call_once(user_content)
                except Exception as e:
                    last_err = e
                    msg = str(e).lower()
                    if (
                        "unsupported_country_region_territory" in msg
                        or "request_forbidden" in msg
                        or ("permissiondeniederror" in msg and "403" in msg)
                    ):
                        break
            if last_err is not None:
                raise last_err
            return ""

        user = json.dumps(prompt, ensure_ascii=False)
        return _call(user).strip()


def _parse_json_object(text: str) -> Dict[str, Any]:
    s = text.strip()

    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", s, flags=re.DOTALL | re.IGNORECASE)
    if m:
        s = m.group(1).strip()

    try:
        obj = json.loads(_escape_unescaped_newlines_in_strings(s))
    except json.JSONDecodeError:
        extracted = _extract_first_json_object(s)
        if extracted is not None:
            obj = json.loads(_escape_unescaped_newlines_in_strings(extracted))
        else:
            start = s.find("{")
            end = s.rfind("}")
            if start >= 0 and end > start:
                obj = json.loads(_escape_unescaped_newlines_in_strings(s[start : end + 1]))
            else:
                raise

    if not isinstance(obj, dict):
        raise ValueError("OpenAI response is not a JSON object")
    return obj


def _strip_strings(x: Any) -> Any:
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, list):
        return [_strip_strings(v) for v in x]
    if isinstance(x, dict):
        return {k: _strip_strings(v) for k, v in x.items()}
    return x


def _escape_unescaped_newlines_in_strings(text: str) -> str:
    out = []
    in_str = False
    esc = False
    for ch in text:
        if in_str:
            if esc:
                esc = False
                out.append(ch)
                continue
            if ch == "\\":
                esc = True
                out.append(ch)
                continue
            if ch == '"':
                in_str = False
                out.append(ch)
                continue
            if ch == "\n":
                out.append("\\n")
                continue
            if ch == "\r":
                out.append("\\r")
                continue
            out.append(ch)
            continue

        if ch == '"':
            in_str = True
        out.append(ch)
    return "".join(out)


def _extract_first_json_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None
