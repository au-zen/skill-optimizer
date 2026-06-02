"""
SkillOpt LLM Client — Optimizer model API bridge.

Calls OpenAI-compatible chat completion endpoints (OpenRouter, etc.)
for the 8 optimizer prompt contracts.  Supports:

  - Multiple API key resolution strategies
  - Automatic fallback across 3 tiers of providers
  - JSON-mode parsing for structured edit output
  - Retry with exponential backoff
  - Token-budget tracking
"""

from __future__ import annotations

import json
import os
import re
import time
import warnings
from pathlib import Path
from typing import Any

import httpx

# ── Auto-load .env from skill root (override mode — no python-dotenv) ─
# NOTE: this is the ONLY config source.  The .env file must contain
# SKILLOPT_API_KEY + SKILLOPT_MODEL + SKILLOPT_API_BASE to be usable.
_skill_root = Path(__file__).resolve().parent.parent
_env_path = _skill_root / ".env"
if _env_path.exists():
    for _line in _env_path.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _key, _sep, _val = _line.partition("=")
            _key = _key.strip()
            _val = _val.strip().strip("'\"")
            if _key and _val:
                os.environ[_key] = _val  # override — decouple from parent env

# ── Config resolution (env-only, no Hermes / system dependency) ───────


def _resolve_all_providers() -> list[dict]:
    """Return all configured providers in priority order from .env.

    Reads ``SKILLOPT_API_KEY[_2][_3]``, ``SKILLOPT_MODEL[_2][_3]``,
    ``SKILLOPT_API_BASE[_2][_3]`` from the environment (seeded by .env
    at module import time).

    Returns a list of dicts with keys: ``model``, ``api_base``, ``api_key``,
    ``label``.  Empty list if nothing is configured.
    """
    providers: list[dict] = []

    # Helper: assemble a provider entry if the api_key is present
    def _push(suffix: str, label: str, default_key: str, default_model: str, default_base: str):
        key = os.environ.get(f"SKILLOPT_API_KEY{suffix}", "") or default_key
        if not key:
            return
        providers.append({
            "model": os.environ.get(f"SKILLOPT_MODEL{suffix}", default_model),
            "api_base": os.environ.get(f"SKILLOPT_API_BASE{suffix}", default_base).rstrip("/"),
            "api_key": key,
            "label": label,
        })

    # Tier 1: opencode-zen (highest priority)
    _push("", "opencode-zen", "", "deepseek-v4-flash-free", "https://opencode.ai/zen/v1")

    # Tier 2: DeepSeek direct (or SKILLOPT_API_KEY_2)
    _push("_2", "deepseek-direct",
          os.environ.get("DEEPSEEK_API_KEY", ""),
          "deepseek-v4-flash",
          "https://api.deepseek.com/v1")

    # Tier 3: OpenRouter free model (or SKILLOPT_API_KEY_3)
    _push("_3", "openrouter-free",
          os.environ.get("OPENROUTER_API_KEY", os.environ.get("OPENROUTER_API_KEY_2", "")),
          "google/gemma-4-31b-it:free",
          "https://openrouter.ai/api/v1")

    return providers


def _resolve_hermes_default_model() -> dict[str, str]:
    """Resolve model config purely from .env / environment vars.

    The .env file (loaded at import time via override mode) is the ONLY
    source.  No Hermes config.yaml, no credential pool, no parent-process
    env inheritance -- complete decoupling from Hermes and the system.
    """
    model = os.environ.get("SKILLOPT_MODEL", "")
    api_base = os.environ.get("SKILLOPT_API_BASE", "")
    api_key = os.environ.get("SKILLOPT_API_KEY", "") or ""

    # opencode-zen gateway — use directly, no rewrite needed
    if "opencode" in api_base:
        return {"model": model, "api_base": api_base, "api_key": api_key, "provider": "opencode-zen"}

    # No-auth providers
    if "localhost" in api_base:
        api_key = ""

    provider = "openrouter" if "openrouter" in api_base else "env"
    return {"model": model, "api_base": api_base, "api_key": api_key, "provider": provider}


def _resolve_api_key(model_hint: str = "") -> str | None:
    """Try env vars only (from .env override + maybe parent env).

    When model_hint contains 'deepseek', DEEPSEEK_API_KEY is preferred
    over generic keys (OPENROUTER_API_KEY etc.) to avoid sending the
    wrong API key to the DeepSeek endpoint.
    """

    is_deepseek = "deepseek" in model_hint.lower()

    # Direct env var (including OpenRouter multi-key pattern)
    env_vars = [
        "SKILLOPT_API_KEY",
        "DEEPSEEK_API_KEY",
        "OPENROUTER_API_KEY",
        "OPENROUTER_API_KEY_2",
        "OPENROUTER_API_KEY_3",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "CLOUDFLARE_GEMINI_API_KEY",
    ]
    for var in env_vars:
        val = os.environ.get(var)
        if val and val not in ("", "***"):
            # For deepseek models, only accept DEEPSEEK_API_KEY or SKILLOPT_API_KEY
            if is_deepseek and var not in ("SKILLOPT_API_KEY", "DEEPSEEK_API_KEY"):
                continue
            return val

    return None


def _resolve_api_base(model_hint: str = "") -> str:
    """Return the API base URL for the optimizer model.

    Priority:
      1. SKILLOPT_API_BASE env var
      2. DEEPSEEK_BASE_URL env var (when model starts with 'deepseek')
      3. OPENROUTER_BASE_URL env var
      4. Default: https://api.deepseek.com/v1 (for deepseek models)
         or https://openrouter.ai/api/v1
    """
    if explicit := os.environ.get("SKILLOPT_API_BASE"):
        return explicit

    model = model_hint or os.environ.get("SKILLOPT_MODEL", "")

    if "deepseek" in model.lower():
        return os.environ.get(
            "DEEPSEEK_BASE_URL",
            "https://api.deepseek.com/v1",
        )

    return os.environ.get(
        "OPENROUTER_BASE_URL",
        "https://openrouter.ai/api/v1",
    )


def _resolve_model() -> str:
    """Return the optimizer model name."""
    return os.environ.get("SKILLOPT_MODEL", "deepseek-chat")


# ── Rate-limiter helper ───────────────────────────────────────────────


class _SimpleRateLimiter:
    """Token-bucket-ish rate limiter: min interval between calls."""

    def __init__(self, calls_per_minute: int = 30):
        self.interval = 60.0 / max(calls_per_minute, 1)
        self._last_call = 0.0

    def wait(self):
        elapsed = time.time() - self._last_call
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self._last_call = time.time()


# ── Structured response extraction ────────────────────────────────────

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\n(.+?)\n```", re.DOTALL)


def _extract_json(text: str) -> list[dict] | dict | None:
    """Try to extract structured JSON from LLM output.

    Tries (in order):
      1. ```json ... ``` code blocks
      2. Direct top-level JSON parse
      3. Nested JSON object inside text
    """
    # First: code blocks
    for block in _JSON_BLOCK_RE.findall(text):
        try:
            result = json.loads(block)
            if isinstance(result, (list, dict)):
                return result
        except json.JSONDecodeError:
            continue

    # Second: direct parse of trimmed text
    cleaned = text.strip()
    if cleaned.startswith("[") or cleaned.startswith("{"):
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

    # Third: find the first { … } or [ … ] in the text
    for start_char, end_char in (("{", "}"), ("[", "]")):
        start = cleaned.find(start_char)
        if start >= 0:
            depth = 0
            for i in range(start, len(cleaned)):
                c = cleaned[i]
                if c == start_char:
                    depth += 1
                elif c == end_char:
                    depth -= 1
                    if depth == 0:
                        candidate = cleaned[start : i + 1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            break
    return None


# _HTTP_STATUSES that should trigger a provider fallback
# These indicate temporary / server-side / capacity issues.
# 401 Unauthorized is also treated as fallback to allow trying next provider in chain.
_FALLBACK_STATUSES = {401, 429, 500, 502, 503, 504, 507, 508}


# ── Main client ───────────────────────────────────────────────────────


class OptimizerLLMClient:
    """Client for calling the optimizer model (frontier LLM).

    Supports automatic fallback across multiple providers.  When created
    via ``from_hermes_config()``, it reads the 3-tier chain from .env
    and tries each one in order if the previous fails.
    """

    def __init__(
        self,
        model: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        calls_per_minute: int = 30,
        _providers: list[dict] | None = None,  # internal: multi-provider chain
    ):
        # ── Multi-provider mode ───────────────────────────────────────
        self._providers = _providers
        if self._providers:
            p = self._providers[0]
            self.model = p["model"]
            self.api_base = p["api_base"]
            self.api_key = p["api_key"]
        else:
            self.model = model or _resolve_model()
            self.api_base = (api_base or _resolve_api_base(self.model)).rstrip("/")
            self.api_key = api_key if api_key is not None else _resolve_api_key(self.model)

        self._providers = _providers  # may be None in single-provider mode
        self._rate_limiter = _SimpleRateLimiter(calls_per_minute)
        self._client = httpx.Client(timeout=120)

        if self.api_key is None or self.api_key == "***":
            warnings.warn(
                "No API key found for optimizer LLM. "
                "Set SKILLOPT_API_KEY, OPENROUTER_API_KEY, or OPENAI_API_KEY."
            )

    @classmethod
    def from_hermes_config(cls, calls_per_minute: int = 30) -> OptimizerLLMClient:
        """Create client with all configured providers for automatic fallback.

        Reads the 3-tier provider chain from .env (SKILLOPT_API_KEY /
        SKILLOPT_API_KEY_2 / SKILLOPT_API_KEY_3).  The ``chat()`` method
        automatically tries each provider in order on failure.

        Returns
        -------
        OptimizerLLMClient
            Client with fallback support.
        """
        providers = _resolve_all_providers()
        if not providers:
            # Fall back to single-provider mode
            info = _resolve_hermes_default_model()
            model = info["model"] or _resolve_model()
            api_base = info["api_base"] or _resolve_api_base(model)
            api_key = info.get("api_key", "")

            no_auth = {"ollama", "local"}
            provider = info.get("provider", "")
            if provider not in no_auth and "localhost" not in api_base:
                if not api_key:
                    api_key = _resolve_api_key(model) or ""

            return cls(
                model=model,
                api_base=api_base,
                api_key=api_key,
                calls_per_minute=calls_per_minute,
            )

        return cls(_providers=providers, calls_per_minute=calls_per_minute)

    # ── Public API ─────────────────────────────────────────────────

    def chat(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        response_format: dict | None = None,
        extra_headers: dict | None = None,
    ) -> str:
        """Single chat completion call with automatic provider fallback.

        Returns the raw text content from the first successful provider.

        Raises
        ------
        RuntimeError
            If all configured providers fail.
        """
        providers = self._providers or [{
            "model": self.model,
            "api_base": self.api_base,
            "api_key": self.api_key,
            "label": "default",
        }]

        last_error: Exception | None = None
        for i, p in enumerate(providers):
            self._rate_limiter.wait()
            try:
                return self._chat_once(p, messages, temperature, max_tokens,
                                       response_format, extra_headers)
            except (httpx.TimeoutException, httpx.RequestError) as e:
                last_error = e
                self._warn_fallback(p, i, len(providers), e)
            except httpx.HTTPStatusError as e:
                last_error = e
                # Non-temporary errors (bad request, auth, not found) → don't fallback
                if e.response.status_code in _FALLBACK_STATUSES:
                    self._warn_fallback(p, i, len(providers), e)
                else:
                    # 400/401/403/404 → re-raise immediately
                    raise

        raise RuntimeError(
            f"All {len(providers)} LLM providers failed. "
            f"Last error: {last_error or 'unknown'}"
        ) from last_error

    def chat_structured(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> list[dict] | dict | None:
        """Chat completion + structured JSON extraction.

        Returns parsed JSON (list or dict) or None on failure.
        """
        text = self.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_headers={"X-Response-Format": "json_object"},
        )
        return _extract_json(text)

    def close(self):
        self._client.close()

    # ── Private helpers ────────────────────────────────────────────

    def _chat_once(
        self,
        provider: dict,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        response_format: dict | None,
        extra_headers: dict | None,
    ) -> str:
        """Make a single chat completion request to *provider*."""
        model = provider["model"]
        api_base = provider["api_base"]
        api_key = provider["api_key"]

        headers = {
            "Content-Type": "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if extra_headers:
            headers.update(extra_headers)

        # OpenRouter-specific headers
        headers.setdefault("HTTP-Referer", "https://github.com/hermes-agent")
        headers.setdefault("X-Title", "SkillOpt Hermes")

        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            body["response_format"] = response_format

        response = self._client.post(
            f"{api_base}/chat/completions",
            headers=headers,
            json=body,
        )

        if response.status_code == 429:
            # Rate limited — wait and retry once
            retry_after = int(response.headers.get("Retry-After", "5"))
            time.sleep(retry_after)
            response = self._client.post(
                f"{api_base}/chat/completions",
                headers=headers,
                json=body,
            )

        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    @staticmethod
    def _warn_fallback(provider: dict, index: int, total: int, exc: Exception):
        """Emit a warning when a provider fails and we're falling through."""
        label = provider.get("label", f"provider-{index}")
        if index < total - 1:
            warnings.warn(
                f"LLM provider '{label}' failed "
                f"({type(exc).__name__}: {exc}), trying next provider..."
            )
        else:
            warnings.warn(
                f"LLM provider '{label}' failed ({type(exc).__name__}: {exc}) — "
                f"no more providers to try."
            )


# ── Prompt templates (loaded from reference files) ────────────────────

_PROMPT_DIR = Path(__file__).resolve().parent.parent / "templates"


def load_prompt(name: str, **kwargs: Any) -> str:
    """Load a prompt template from the templates/ directory.

    Templates are markdown files named <name>.md under templates/.
    Supports ``{key}``-style substitution — only known keys are replaced;
    other curly braces (e.g. JSON examples) are left untouched.
    """
    path = _PROMPT_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    content = path.read_text(encoding="utf-8")
    for key, value in kwargs.items():
        content = content.replace("{" + key + "}", str(value))
    return content


# ── Quick test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    client = OptimizerLLMClient.from_hermes_config()
    print(f"Model : {client.model}")
    print(f"API   : {client.api_base}")
    print(f"Key   : {'***' if client.api_key else 'NONE'}")
    if client._providers:
        print(f"Chain : {', '.join(p.get('label', '?') for p in client._providers)}")
        for p in client._providers:
            print(f"  {p['label']:20s}  {p['model']:35s}  {p['api_base']}")
