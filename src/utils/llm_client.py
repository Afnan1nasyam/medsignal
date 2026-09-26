"""Groq LLM client with retry, rate limiting, and JSON parsing.

Wraps the Groq chat-completions API with:
- tenacity exponential-backoff retry on rate-limit (HTTP 429) errors,
- request-rate limiting derived from ``settings.GROQ_RPM``,
- automatic fallback from the primary model to ``settings.FALLBACK_MODEL``,
- a ``generate_json`` helper that parses JSON and repairs malformed output once.

The heavy ``groq`` import is deferred to client construction so importing this
module performs no network activity (this dev machine's proxy blocks Groq).
"""

import json
import time
from typing import Any

from loguru import logger
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from src.config import settings

# Prompt used to ask the model to repair invalid JSON output.
JSON_REPAIR_PROMPT = (  # V1
    "Your previous response was not valid JSON. Return ONLY the corrected, "
    "strictly-valid JSON object with no markdown fences or commentary.\n\n"
    "Invalid response:\n{bad_output}"
)


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Return True if an exception looks like a Groq rate-limit (429) error."""
    if getattr(exc, "status_code", None) == 429:
        return True
    name = type(exc).__name__.lower()
    return "ratelimit" in name or "429" in str(exc)


def _strip_json_fences(raw: str) -> str:
    """Remove surrounding markdown code fences from a model's JSON response."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[: -3]
    return text.strip().removeprefix("json").strip()


class GroqClient:
    """Thin, resilient wrapper around the Groq chat-completions API."""

    def __init__(self, model_name: str | None = None) -> None:
        """Initialize the client.

        Args:
            model_name: Model to use as primary. Defaults to
                ``settings.PRIMARY_MODEL``.
        """
        import groq  # deferred: importing this module must not touch the network

        self.model_name = model_name or settings.PRIMARY_MODEL
        self.fallback_model = settings.FALLBACK_MODEL
        self._client = groq.Groq(api_key=settings.GROQ_API_KEY)
        self._min_interval = 60.0 / max(settings.GROQ_RPM, 1)
        self._last_request_time = 0.0

    # -- internals ---------------------------------------------------------- #
    def _throttle(self) -> None:
        """Sleep as needed to respect the configured requests-per-minute cap."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.monotonic()

    @retry(
        retry=retry_if_exception(_is_rate_limit_error),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(settings.GROQ_RETRY_ATTEMPTS),
        reraise=True,
    )
    def _complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Call the Groq API once (with rate limiting); retried on 429."""
        self._throttle()
        start = time.monotonic()
        response = self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        latency = time.monotonic() - start
        usage = getattr(response, "usage", None)
        total_tokens = getattr(usage, "total_tokens", "?") if usage else "?"
        logger.info(
            "Groq call model={} tokens={} latency={:.2f}s",
            model,
            total_tokens,
            latency,
        )
        return response.choices[0].message.content or ""

    # -- public API --------------------------------------------------------- #
    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> str:
        """Generate a text completion, falling back to the secondary model.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            The generated text.

        Raises:
            Exception: Re-raised if both primary and fallback models fail.
        """
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            return self._complete(self.model_name, messages, temperature, max_tokens)
        except Exception as primary_error:  # noqa: BLE001 - log and fall back
            logger.warning(
                "Primary model {} failed ({}); trying fallback {}",
                self.model_name,
                primary_error,
                self.fallback_model,
            )
            try:
                return self._complete(
                    self.fallback_model, messages, temperature, max_tokens
                )
            except Exception as fallback_error:  # noqa: BLE001
                logger.error(
                    "Fallback model {} also failed: {}",
                    self.fallback_model,
                    fallback_error,
                )
                raise

    def generate_json(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Generate a completion and parse it as a JSON object.

        On a JSON parse failure, retries once with a repair prompt that feeds
        the malformed output back to the model.

        Args:
            prompt: The user prompt (should instruct the model to return JSON).
            system_prompt: Optional system prompt.
            temperature: Sampling temperature (defaults to deterministic 0.0).

        Returns:
            The parsed JSON object as a dict.

        Raises:
            json.JSONDecodeError: If output cannot be parsed even after repair.
        """
        raw = self.generate(prompt, system_prompt, temperature=temperature)
        try:
            return json.loads(_strip_json_fences(raw))
        except json.JSONDecodeError:
            logger.warning("JSON parse failed; attempting one repair pass")
            repaired = self.generate(
                JSON_REPAIR_PROMPT.format(bad_output=raw),
                system_prompt,
                temperature=0.0,
            )
            return json.loads(_strip_json_fences(repaired))
