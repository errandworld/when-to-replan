"""Chat models, reached through LiteLLM."""
from __future__ import annotations

import json
import time
from typing import Any

from app.types import LLMResult, ToolCall


class LiteLLM:
    """One model, called with a fixed temperature. Model names are LiteLLM's, for example
    "gemini/gemini-3.5-flash-lite" or "ollama/qwen3.8:27b"."""

    def __init__(self, model: str, temperature: float = 0.0):
        self.model = model
        self.temperature = temperature
        self._takes_temperature: bool | None = None

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        response_format: Any | None = None,
    ) -> LLMResult:
        import litellm

        kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
        if self.takes_temperature:
            kwargs["temperature"] = self.temperature
        if tools:
            kwargs["tools"] = tools
        if response_format is not None:
            kwargs["response_format"] = response_format

        start = time.perf_counter()
        response = litellm.completion(**kwargs)
        return _to_result(response, time.perf_counter() - start)


    @property
    def takes_temperature(self) -> bool:
        """Whether this model accepts a temperature. A few, such as gemini-3.5-flash-lite,
        do not, and sending one to them is an error."""
        if self._takes_temperature is None:
            import litellm

            provider, _, name = self.model.partition("/")
            try:
                supported = litellm.get_supported_openai_params(
                    model=name or provider,
                    custom_llm_provider=provider if name else None,
                )
            except Exception:
                supported = None
            self._takes_temperature = supported is None or "temperature" in supported
        return self._takes_temperature


def _to_result(response: Any, latency_s: float) -> LLMResult:
    message = response.choices[0].message
    usage = getattr(response, "usage", None)

    prompt_tokens = _int(getattr(usage, "prompt_tokens", 0))
    details = getattr(usage, "prompt_tokens_details", None)
    cached = _int(getattr(details, "cached_tokens", 0)) if details else 0

    return LLMResult(
        text=getattr(message, "content", "") or "",
        tool_calls=[_to_call(c) for c in (getattr(message, "tool_calls", None) or [])],
        input_tokens=max(prompt_tokens - cached, 0),
        cached_input_tokens=cached,
        output_tokens=_int(getattr(usage, "completion_tokens", 0)),
        latency_s=latency_s,
        model_version=getattr(response, "model", "") or "",
    )


def _to_call(call: Any) -> ToolCall:
    """Raises ValueError if the model sent arguments that are not JSON."""
    arguments = call.function.arguments
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError as exc:
            raise ValueError(f"{call.function.name} got unparseable arguments: {arguments!r}") from exc
    return ToolCall(name=call.function.name, args=arguments or {}, id=getattr(call, "id", "") or "")


def _int(value: Any) -> int:
    return int(value) if value else 0


if __name__ == "__main__":
    import sys

    for name in sys.argv[1:]:
        model = LiteLLM(name)
        result = model.complete([{"role": "user", "content": "Reply with one word: ok"}])
        print(
            f"{name}\n"
            f"  version  {result.model_version}\n"
            f"  temp     {'0' if model.takes_temperature else 'not supported by this model'}\n"
            f"  tokens   {result.input_tokens} in ({result.cached_input_tokens} cached), "
            f"{result.output_tokens} out\n"
            f"  latency  {result.latency_s:.2f}s\n"
            f"  reply    {result.text.strip()[:60]!r}"
        )
