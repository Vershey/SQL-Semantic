from __future__ import annotations


class AnthropicLLM:
    """Real LanguageModel adapter. Set USE_REAL_ANTHROPIC=1 and ANTHROPIC_API_KEY
    in the environment, and `pip install anthropic`, to use this instead of FakeLLM."""

    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str | None = None) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install the SDK first: pip install anthropic") from exc
        self._client = anthropic.AsyncAnthropic(api_key=api_key)  # reads ANTHROPIC_API_KEY
        self._model = model

    async def complete(
        self, prompt: str, *, system: str | None = None, max_tokens: int = 1024
    ) -> str:
        message = await self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system or "",
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in message.content if block.type == "text")
