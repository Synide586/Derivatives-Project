from __future__ import annotations
import os
from typing import Optional
from anthropic import Anthropic

class AnthropicLLM:
    def __init__(
        self,
        model: str = "claude-haiku-4-5",
        max_output_tokens: int = 16000,
        temperature: float = 0.2,
    ) -> None:
        self.client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature

    def generate(
        self,
        system: str,
        user: str,
        max_output_tokens: Optional[int] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        *,
        auto_tokens: bool = False,
        tokens_per_doc: int = 700,
        base_tokens: int = 800,
    ) -> str:
        m = model or self.model
        temp = temperature if temperature is not None else self.temperature

        if auto_tokens:
            # estimate doc count from your serialized retrieved_docs
            # (cheap heuristic; replace with actual len(retrieved_docs) if you have it in caller)
            doc_count = user.count('"doc_id"')
            mot = base_tokens + tokens_per_doc * max(1, doc_count)
        else:
            mot = max_output_tokens if max_output_tokens is not None else self.max_output_tokens

        msg = self.client.messages.create(
            model=m,
            max_tokens=int(mot),
            temperature=temp,
            system=system,
            messages=[{"role": "user", "content": user}],
        )

        parts = []
        for block in msg.content:
            if hasattr(block, "text") and block.text:
                parts.append(block.text)
        return "".join(parts)
