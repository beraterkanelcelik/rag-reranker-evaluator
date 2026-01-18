from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

from openai import OpenAI

from backend.prompts.generation import GENERATION_SYSTEM_PROMPT, GENERATION_USER_PROMPT


@dataclass
class GenerationResult:
    answer: str
    input_tokens: int
    output_tokens: int
    total_tokens: int


class GenerationService:
    def __init__(self, client: OpenAI | None = None) -> None:
        self._client = client

    def generate_answer(
        self,
        question: str,
        contexts: Iterable[str],
        model_name: str,
        api_key: str,
        temperature: float = 0.0,
    ) -> GenerationResult:
        if not api_key:
            raise ValueError("OpenAI API key is required")
        client = self._client or OpenAI(api_key=api_key)
        context_text = self._join_contexts(contexts)
        user_prompt = GENERATION_USER_PROMPT.format(context=context_text, question=question)
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
        answer = response.choices[0].message.content or ""
        input_tokens, output_tokens, total_tokens = self._extract_usage(response)
        return GenerationResult(
            answer=answer,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )

    def _join_contexts(self, contexts: Iterable[str]) -> str:
        return "\n\n".join(text for text in contexts if text)

    def _extract_usage(self, response: object) -> Tuple[int, int, int]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return 0, 0, 0
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
        return prompt_tokens, completion_tokens, total_tokens
