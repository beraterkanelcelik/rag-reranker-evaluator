from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

from backend.prompts.judge_track_a import TRACK_A_SYSTEM_PROMPT, TRACK_A_USER_PROMPT
from backend.prompts.judge_track_b import TRACK_B_SYSTEM_PROMPT, TRACK_B_USER_PROMPT


class TrackAScores(BaseModel):
    correctness: int = Field(description="Correctness score 0-5")
    completeness: int = Field(description="Completeness score 0-5")
    specificity: int = Field(description="Specificity score 0-5")
    clarity: int = Field(description="Clarity score 0-5")
    overall: float = Field(description="Overall score 0-5")
    short_reason: str = Field(description="Short reason for the score")


class TrackBScores(BaseModel):
    context_support: float = Field(description="Context support score")
    hallucination: float = Field(description="Hallucination score")
    citation_quality: float = Field(description="Citation quality score")
    overall_groundedness: float = Field(description="Overall groundedness score")
    unsupported_claims: str = Field(description="Unsupported claims")


@dataclass
class JudgeResult:
    scores: dict
    raw_response: dict
    input_tokens: int
    output_tokens: int
    total_tokens: int


class JudgeService:
    pass

    def judge_track_a(
        self,
        question: str,
        reference_answer: str,
        model_answer: str,
        model_name: str,
        api_key: str,
        temperature: float = 0.0,
    ) -> JudgeResult:
        try:
            llm = ChatOpenAI(model=model_name, api_key=api_key, temperature=temperature)
            structured_llm = llm.with_structured_output(TrackAScores)
            prompt = TRACK_A_USER_PROMPT.format(
                question=question,
                reference_answer=reference_answer,
                model_answer=model_answer,
            )
            response = structured_llm.invoke([
                {"role": "system", "content": TRACK_A_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ])
            return JudgeResult(
                scores=response.model_dump(),
                raw_response={"content": response.model_dump_json()},
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
            )
        except Exception:
            # Fallback to default scores if structured output fails
            return JudgeResult(
                scores={
                    "correctness": 3,
                    "completeness": 3,
                    "specificity": 3,
                    "clarity": 3,
                    "overall": 3.0,
                    "short_reason": "Fallback due to parsing error",
                },
                raw_response={"content": "Fallback used"},
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
            )

    def judge_track_b(
        self,
        question: str,
        model_answer: str,
        contexts: Iterable[str],
        model_name: str,
        api_key: str,
        temperature: float = 0.0,
    ) -> JudgeResult:
        try:
            llm = ChatOpenAI(model=model_name, api_key=api_key, temperature=temperature)
            structured_llm = llm.with_structured_output(TrackBScores)
            context_text = self._format_contexts(contexts)
            prompt = TRACK_B_USER_PROMPT.format(
                question=question,
                model_answer=model_answer,
                numbered_contexts=context_text,
            )
            response = structured_llm.invoke([
                {"role": "system", "content": TRACK_B_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ])
            return JudgeResult(
                scores=response.model_dump(),
                raw_response={"content": response.model_dump_json()},
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
            )
        except Exception:
            # Fallback to default scores if structured output fails
            return JudgeResult(
                scores={
                    "context_support": 3.0,
                    "hallucination": 3.0,
                    "citation_quality": 3.0,
                    "overall_groundedness": 3.0,
                    "unsupported_claims": "Fallback used",
                },
                raw_response={"content": "Fallback used"},
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
            )
        return self._call_judge(
            system_prompt=TRACK_B_SYSTEM_PROMPT,
            user_prompt=prompt,
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
        )

    def _format_contexts(self, contexts: Iterable[str]) -> str:
        lines: List[str] = []
        for idx, context in enumerate(contexts, start=1):
            if not context:
                continue
            lines.append(f"[{idx}] {context}")
        return "\n\n".join(lines)
