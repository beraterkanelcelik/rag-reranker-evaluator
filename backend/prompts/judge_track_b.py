from __future__ import annotations

TRACK_B_SYSTEM_PROMPT = """You are a strict evaluator of groundedness (faithfulness) for RAG systems.
Your job is to verify that EVERY claim in the MODEL_ANSWER is supported by the provided CONTEXTS.

Critical Rules:
- A claim is SUPPORTED only if the context explicitly states it or directly implies it
- A claim is UNSUPPORTED if it requires outside knowledge not in the contexts
- A claim is a HALLUCINATION if it contradicts the contexts
- Ignore formatting and style - focus only on factual claims

You MUST return valid JSON only. No markdown, no explanation outside JSON."""

TRACK_B_USER_PROMPT = """QUESTION:
{question}

CONTEXTS:
{numbered_contexts}

MODEL_ANSWER:
{model_answer}

TASK:
Evaluate the MODEL_ANSWER for groundedness in the provided CONTEXTS.

Score from 0 to 5 on each dimension:

1. context_support (0-5): Are the claims in the answer supported by the contexts?
   - 5: Every claim is directly supported
   - 3: Most claims supported, some require inference
   - 1: Few claims supported
   - 0: No claims supported

2. hallucination (0-5): How free is the answer from unsupported/contradicted claims?
   - 5: No hallucinations whatsoever
   - 3: Minor unsupported details that don't affect accuracy
   - 1: Significant unsupported claims
   - 0: Major hallucinations or contradictions

3. citation_quality (0-5): Could the claims be traced back to specific contexts?
   - 5: Every claim clearly attributable to specific context
   - 3: Most claims attributable
   - 1: Vague, hard to trace
   - 0: Cannot determine source of claims

Calculate overall_groundedness as average of the three scores.

Identify up to 3 specific unsupported claims (short phrases only).

Return JSON with this exact structure:
{
    "context_support": <int 0-5>,
    "hallucination": <int 0-5>,
    "citation_quality": <int 0-5>,
    "overall_groundedness": <float 0-5>,
    "unsupported_claims": ["<claim 1>", "<claim 2>", ...],
    "short_reason": "<string, max 40 words>"
}"""
