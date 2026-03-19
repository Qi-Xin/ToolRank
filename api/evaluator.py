"""
LLM-as-Judge evaluator pipeline.

Fetches unevaluated execution logs, asks an LLM to score the tool call,
and stores the result back as a review with an llm_score.

Supports both Anthropic (Claude) and OpenAI. Falls back gracefully if
no API key is set.
"""

import json
import os
import asyncio
import logging
from typing import Optional

from . import database as db

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """You are evaluating an AI agent tool execution. Your job is to assess how useful the tool was.

Tool name: {tool_name}
Task the agent was trying to accomplish: {input_query}
Tool output summary: {output_summary}
Execution succeeded: {success}
Latency: {latency_ms:.0f}ms

Rate this tool execution on a scale of 1-5:
1 = Tool failed completely or output was useless
2 = Tool barely helped, required significant workaround
3 = Tool worked but output needed post-processing
4 = Tool worked well with only minor issues
5 = Tool perfectly solved the task with clean, usable output

Respond with JSON only, no other text:
{{"score": <integer 1-5>, "reason": "<one sentence explanation>"}}"""


async def evaluate_log(log: dict, tool: dict) -> Optional[tuple[float, str]]:
    """Score a single execution log. Returns (score, reason) or None."""
    prompt = JUDGE_PROMPT.format(
        tool_name=tool["name"],
        input_query=log.get("input_query") or "unknown task",
        output_summary=log.get("output_summary") or "no output recorded",
        success="yes" if log["success"] else "no",
        latency_ms=log.get("latency_ms") or 0,
    )

    # Try Anthropic first
    if os.environ.get("ANTHROPIC_API_KEY"):
        return await _judge_anthropic(prompt)

    # Fallback to OpenAI
    if os.environ.get("OPENAI_API_KEY"):
        return await _judge_openai(prompt)

    logger.warning("No LLM API key set. Set ANTHROPIC_API_KEY or OPENAI_API_KEY.")
    return None


async def _judge_anthropic(prompt: str) -> Optional[tuple[float, str]]:
    try:
        import anthropic
        client = anthropic.AsyncAnthropic()
        message = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()
        data = json.loads(raw)
        return float(data["score"]), str(data["reason"])
    except Exception as e:
        logger.error(f"Anthropic judge failed: {e}")
        return None


async def _judge_openai(prompt: str) -> Optional[tuple[float, str]]:
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI()
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        return float(data["score"]), str(data["reason"])
    except Exception as e:
        logger.error(f"OpenAI judge failed: {e}")
        return None


async def run_evaluation_batch(limit: int = 20) -> dict:
    """
    Fetch up to `limit` unevaluated logs, score each one, and store results.
    Returns a summary dict.
    """
    logs = db.get_unevaluated_logs(limit=limit)
    if not logs:
        return {"evaluated": 0, "skipped": 0, "errors": 0}

    evaluated = skipped = errors = 0

    for log in logs:
        tool = db.get_tool(log["tool_id"])
        if not tool:
            db.mark_log_evaluated(log["id"])
            skipped += 1
            continue

        result = await evaluate_log(log, tool)
        db.mark_log_evaluated(log["id"])

        if result is None:
            skipped += 1
            continue

        score, reason = result

        # Store as a review with the LLM score
        review_data = {
            "tool_id": log["tool_id"],
            "rating": max(1, min(5, round(score))),
            "success": bool(log["success"]),
            "latency_ms": log.get("latency_ms"),
            "token_cost": (log.get("token_input") or 0) + (log.get("token_output") or 0),
            "agent_model": log.get("agent_model"),
            "task_description": log.get("input_query"),
            "use_case": "auto-evaluated",
        }
        review = db.insert_review(review_data)

        # Attach the LLM score directly to the review
        with db.get_conn() as conn:
            conn.execute(
                "UPDATE reviews SET llm_score = ?, llm_reason = ? WHERE id = ?",
                (score, reason, review["id"]),
            )

        db.update_tool_stats(log["tool_id"])
        evaluated += 1

    return {"evaluated": evaluated, "skipped": skipped, "errors": errors}
