"""
ToolRank Tracker SDK
A lightweight decorator for implicit tool call tracking.

Usage:
    from toolrank import track_tool

    @track_tool("brave-search", api_url="http://localhost:8000")
    def search_web(query: str) -> str:
        # your tool call here
        return results

    # Async version
    @track_tool("brave-search")
    async def search_web_async(query: str) -> str:
        return results
"""

import time
import asyncio
import functools
import threading
from typing import Optional, Callable
import httpx


def track_tool(
    tool_name: str,
    api_url: str = "http://localhost:8000",
    agent_model: Optional[str] = None,
):
    """
    Decorator that wraps a tool call and automatically logs execution data to ToolRank.
    Works with both sync and async functions.

    Args:
        tool_name: Name of the tool as registered in ToolRank
        api_url: ToolRank API base URL
        agent_model: The LLM model making the call (e.g. "claude-sonnet-4-6")
    """
    def decorator(fn: Callable) -> Callable:
        if asyncio.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs):
                start = time.perf_counter()
                error = None
                result = None
                try:
                    result = await fn(*args, **kwargs)
                    return result
                except Exception as e:
                    error = str(e)
                    raise
                finally:
                    latency_ms = (time.perf_counter() - start) * 1000
                    _log_async(
                        api_url, tool_name, error is None,
                        latency_ms, error, agent_model,
                        _estimate_tokens(args, kwargs, result),
                    )
            return async_wrapper
        else:
            @functools.wraps(fn)
            def sync_wrapper(*args, **kwargs):
                start = time.perf_counter()
                error = None
                result = None
                try:
                    result = fn(*args, **kwargs)
                    return result
                except Exception as e:
                    error = str(e)
                    raise
                finally:
                    latency_ms = (time.perf_counter() - start) * 1000
                    _log_background(
                        api_url, tool_name, error is None,
                        latency_ms, error, agent_model,
                        _estimate_tokens(args, kwargs, result),
                    )
            return sync_wrapper
    return decorator


def _estimate_tokens(args, kwargs, result) -> tuple[int, int]:
    """Rough token estimate: 1 token ≈ 4 chars."""
    input_text = str(args) + str(kwargs)
    output_text = str(result) if result is not None else ""
    return len(input_text) // 4, len(output_text) // 4


def _build_payload(
    tool_name: str,
    success: bool,
    latency_ms: float,
    error: Optional[str],
    agent_model: Optional[str],
    tokens: tuple[int, int],
) -> dict:
    return {
        "tool_name": tool_name,
        "success": success,
        "latency_ms": round(latency_ms, 2),
        "error_message": error,
        "agent_model": agent_model,
        "token_input": tokens[0],
        "token_output": tokens[1],
    }


def _log_background(api_url, tool_name, success, latency_ms, error, agent_model, tokens):
    """Fire-and-forget log in a daemon thread."""
    payload = _build_payload(tool_name, success, latency_ms, error, agent_model, tokens)
    t = threading.Thread(
        target=_post_log,
        args=(api_url, tool_name, payload),
        daemon=True,
    )
    t.start()


def _log_async(api_url, tool_name, success, latency_ms, error, agent_model, tokens):
    """Schedule async log without blocking the caller."""
    payload = _build_payload(tool_name, success, latency_ms, error, agent_model, tokens)
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(_post_log_async(api_url, tool_name, payload))
        else:
            loop.run_until_complete(_post_log_async(api_url, tool_name, payload))
    except Exception:
        pass


def _post_log(api_url: str, tool_name: str, payload: dict):
    """Resolve tool ID then post execution log (sync)."""
    try:
        with httpx.Client(base_url=api_url, timeout=5.0) as client:
            resp = client.get("/search", params={"q": tool_name, "top_k": 1})
            if resp.status_code != 200 or not resp.json():
                return
            tool_id = resp.json()[0]["id"]
            client.post("/logs", json={**payload, "tool_id": tool_id})
    except Exception:
        pass  # Never crash the caller


async def _post_log_async(api_url: str, tool_name: str, payload: dict):
    """Resolve tool ID then post execution log (async)."""
    try:
        async with httpx.AsyncClient(base_url=api_url, timeout=5.0) as client:
            resp = await client.get("/search", params={"q": tool_name, "top_k": 1})
            if resp.status_code != 200 or not resp.json():
                return
            tool_id = resp.json()[0]["id"]
            await client.post("/logs", json={**payload, "tool_id": tool_id})
    except Exception:
        pass
