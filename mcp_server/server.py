"""
SkillRank MCP Server
Exposes three tools to any MCP-compatible agent:
  - find_tool(task): find best tools for a task
  - get_reviews(tool_name): get reviews for a specific tool
  - submit_review(...): submit a structured review
"""

import asyncio
import os
import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

API_BASE = os.environ.get("SKILLRANK_API_URL", "http://localhost:8000")

server = Server("skillrank")


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=API_BASE, timeout=10.0)


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="find_tool",
            description=(
                "Find the best AI agent tools (MCPs, Skills) for a given task. "
                "Returns a ranked list with scores, pros, cons, and install instructions."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Describe what you need to accomplish",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default 5)",
                        "default": 5,
                    },
                    "category": {
                        "type": "string",
                        "description": "Optional category filter (e.g. 'search', 'code', 'data')",
                    },
                },
                "required": ["task"],
            },
        ),
        types.Tool(
            name="get_reviews",
            description="Get detailed reviews and stats for a specific tool by name.",
            inputSchema={
                "type": "object",
                "properties": {
                    "tool_name": {
                        "type": "string",
                        "description": "Exact name of the tool",
                    }
                },
                "required": ["tool_name"],
            },
        ),
        types.Tool(
            name="submit_review",
            description=(
                "Submit a structured review after using a tool. "
                "Helps the community by contributing real usage data."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "tool_name": {"type": "string"},
                    "rating": {
                        "type": "integer",
                        "description": "1-5 star rating",
                        "minimum": 1,
                        "maximum": 5,
                    },
                    "success": {
                        "type": "boolean",
                        "description": "Did the tool complete the task successfully?",
                    },
                    "use_case": {"type": "string", "description": "What task did you use it for?"},
                    "pros": {"type": "string"},
                    "cons": {"type": "string"},
                    "latency_ms": {"type": "number"},
                    "token_cost": {"type": "integer"},
                    "agent_model": {"type": "string", "description": "e.g. claude-sonnet-4-6"},
                },
                "required": ["tool_name", "rating", "success"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    async with _client() as client:
        if name == "find_tool":
            return await _find_tool(client, arguments)
        elif name == "get_reviews":
            return await _get_reviews(client, arguments)
        elif name == "submit_review":
            return await _submit_review(client, arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")


async def _find_tool(client: httpx.AsyncClient, args: dict) -> list[types.TextContent]:
    params = {"q": args["task"], "top_k": args.get("top_k", 5)}
    if args.get("category"):
        params["category"] = args["category"]

    resp = await client.get("/search", params=params)
    resp.raise_for_status()
    results = resp.json()

    if not results:
        return [types.TextContent(type="text", text="No tools found for this task.")]

    lines = [f"Top {len(results)} tools for: \"{args['task']}\"\n"]
    for i, tool in enumerate(results, 1):
        lines.append(f"{i}. **{tool['name']}** (score: {tool['rank_score']:.2f})")
        lines.append(f"   {tool['description']}")
        if tool.get("install_cmd"):
            lines.append(f"   Install: `{tool['install_cmd']}`")
        if tool.get("url"):
            lines.append(f"   URL: {tool['url']}")
        stats = (
            f"   Stats: {tool['total_calls']} runs | "
            f"{tool['avg_success_rate']*100:.0f}% success | "
            f"{tool['avg_latency_ms']:.0f}ms avg latency"
        )
        lines.append(stats)
        lines.append("")

    return [types.TextContent(type="text", text="\n".join(lines))]


async def _get_reviews(client: httpx.AsyncClient, args: dict) -> list[types.TextContent]:
    # First find the tool by name via search
    resp = await client.get("/search", params={"q": args["tool_name"], "top_k": 1})
    resp.raise_for_status()
    results = resp.json()

    if not results:
        return [types.TextContent(type="text", text=f"Tool '{args['tool_name']}' not found.")]

    tool = results[0]
    tool_id = tool["id"]

    reviews_resp = await client.get(f"/tools/{tool_id}/reviews")
    reviews_resp.raise_for_status()
    reviews = reviews_resp.json()

    lines = [f"Reviews for **{tool['name']}**"]
    lines.append(f"Avg success: {tool['avg_success_rate']*100:.0f}% | "
                 f"Avg latency: {tool['avg_latency_ms']:.0f}ms | "
                 f"Total runs: {tool['total_calls']}\n")

    if not reviews:
        lines.append("No reviews yet. Be the first to review this tool!")
    else:
        for r in reviews[:10]:
            stars = "★" * r["rating"] + "☆" * (5 - r["rating"])
            lines.append(f"{stars} {'✓' if r['success'] else '✗'} — {r.get('use_case', 'general use')}")
            if r.get("pros"):
                lines.append(f"  + {r['pros']}")
            if r.get("cons"):
                lines.append(f"  - {r['cons']}")

    return [types.TextContent(type="text", text="\n".join(lines))]


async def _submit_review(client: httpx.AsyncClient, args: dict) -> list[types.TextContent]:
    # Resolve tool name to ID
    search_resp = await client.get("/search", params={"q": args["tool_name"], "top_k": 1})
    search_resp.raise_for_status()
    results = search_resp.json()

    if not results:
        return [types.TextContent(type="text", text=f"Tool '{args['tool_name']}' not found.")]

    tool_id = results[0]["id"]

    payload = {
        "tool_id": tool_id,
        "rating": args["rating"],
        "success": args["success"],
        "use_case": args.get("use_case"),
        "pros": args.get("pros"),
        "cons": args.get("cons"),
        "latency_ms": args.get("latency_ms"),
        "token_cost": args.get("token_cost"),
        "agent_model": args.get("agent_model"),
    }

    resp = await client.post("/reviews", json=payload)
    resp.raise_for_status()

    return [types.TextContent(
        type="text",
        text=f"Review submitted for **{args['tool_name']}**. Thank you for contributing to SkillRank!",
    )]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
