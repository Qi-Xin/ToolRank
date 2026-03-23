import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Optional
import os
import pathlib

from .models import (
    Tool, ToolCreate, ToolSearchResult,
    Review, ReviewCreate,
    ExecutionLog, ExecutionLogCreate,
    SearchRequest,
)
from . import database as db
from .embeddings import get_default_provider, EmbeddingProvider
from .ranking import rank_tools
from .evaluator import run_evaluation_batch

BASE_DIR = pathlib.Path(__file__).parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

EVAL_INTERVAL = int(os.environ.get("EVAL_INTERVAL_SECONDS", "300"))  # 5 min default


async def _evaluator_loop():
    """Background task: run LLM-as-Judge every EVAL_INTERVAL seconds."""
    while True:
        await asyncio.sleep(EVAL_INTERVAL)
        try:
            result = await run_evaluation_batch(limit=20)
            if result["evaluated"] > 0:
                import logging
                logging.getLogger(__name__).info(f"Evaluator: {result}")
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Evaluator error: {e}")


def _reembed_all():
    """Re-compute embeddings for all tools using the current provider."""
    provider = get_provider()
    tools = db.list_tools()
    # Fit TF-IDF on the full corpus before embedding
    from .embeddings import TFIDFEmbeddingProvider
    if isinstance(provider, TFIDFEmbeddingProvider):
        provider.fit([t["description"] for t in tools])
    count = 0
    for tool in tools:
        embedding = provider.embed(tool["description"])
        db.update_tool_embedding(tool["id"], embedding)
        count += 1
    return count


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    # Re-embed all tools on startup so embeddings match the active provider
    await asyncio.get_event_loop().run_in_executor(None, _reembed_all)
    task = asyncio.create_task(_evaluator_loop())
    yield
    task.cancel()


app = FastAPI(
    title="ToolRank API",
    description="SkillRank — discover, compare, and review AI agent tools",
    version="0.1.0",
    lifespan=lifespan,
)

_provider: Optional[EmbeddingProvider] = None


def get_provider() -> EmbeddingProvider:
    global _provider
    if _provider is None:
        _provider = get_default_provider()
    return _provider


# ── Web UI ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def home(request: Request, q: Optional[str] = None):
    results = []
    if q:
        all_tools = db.list_tools()
        query_embedding = get_provider().embed(q)
        results = rank_tools(all_tools, query_embedding, top_k=10)
    else:
        results = db.list_tools()
        results.sort(key=lambda t: t.get("avg_success_rate", 0), reverse=True)
        results = results[:10]
    return templates.TemplateResponse("index.html", {
        "request": request, "query": q or "", "results": results,
    })


@app.get("/tools/{tool_id}/page", response_class=HTMLResponse)
def tool_page(request: Request, tool_id: str):
    tool = db.get_tool(tool_id)
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")
    reviews = db.get_reviews_for_tool(tool_id)
    return templates.TemplateResponse("tool.html", {
        "request": request, "tool": tool, "reviews": reviews,
    })


# ── Tools API ─────────────────────────────────────────────────────────────────

@app.post("/tools", response_model=Tool, status_code=201)
def create_tool(body: ToolCreate):
    existing = db.get_tool_by_name(body.name)
    if existing:
        raise HTTPException(status_code=409, detail=f"Tool '{body.name}' already exists")
    data = body.model_dump()
    data["embedding"] = get_provider().embed(body.description)
    return db.insert_tool(data)


@app.get("/tools", response_model=list[Tool])
def list_tools(category: Optional[str] = Query(default=None)):
    return db.list_tools(category=category)


@app.get("/tools/{tool_id}", response_model=Tool)
def get_tool(tool_id: str):
    tool = db.get_tool(tool_id)
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")
    return tool


# ── Search API ────────────────────────────────────────────────────────────────

@app.post("/search", response_model=list[ToolSearchResult])
def search_tools(body: SearchRequest):
    all_tools = db.list_tools(category=body.category)
    if not all_tools:
        return []
    query_embedding = get_provider().embed(body.query)
    return rank_tools(all_tools, query_embedding, top_k=body.top_k)


@app.get("/search", response_model=list[ToolSearchResult])
def search_tools_get(
    q: str = Query(...),
    top_k: int = Query(default=5, ge=1, le=20),
    category: Optional[str] = Query(default=None),
):
    return search_tools(SearchRequest(query=q, top_k=top_k, category=category))


# ── Reviews API ───────────────────────────────────────────────────────────────

@app.post("/reviews", response_model=Review, status_code=201)
def submit_review(body: ReviewCreate):
    if not db.get_tool(body.tool_id):
        raise HTTPException(status_code=404, detail="Tool not found")
    return db.insert_review(body.model_dump())


@app.get("/tools/{tool_id}/reviews", response_model=list[Review])
def get_tool_reviews(tool_id: str):
    if not db.get_tool(tool_id):
        raise HTTPException(status_code=404, detail="Tool not found")
    return db.get_reviews_for_tool(tool_id)


# ── Execution Logs API ────────────────────────────────────────────────────────

@app.post("/logs", response_model=ExecutionLog, status_code=201)
def submit_log(body: ExecutionLogCreate):
    if not db.get_tool(body.tool_id):
        raise HTTPException(status_code=404, detail="Tool not found")
    log = db.insert_execution_log(body.model_dump())
    db.update_tool_stats(body.tool_id)
    return log


# ── Admin API ─────────────────────────────────────────────────────────────────

@app.post("/admin/evaluate")
async def trigger_evaluation(limit: int = Query(default=20, ge=1, le=100)):
    """Manually trigger LLM-as-Judge evaluation batch."""
    return await run_evaluation_batch(limit=limit)


@app.post("/admin/reembed")
async def trigger_reembed():
    """Re-compute embeddings for all tools using the current provider."""
    count = await asyncio.get_event_loop().run_in_executor(None, _reembed_all)
    return {"reembedded": count}


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "0.1.0"}
