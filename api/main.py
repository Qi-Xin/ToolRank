from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from typing import Optional
import os

from .models import (
    Tool, ToolCreate, ToolSearchResult,
    Review, ReviewCreate,
    ExecutionLog, ExecutionLogCreate,
    SearchRequest,
)
from . import database as db
from .embeddings import get_default_provider, EmbeddingProvider
from .ranking import rank_tools, RankingWeights

@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    yield


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


# --- Tools ---

@app.post("/tools", response_model=Tool, status_code=201)
def create_tool(body: ToolCreate):
    existing = db.get_tool_by_name(body.name)
    if existing:
        raise HTTPException(status_code=409, detail=f"Tool '{body.name}' already exists")

    data = body.model_dump()
    embedding = get_provider().embed(body.description)
    data["embedding"] = embedding
    tool = db.insert_tool(data)
    return tool


@app.get("/tools", response_model=list[Tool])
def list_tools(category: Optional[str] = Query(default=None)):
    return db.list_tools(category=category)


@app.get("/tools/{tool_id}", response_model=Tool)
def get_tool(tool_id: str):
    tool = db.get_tool(tool_id)
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")
    return tool


# --- Search ---

@app.post("/search", response_model=list[ToolSearchResult])
def search_tools(body: SearchRequest):
    all_tools = db.list_tools(category=body.category)
    if not all_tools:
        return []

    query_embedding = get_provider().embed(body.query)
    results = rank_tools(all_tools, query_embedding, top_k=body.top_k)
    return results


@app.get("/search", response_model=list[ToolSearchResult])
def search_tools_get(
    q: str = Query(..., description="Task description to find tools for"),
    top_k: int = Query(default=5, ge=1, le=20),
    category: Optional[str] = Query(default=None),
):
    return search_tools(SearchRequest(query=q, top_k=top_k, category=category))


# --- Reviews ---

@app.post("/reviews", response_model=Review, status_code=201)
def submit_review(body: ReviewCreate):
    tool = db.get_tool(body.tool_id)
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")
    review = db.insert_review(body.model_dump())
    return review


@app.get("/tools/{tool_id}/reviews", response_model=list[Review])
def get_tool_reviews(tool_id: str):
    tool = db.get_tool(tool_id)
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")
    return db.get_reviews_for_tool(tool_id)


# --- Execution Logs ---

@app.post("/logs", response_model=ExecutionLog, status_code=201)
def submit_log(body: ExecutionLogCreate):
    tool = db.get_tool(body.tool_id)
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")
    log = db.insert_execution_log(body.model_dump())
    db.update_tool_stats(body.tool_id)
    return log


# --- Health ---

@app.get("/health")
def health():
    return {"status": "ok", "version": "0.1.0"}
