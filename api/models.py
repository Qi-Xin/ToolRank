from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
import uuid


def new_id() -> str:
    return str(uuid.uuid4())


class ToolCreate(BaseModel):
    name: str
    description: str
    category: Optional[str] = None
    install_cmd: Optional[str] = None
    url: Optional[str] = None
    api_schema: Optional[dict] = None


class Tool(BaseModel):
    id: str
    name: str
    description: str
    category: Optional[str] = None
    install_cmd: Optional[str] = None
    url: Optional[str] = None
    api_schema: Optional[dict] = None
    avg_success_rate: float = 0.0
    avg_latency_ms: float = 0.0
    avg_token_cost: float = 0.0
    avg_llm_score: float = 0.0
    total_calls: int = 0
    created_at: str = ""

    model_config = {"from_attributes": True}


class ToolSearchResult(Tool):
    rank_score: float = 0.0
    rank_breakdown: Optional[dict] = None


class ReviewCreate(BaseModel):
    tool_id: str
    rating: int = Field(..., ge=1, le=5)
    success: bool
    latency_ms: Optional[float] = None
    token_cost: Optional[int] = None
    use_case: Optional[str] = None
    pros: Optional[str] = None
    cons: Optional[str] = None
    agent_model: Optional[str] = None
    task_description: Optional[str] = None


class Review(ReviewCreate):
    id: str
    llm_score: Optional[float] = None
    llm_reason: Optional[str] = None
    created_at: str = ""


class ExecutionLogCreate(BaseModel):
    tool_id: str
    success: bool
    latency_ms: float
    token_input: Optional[int] = None
    token_output: Optional[int] = None
    error_message: Optional[str] = None
    input_query: Optional[str] = None
    output_summary: Optional[str] = None
    agent_model: Optional[str] = None


class ExecutionLog(ExecutionLogCreate):
    id: str
    evaluated: bool = False
    created_at: str = ""


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    category: Optional[str] = None
