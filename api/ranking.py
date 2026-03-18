"""
SkillRank: weighted ranking algorithm for agent tools.

TotalScore = w1 * Sim(q,d) + w2 * P_success + w3 * S_llm_norm
           + w4 * Speed_norm + w5 * Efficiency_norm

Where:
  Sim(q,d)       - cosine similarity between query and tool description embeddings
  P_success      - historical API success rate [0, 1]
  S_llm_norm     - LLM-as-Judge avg score normalized to [0, 1]
  Speed_norm     - latency score: 1 / (1 + latency_ms / 1000)
  Efficiency_norm - token efficiency: 1 / (1 + avg_token_cost / 1000)
"""

from dataclasses import dataclass, field
from typing import Optional
from .embeddings import cosine_similarity


@dataclass
class RankingWeights:
    semantic: float = 0.40    # w1: query-description similarity
    success_rate: float = 0.20  # w2: historical reliability
    llm_score: float = 0.20   # w3: LLM-as-Judge quality
    speed: float = 0.10       # w4: latency (lower is better)
    efficiency: float = 0.10  # w5: token cost (lower is better)

    def validate(self):
        total = self.semantic + self.success_rate + self.llm_score + self.speed + self.efficiency
        assert abs(total - 1.0) < 1e-6, f"Weights must sum to 1.0, got {total}"


DEFAULT_WEIGHTS = RankingWeights()


def score_tool(
    tool: dict,
    query_embedding: list[float],
    weights: RankingWeights = DEFAULT_WEIGHTS,
) -> tuple[float, dict]:
    """
    Score a single tool against a query.
    Returns (total_score, breakdown_dict).
    """
    tool_embedding = tool.get("embedding") or []

    # Semantic similarity [0, 1]
    sim = cosine_similarity(query_embedding, tool_embedding)
    # Shift from [-1,1] to [0,1] for cosine
    sim_norm = (sim + 1.0) / 2.0

    # Success rate already [0, 1]; default 0.5 if no data
    p_success = tool.get("avg_success_rate") or 0.0
    if tool.get("total_calls", 0) == 0:
        p_success = 0.5  # neutral prior for new tools

    # LLM score [1,5] → normalized [0,1]; default 0.5 if no data
    llm_raw = tool.get("avg_llm_score") or 0.0
    llm_norm = (llm_raw - 1.0) / 4.0 if llm_raw > 0 else 0.5

    # Speed: lower latency is better
    latency = tool.get("avg_latency_ms") or 0.0
    speed_norm = 1.0 / (1.0 + latency / 1000.0)

    # Efficiency: lower token cost is better
    token_cost = tool.get("avg_token_cost") or 0.0
    efficiency_norm = 1.0 / (1.0 + token_cost / 1000.0)

    total = (
        weights.semantic * sim_norm
        + weights.success_rate * p_success
        + weights.llm_score * llm_norm
        + weights.speed * speed_norm
        + weights.efficiency * efficiency_norm
    )

    breakdown = {
        "semantic": round(sim_norm, 4),
        "success_rate": round(p_success, 4),
        "llm_score": round(llm_norm, 4),
        "speed": round(speed_norm, 4),
        "efficiency": round(efficiency_norm, 4),
        "total": round(total, 4),
    }

    return total, breakdown


def rank_tools(
    tools: list[dict],
    query_embedding: list[float],
    top_k: int = 5,
    weights: RankingWeights = DEFAULT_WEIGHTS,
) -> list[dict]:
    """
    Rank a list of tools by SkillRank score.
    Returns top_k tools with rank_score and rank_breakdown added.
    """
    if not tools:
        return []

    scored = []
    for tool in tools:
        total, breakdown = score_tool(tool, query_embedding, weights)
        result = dict(tool)
        result["rank_score"] = round(total, 4)
        result["rank_breakdown"] = breakdown
        scored.append(result)

    scored.sort(key=lambda t: t["rank_score"], reverse=True)
    return scored[:top_k]
