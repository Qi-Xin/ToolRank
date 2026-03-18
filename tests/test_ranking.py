"""Tests for the SkillRank ranking algorithm."""
import pytest
from api.ranking import rank_tools, score_tool, RankingWeights
from api.embeddings import MockEmbeddingProvider, cosine_similarity


MOCK = MockEmbeddingProvider()


def make_tool(name, description, success_rate=0.0, latency_ms=0.0, token_cost=0.0, llm_score=0.0, total_calls=0):
    emb = MOCK.embed(description)
    return {
        "id": f"id-{name}",
        "name": name,
        "description": description,
        "embedding": emb,
        "avg_success_rate": success_rate,
        "avg_latency_ms": latency_ms,
        "avg_token_cost": token_cost,
        "avg_llm_score": llm_score,
        "total_calls": total_calls,
    }


class TestCosigneSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_opposite_vectors(self):
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_orthogonal_vectors(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_empty_vectors_return_zero(self):
        assert cosine_similarity([], []) == 0.0

    def test_mismatched_length_returns_zero(self):
        assert cosine_similarity([1.0, 0.0], [1.0]) == 0.0


class TestScoreTool:
    def test_new_tool_gets_neutral_prior(self):
        tool = make_tool("new-tool", "web search", total_calls=0)
        q_emb = MOCK.embed("web search")
        score, breakdown = score_tool(tool, q_emb)
        # New tools get 0.5 prior for success_rate
        assert breakdown["success_rate"] == pytest.approx(0.5)
        assert 0.0 <= score <= 1.0

    def test_perfect_tool_scores_high(self):
        tool = make_tool(
            "perfect",
            "web search engine",
            success_rate=1.0,
            latency_ms=10.0,
            token_cost=50.0,
            llm_score=5.0,
            total_calls=100,
        )
        q_emb = MOCK.embed("web search engine")
        score, _ = score_tool(tool, q_emb)
        assert score > 0.8

    def test_bad_tool_scores_low(self):
        tool = make_tool(
            "bad-tool",
            "some tool",
            success_rate=0.1,
            latency_ms=9000.0,
            token_cost=5000.0,
            llm_score=1.0,
            total_calls=100,
        )
        q_emb = MOCK.embed("some tool")
        score, _ = score_tool(tool, q_emb)
        assert score < 0.5

    def test_score_breakdown_fields_present(self):
        tool = make_tool("t", "tool", total_calls=10, success_rate=0.8)
        score, breakdown = score_tool(tool, MOCK.embed("tool"))
        for key in ("semantic", "success_rate", "llm_score", "speed", "efficiency", "total"):
            assert key in breakdown

    def test_custom_weights(self):
        # With all weight on success_rate, a 100% success tool should score close to 1.0
        weights = RankingWeights(
            semantic=0.0, success_rate=1.0, llm_score=0.0, speed=0.0, efficiency=0.0
        )
        tool = make_tool("t", "any", success_rate=1.0, total_calls=10)
        score, _ = score_tool(tool, MOCK.embed("any"), weights=weights)
        assert score == pytest.approx(1.0)

    def test_weights_must_sum_to_one(self):
        bad = RankingWeights(semantic=0.5, success_rate=0.5, llm_score=0.5, speed=0.0, efficiency=0.0)
        with pytest.raises(AssertionError):
            bad.validate()


class TestRankTools:
    def test_returns_top_k(self):
        tools = [make_tool(f"tool-{i}", f"description {i}", total_calls=5) for i in range(10)]
        q_emb = MOCK.embed("description 3")
        results = rank_tools(tools, q_emb, top_k=3)
        assert len(results) == 3

    def test_results_sorted_descending(self):
        tools = [make_tool(f"t{i}", f"text {i}", success_rate=i / 10, total_calls=10) for i in range(5)]
        q_emb = MOCK.embed("text")
        results = rank_tools(tools, q_emb, top_k=5)
        scores = [r["rank_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_tools_returns_empty(self):
        assert rank_tools([], MOCK.embed("query"), top_k=5) == []

    def test_semantically_relevant_tool_ranks_higher(self):
        # MockEmbeddingProvider is hash-based (no real semantics), so we use
        # identical text for the query and one tool to guarantee max similarity.
        query = "search the web and find URLs from the internet"
        tools = [
            make_tool("web-search", query, total_calls=10, success_rate=0.7),
            make_tool("db-tool", "query SQL databases and manage records", total_calls=10, success_rate=0.7),
        ]
        q_emb = MOCK.embed(query)
        results = rank_tools(tools, q_emb, top_k=2)
        # web-search has cosine similarity 1.0 (identical text); db-tool has lower similarity
        assert results[0]["name"] == "web-search"

    def test_top_k_larger_than_tools(self):
        tools = [make_tool(f"t{i}", f"desc {i}") for i in range(3)]
        results = rank_tools(tools, MOCK.embed("desc"), top_k=10)
        assert len(results) == 3
