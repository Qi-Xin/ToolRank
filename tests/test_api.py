"""Integration tests for the FastAPI application."""
import pytest
import os
import tempfile
from fastapi.testclient import TestClient

# Use a temp DB for each test session
_tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_tmp.close()
os.environ["SKILLRANK_TEST_DB"] = _tmp.name

import api.database as db
db.set_db_path(_tmp.name)

from api.main import app, get_provider
from api.embeddings import MockEmbeddingProvider

# Override embedding provider with mock for tests
app.dependency_overrides = {}
_mock_provider = MockEmbeddingProvider()


def _override_provider():
    return _mock_provider


import api.main as main_module
main_module._provider = _mock_provider

db.init_db()

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_db():
    """Reset DB state before each test."""
    with db.get_conn() as conn:
        conn.execute("DELETE FROM execution_logs")
        conn.execute("DELETE FROM reviews")
        conn.execute("DELETE FROM tools")
    yield


def create_tool(name="brave-search", description="Search the web using Brave API", category="search"):
    return client.post("/tools", json={
        "name": name,
        "description": description,
        "category": category,
        "install_cmd": f"npx {name}-mcp",
        "url": f"https://example.com/{name}",
    })


class TestHealth:
    def test_health_check(self):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


class TestToolCRUD:
    def test_create_tool(self):
        r = create_tool()
        assert r.status_code == 201
        data = r.json()
        assert data["name"] == "brave-search"
        assert data["category"] == "search"
        assert "id" in data

    def test_create_duplicate_tool_fails(self):
        create_tool()
        r = create_tool()
        assert r.status_code == 409

    def test_get_tool(self):
        tool_id = create_tool().json()["id"]
        r = client.get(f"/tools/{tool_id}")
        assert r.status_code == 200
        assert r.json()["id"] == tool_id

    def test_get_nonexistent_tool(self):
        r = client.get("/tools/does-not-exist")
        assert r.status_code == 404

    def test_list_tools(self):
        create_tool("tool-a", "Tool A description", "search")
        create_tool("tool-b", "Tool B description", "code")
        r = client.get("/tools")
        assert r.status_code == 200
        assert len(r.json()) == 2

    def test_list_tools_filter_by_category(self):
        create_tool("tool-a", "Tool A", "search")
        create_tool("tool-b", "Tool B", "code")
        r = client.get("/tools?category=search")
        assert r.status_code == 200
        names = [t["name"] for t in r.json()]
        assert "tool-a" in names
        assert "tool-b" not in names


class TestSearch:
    def test_search_returns_results(self):
        create_tool("brave-search", "Search the web and find internet content")
        create_tool("github", "Manage GitHub repositories and code", "code")
        r = client.get("/search?q=find+information+on+the+web")
        assert r.status_code == 200
        results = r.json()
        assert len(results) >= 1
        assert "rank_score" in results[0]
        assert "rank_breakdown" in results[0]

    def test_search_respects_top_k(self):
        for i in range(10):
            create_tool(f"tool-{i}", f"Tool number {i} for various tasks")
        r = client.get("/search?q=various+tasks&top_k=3")
        assert r.status_code == 200
        assert len(r.json()) == 3

    def test_search_empty_db(self):
        r = client.get("/search?q=anything")
        assert r.status_code == 200
        assert r.json() == []

    def test_search_post_endpoint(self):
        create_tool("brave-search", "Search the web")
        r = client.post("/search", json={"query": "web search", "top_k": 5})
        assert r.status_code == 200
        assert len(r.json()) >= 1

    def test_search_results_have_decreasing_scores(self):
        create_tool("tool-a", "web search engine for finding content online")
        create_tool("tool-b", "database query tool for SQL operations")
        create_tool("tool-c", "file system management and reading writing files")
        r = client.get("/search?q=search+the+web&top_k=3")
        scores = [t["rank_score"] for t in r.json()]
        assert scores == sorted(scores, reverse=True)


class TestReviews:
    def test_submit_review(self):
        tool_id = create_tool().json()["id"]
        r = client.post("/reviews", json={
            "tool_id": tool_id,
            "rating": 4,
            "success": True,
            "use_case": "web search",
            "pros": "Fast and accurate",
            "cons": "Rate limited",
        })
        assert r.status_code == 201
        data = r.json()
        assert data["rating"] == 4
        assert data["success"] == True

    def test_review_invalid_rating(self):
        tool_id = create_tool().json()["id"]
        r = client.post("/reviews", json={
            "tool_id": tool_id,
            "rating": 6,
            "success": True,
        })
        assert r.status_code == 422

    def test_review_nonexistent_tool(self):
        r = client.post("/reviews", json={
            "tool_id": "nonexistent",
            "rating": 3,
            "success": False,
        })
        assert r.status_code == 404

    def test_get_tool_reviews(self):
        tool_id = create_tool().json()["id"]
        client.post("/reviews", json={"tool_id": tool_id, "rating": 5, "success": True})
        client.post("/reviews", json={"tool_id": tool_id, "rating": 3, "success": False})
        r = client.get(f"/tools/{tool_id}/reviews")
        assert r.status_code == 200
        assert len(r.json()) == 2

    def test_get_reviews_empty(self):
        tool_id = create_tool().json()["id"]
        r = client.get(f"/tools/{tool_id}/reviews")
        assert r.status_code == 200
        assert r.json() == []


class TestExecutionLogs:
    def test_submit_log(self):
        tool_id = create_tool().json()["id"]
        r = client.post("/logs", json={
            "tool_id": tool_id,
            "success": True,
            "latency_ms": 234.5,
            "token_input": 100,
            "token_output": 200,
        })
        assert r.status_code == 201
        data = r.json()
        assert data["latency_ms"] == 234.5

    def test_log_updates_tool_stats(self):
        tool_id = create_tool().json()["id"]
        # Submit 4 successful and 1 failed log
        for _ in range(4):
            client.post("/logs", json={"tool_id": tool_id, "success": True, "latency_ms": 100.0})
        client.post("/logs", json={"tool_id": tool_id, "success": False, "latency_ms": 500.0})

        tool = client.get(f"/tools/{tool_id}").json()
        assert tool["total_calls"] == 5
        assert tool["avg_success_rate"] == pytest.approx(0.8)
        assert tool["avg_latency_ms"] == pytest.approx(180.0)

    def test_log_nonexistent_tool(self):
        r = client.post("/logs", json={
            "tool_id": "nonexistent",
            "success": True,
            "latency_ms": 100.0,
        })
        assert r.status_code == 404

    def test_stats_affect_ranking(self):
        """Tools with better stats should rank higher when scores are otherwise equal."""
        id_a = create_tool("tool-a", "general purpose tool for tasks").json()["id"]
        id_b = create_tool("tool-b", "general purpose tool for tasks", "search").json()["id"]

        # tool-a: 95% success, low latency
        for _ in range(19):
            client.post("/logs", json={"tool_id": id_a, "success": True, "latency_ms": 50.0})
        client.post("/logs", json={"tool_id": id_a, "success": False, "latency_ms": 50.0})

        # tool-b: 30% success, high latency
        for _ in range(3):
            client.post("/logs", json={"tool_id": id_b, "success": True, "latency_ms": 5000.0})
        for _ in range(7):
            client.post("/logs", json={"tool_id": id_b, "success": False, "latency_ms": 5000.0})

        r = client.get("/search?q=general+purpose+tool+for+tasks&top_k=2")
        results = r.json()
        assert results[0]["name"] == "tool-a"
