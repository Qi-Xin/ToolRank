"""Tests for the Tracker SDK."""
import asyncio
import time
import pytest
from unittest.mock import patch, MagicMock
from sdk.skillrank.tracker import track_tool, _estimate_tokens, _build_payload


class TestEstimateTokens:
    def test_empty_inputs(self):
        inp, out = _estimate_tokens((), {}, None)
        assert inp >= 0
        assert out == 0

    def test_nonempty_result(self):
        _, out = _estimate_tokens((), {}, "hello world this is a result")
        assert out > 0

    def test_larger_input_more_tokens(self):
        inp_small, _ = _estimate_tokens(("short",), {}, None)
        inp_large, _ = _estimate_tokens(("a very long input string " * 50,), {}, None)
        assert inp_large > inp_small


class TestBuildPayload:
    def test_payload_structure(self):
        payload = _build_payload("brave-search", True, 123.4, None, "claude-sonnet-4-6", (10, 20))
        assert payload["tool_name"] == "brave-search"
        assert payload["success"] is True
        assert payload["latency_ms"] == 123.4
        assert payload["agent_model"] == "claude-sonnet-4-6"
        assert payload["token_input"] == 10
        assert payload["token_output"] == 20

    def test_error_payload(self):
        payload = _build_payload("tool", False, 50.0, "TimeoutError", None, (5, 0))
        assert payload["success"] is False
        assert payload["error_message"] == "TimeoutError"


class TestSyncDecorator:
    def test_passthrough_return_value(self):
        @track_tool("test-tool", api_url="http://localhost:9999")
        def my_tool(x):
            return x * 2

        assert my_tool(5) == 10

    def test_exception_propagates(self):
        @track_tool("test-tool", api_url="http://localhost:9999")
        def failing_tool():
            raise ValueError("something went wrong")

        with pytest.raises(ValueError, match="something went wrong"):
            failing_tool()

    def test_measures_latency(self):
        call_log = []

        with patch("sdk.skillrank.tracker._log_background") as mock_log:
            mock_log.side_effect = lambda *args, **kwargs: call_log.append(args)

            @track_tool("test-tool", api_url="http://localhost:9999")
            def slow_tool():
                time.sleep(0.05)
                return "done"

            slow_tool()

        assert len(call_log) == 1
        # latency_ms is the 3rd arg (api_url, tool_name, success, latency_ms, ...)
        latency_ms = call_log[0][3]
        assert latency_ms >= 40  # at least 40ms

    def test_logs_failure_on_exception(self):
        call_log = []

        with patch("sdk.skillrank.tracker._log_background") as mock_log:
            mock_log.side_effect = lambda *args, **kwargs: call_log.append(args)

            @track_tool("test-tool", api_url="http://localhost:9999")
            def failing():
                raise RuntimeError("fail")

            with pytest.raises(RuntimeError):
                failing()

        assert len(call_log) == 1
        success = call_log[0][2]
        assert success is False


class TestAsyncDecorator:
    def test_async_passthrough(self):
        @track_tool("test-tool", api_url="http://localhost:9999")
        async def async_tool(x):
            return x + 1

        result = asyncio.run(async_tool(10))
        assert result == 11

    def test_async_exception_propagates(self):
        @track_tool("test-tool", api_url="http://localhost:9999")
        async def async_failing():
            raise ValueError("async error")

        with pytest.raises(ValueError):
            asyncio.run(async_failing())

    def test_async_measures_latency(self):
        call_log = []

        with patch("sdk.skillrank.tracker._log_async") as mock_log:
            mock_log.side_effect = lambda *args, **kwargs: call_log.append(args)

            @track_tool("test-tool", api_url="http://localhost:9999")
            async def slow_async():
                await asyncio.sleep(0.05)
                return "done"

            asyncio.run(slow_async())

        assert len(call_log) == 1
        latency_ms = call_log[0][3]
        assert latency_ms >= 40
