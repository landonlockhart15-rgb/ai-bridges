import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


class MockFastMCP:
    def __init__(self, name):
        self.name = name
        self.tools = {}

    def tool(self, *args, **kwargs):
        def decorator(func):
            self.tools[func.__name__] = func
            return func
        return decorator

    def run(self, *args, **kwargs):
        pass


class MockRateLimitError(Exception):
    pass


class APIConnectionError(Exception):
    pass


class MockOpenAI:
    calls = []
    failing_models = set()

    def __init__(self, api_key=None, base_url=None, default_headers=None):
        self.api_key = api_key
        self.base_url = base_url
        self.default_headers = default_headers
        self.chat = MagicMock()
        self.chat.completions = MagicMock()
        self.chat.completions.create = MagicMock(side_effect=self._mock_create)

    def _mock_create(self, model, messages, **kwargs):
        self.calls.append((self.base_url, model))
        if model in self.failing_models:
            raise MockRateLimitError(f"rate limit for {model}")
        if self.base_url and ("localhost" in self.base_url or "127.0.0.1" in self.base_url):
            raise APIConnectionError("Connection refused")
        response = MagicMock()
        response.usage.prompt_tokens = 3
        response.usage.completion_tokens = 4
        response.choices = [MagicMock()]
        response.choices[0].message.content = f"{model}: {messages[-1]['content']}"
        return response


mock_fastmcp_mod = MagicMock()
mock_fastmcp_mod.FastMCP = MockFastMCP
sys.modules["mcp"] = MagicMock()
sys.modules["mcp.server"] = MagicMock()
sys.modules["mcp.server.fastmcp"] = mock_fastmcp_mod

mock_openai_mod = MagicMock()
mock_openai_mod.OpenAI = MockOpenAI
mock_openai_mod.RateLimitError = MockRateLimitError
mock_openai_mod.APIConnectionError = APIConnectionError
sys.modules["openai"] = mock_openai_mod

mock_google_mod = MagicMock()
mock_genai_mod = MagicMock()
mock_google_mod.genai = mock_genai_mod
sys.modules["google"] = mock_google_mod
sys.modules["google.genai"] = mock_genai_mod

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("smart_router_server", ROOT / "smart-router-bridge" / "server.py")
smart_router = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smart_router)


class TestSmartRouterBridge(unittest.TestCase):
    def setUp(self):
        MockOpenAI.calls = []
        MockOpenAI.failing_models = set()
        smart_router.genai.Client.side_effect = None
        for path in (
            smart_router.bridge_state.STATE_FILE_PATH,
            smart_router.bridge_state.LOCK_FILE_PATH,
            smart_router.usage_tracker.USAGE_FILE_PATH,
            smart_router.usage_tracker.LOCK_FILE_PATH,
        ):
            if Path(path).exists():
                Path(path).unlink()

    def test_default_order_is_free_local_then_paid(self):
        routes = smart_router._routes_for("auto")
        tiers = [route.cost_tier for route in routes]
        self.assertEqual(tiers[:-1], ["free-cloud", "free-cloud", "free-cloud", "free-cloud", "local"])
        self.assertEqual(tiers[-1], "paid")

    @patch.dict("os.environ", {"GROQ_API_KEY": "gsk_test_key"}, clear=True)
    def test_ask_smart_uses_first_available_free_route(self):
        result = smart_router.ask_smart("hello", "auto")
        self.assertEqual(result, "llama-3.3-70b-versatile: hello")
        self.assertEqual(MockOpenAI.calls, [("https://api.groq.com/openai/v1", "llama-3.3-70b-versatile")])

    @patch.dict("os.environ", {"GROQ_API_KEY": "gsk_test_key", "OPENAI_API_KEY": "paid-key"}, clear=True)
    def test_paid_is_last_resort_after_free_rate_limit(self):
        MockOpenAI.failing_models = {"llama-3.3-70b-versatile", "gemma4:latest"}
        result = smart_router.ask_smart("fallback", "auto")
        self.assertEqual(result, "gpt-4o-mini: fallback")
        self.assertEqual(
            MockOpenAI.calls,
            [
                ("https://api.groq.com/openai/v1", "llama-3.3-70b-versatile"),
                ("http://localhost:11434/v1", "gemma4:latest"),
                (None, "gpt-4o-mini"),
            ],
        )

    @patch.dict("os.environ", {"GEMINI_API_KEY": "gemini-key", "OPENAI_API_KEY": "paid-key"}, clear=True)
    def test_provider_error_falls_back_to_next_route(self):
        smart_router.genai.Client.side_effect = RuntimeError("gemini unavailable")
        MockOpenAI.failing_models = {"gemma4:latest"}
        result = smart_router.ask_smart("provider down", "auto")
        self.assertEqual(result, "gpt-4o-mini: provider down")

    @patch.dict("os.environ", {"GROQ_API_KEY": "gsk_test_key", "OPENAI_API_KEY": "paid-key"}, clear=True)
    def test_paid_task_type_requests_gpt_first(self):
        result = smart_router.ask_smart("use paid", "paid")
        self.assertEqual(result, "gpt-4o-mini: use paid")
        self.assertEqual(MockOpenAI.calls, [(None, "gpt-4o-mini")])

    @patch.dict("os.environ", {}, clear=True)
    def test_ask_smart_fails_when_no_keys_configured(self):
        with self.assertRaises(smart_router.bridge_state.ProviderUnavailableError) as ctx:
            smart_router.ask_smart("hello", "auto")
        self.assertIn("No smart-router routes succeeded", str(ctx.exception))
        self.assertIn("missing GROQ_API_KEY", str(ctx.exception))

    @patch.dict("os.environ", {"GROQ_API_KEY": "gsk_test_key", "OPENAI_API_KEY": "paid-key"}, clear=True)
    def test_cooldown_avoids_previously_failed_provider(self):
        # First call: groq fails with a rate limit
        MockOpenAI.failing_models = {"llama-3.3-70b-versatile"}
        result = smart_router.ask_smart("first call", "auto")
        # Should fallback and succeed with gpt-4o-mini
        self.assertEqual(result, "gpt-4o-mini: first call")

        # Second call: groq should be in cooldown, so only gpt-4o-mini is tried
        MockOpenAI.calls = []
        result2 = smart_router.ask_smart("second call", "auto")
        self.assertEqual(result2, "gpt-4o-mini: second call")
        called_models = [call[1] for call in MockOpenAI.calls]
        self.assertNotIn("llama-3.3-70b-versatile", called_models)
        self.assertIn("gpt-4o-mini", called_models)

    @patch.dict("os.environ", {"GROQ_API_KEY": "gsk_test_key"}, clear=True)
    def test_budget_cap_exceeded_error_propagates_directly(self):
        with patch.object(smart_router.usage_tracker, "check_budget", side_effect=ValueError("Budget cap of $10.00 exceeded")):
            with self.assertRaises(ValueError) as ctx:
                smart_router.ask_smart("budget check", "auto")
            self.assertIn("Budget cap of $10.00 exceeded", str(ctx.exception))

    @patch.dict("os.environ", {"GROQ_API_KEY": "gsk_test_key", "OPENAI_API_KEY": "paid-key"}, clear=True)
    def test_poison_pill_input_disables_all_providers_for_subsequent_calls(self):
        original_create = MockOpenAI._mock_create

        def bad_create(self_obj, model, messages, **kwargs):
            if messages[-1]['content'] == "poison-pill":
                raise TypeError("object of type 'int' is not JSON serializable")
            return original_create(self_obj, model, messages, **kwargs)

        with patch.object(MockOpenAI, "_mock_create", bad_create):
            with self.assertRaises(smart_router.bridge_state.ProviderUnavailableError):
                smart_router.ask_smart("poison-pill", "auto")

        # A perfectly valid subsequent call will fail because all routes are cooling down
        with self.assertRaises(smart_router.bridge_state.ProviderUnavailableError) as ctx:
            smart_router.ask_smart("valid-prompt", "auto")
        self.assertIn("cooling down", str(ctx.exception))

    @patch.dict("os.environ", {"GROQ_API_KEY": "gsk_test_key"}, clear=True)
    def test_different_task_types_route_ordering(self):
        # 1. 'paid' order: paid model first, then free
        routes_paid = smart_router._routes_for("paid")
        self.assertEqual(routes_paid[0].cost_tier, "paid")

        # 2. 'local' order: local model first
        routes_local = smart_router._routes_for("local")
        self.assertEqual(routes_local[0].cost_tier, "local")

        # 3. 'simple' order: uses simple model 'llama-3.1-8b-instant'
        routes_simple = smart_router._routes_for("simple")
        self.assertEqual(routes_simple[0].model, "llama-3.1-8b-instant")

        # 4. None/empty task type: defaults to auto
        routes_none = smart_router._routes_for(None)
        self.assertEqual(routes_none[0].model, "llama-3.3-70b-versatile")

    @patch.dict("os.environ", {
        "GROQ_API_KEY": "gsk_test_key",
        "SMART_ROUTER_LOCAL_MODEL": "my-local-llama",
        "SMART_ROUTER_PAID_MODEL": "my-paid-gpt"
    }, clear=True)
    def test_env_var_model_overrides(self):
        routes = smart_router._routes_for("auto")
        local_route = [r for r in routes if r.provider == "hf-bridge"][0]
        paid_route = [r for r in routes if r.provider == "gpt-bridge"][0]
        self.assertEqual(local_route.model, "my-local-llama")
        self.assertEqual(paid_route.model, "my-paid-gpt")

    def test_lock_timeout_handling(self):
        lock_file = Path(smart_router.bridge_state.LOCK_FILE_PATH)
        lock_file.touch()
        try:
            with patch.object(smart_router.bridge_state, "DEFAULT_COOLDOWN_SECONDS", 1):
                with self.assertRaises(TimeoutError):
                    with smart_router.bridge_state.SimpleFileLock(str(lock_file), timeout=0.1):
                        pass
        finally:
            if lock_file.exists():
                lock_file.unlink()

    @patch.dict("os.environ", {"GROQ_API_KEY": "gsk_test_key", "GEMINI_API_KEY": "gemini-key"}, clear=True)
    def test_dynamic_latency_routing_prioritizes_faster_bridge(self):
        # 1. No metrics initially: groq-bridge is first in tier because of default definition order
        routes1 = smart_router._routes_for("auto")
        free_cloud_providers1 = [r.provider for r in routes1 if r.cost_tier == "free-cloud" and r.provider in ("groq-bridge", "gemini-bridge")]
        self.assertEqual(free_cloud_providers1[0], "groq-bridge")
        self.assertEqual(free_cloud_providers1[1], "gemini-bridge")

        # 2. Record metrics: Groq avg latency = 2.0s, Gemini avg latency = 0.5s
        smart_router.bridge_state.record_metric("groq-bridge", "llama-3.3-70b-versatile", latency=2.0, success=True)
        smart_router.bridge_state.record_metric("gemini-bridge", "gemini-2.5-flash", latency=0.5, success=True)

        # 3. Routes should be sorted so that Gemini comes first in the free-cloud tier
        routes2 = smart_router._routes_for("auto")
        free_cloud_providers2 = [r.provider for r in routes2 if r.cost_tier == "free-cloud" and r.provider in ("groq-bridge", "gemini-bridge")]
        self.assertEqual(free_cloud_providers2[0], "gemini-bridge")
        self.assertEqual(free_cloud_providers2[1], "groq-bridge")

    @patch.dict("os.environ", {"GROQ_API_KEY": "gsk_test_key", "GEMINI_API_KEY": "gemini-key"}, clear=True)
    def test_dynamic_latency_routing_deprioritizes_unreliable_bridge(self):
        # Record metrics:
        # Groq: 0.2s latency, but 50% success (1 success, 1 failure)
        smart_router.bridge_state.record_metric("groq-bridge", "llama-3.3-70b-versatile", latency=0.2, success=True)
        smart_router.bridge_state.record_metric("groq-bridge", "llama-3.3-70b-versatile", latency=0.2, success=False)
        # Gemini: 0.5s latency, 100% success (2 successes)
        smart_router.bridge_state.record_metric("gemini-bridge", "gemini-2.5-flash", latency=0.5, success=True)
        smart_router.bridge_state.record_metric("gemini-bridge", "gemini-2.5-flash", latency=0.5, success=True)

        # 3. Gemini has higher success rate, so it is prioritized even though Groq is faster when it succeeds
        routes = smart_router._routes_for("auto")
        free_cloud_providers = [r.provider for r in routes if r.cost_tier == "free-cloud" and r.provider in ("groq-bridge", "gemini-bridge")]
        self.assertEqual(free_cloud_providers[0], "gemini-bridge")
        self.assertEqual(free_cloud_providers[1], "groq-bridge")

    @patch.dict("os.environ", {"GROQ_API_KEY": "gsk_test_key"}, clear=True)
    def test_successful_ask_smart_persists_latency_metrics(self):
        result = smart_router.ask_smart("metric persistence", "auto")
        self.assertEqual(result, "llama-3.3-70b-versatile: metric persistence")

        metrics = smart_router.bridge_state.get_metrics("groq-bridge", "llama-3.3-70b-versatile")
        self.assertEqual(metrics["success_history"], [1])
        self.assertEqual(len(metrics["latency_history"]), 1)
        self.assertGreaterEqual(metrics["latency_history"][0], 0)
        self.assertLess(metrics["avg_latency"], 9999.0)

    def test_record_metric_keeps_only_the_ten_most_recent_samples(self):
        for i in range(12):
            smart_router.bridge_state.record_metric("groq-bridge", "llama-3.3-70b-versatile", latency=float(i), success=i % 3 != 0)

        metrics = smart_router.bridge_state.get_metrics("groq-bridge", "llama-3.3-70b-versatile")
        self.assertEqual(metrics["latency_history"], [float(i) for i in range(2, 12)])
        self.assertEqual(metrics["success_history"], [1, 0, 1, 1, 0, 1, 1, 0, 1, 1])
        self.assertEqual(metrics["success_rate"], 0.7)
        self.assertEqual(metrics["avg_latency"], 6.5)

    def test_get_metrics_treats_missing_model_as_unseen_and_reliable_but_slow(self):
        metrics = smart_router.bridge_state.get_metrics("missing-provider", "missing-model")
        self.assertEqual(metrics["latency_history"], [])
        self.assertEqual(metrics["success_history"], [])
        self.assertEqual(metrics["avg_latency"], 9999.0)
        self.assertEqual(metrics["success_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()

