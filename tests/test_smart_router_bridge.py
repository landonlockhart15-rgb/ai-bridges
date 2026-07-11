import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


class MockFastMCP:
    def __init__(self, name):
        self.name = name
        self.tools = {}
        self.resources = {}

    def tool(self, *args, **kwargs):
        def decorator(func):
            self.tools[func.__name__] = func
            return func
        return decorator

    def resource(self, *args, **kwargs):
        def decorator(func):
            self.resources[func.__name__] = func
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
    truncated_models = set()

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
        response.choices[0].finish_reason = "length" if model in self.truncated_models else "stop"
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
        MockOpenAI.truncated_models = set()
        smart_router.genai.Client.side_effect = None
        for path in (
            smart_router.bridge_state.STATE_FILE_PATH,
            smart_router.bridge_state.LOCK_FILE_PATH,
            smart_router.usage_tracker.USAGE_FILE_PATH,
            smart_router.usage_tracker.LOCK_FILE_PATH,
        ):
            if Path(path).exists():
                Path(path).unlink()

    @patch.dict("os.environ", {"OPENAI_API_KEY": "paid-key"}, clear=True)
    def test_api_key_alone_does_not_enable_paid_fallback(self):
        routes = smart_router._routes_for("auto")
        tiers = [route.cost_tier for route in routes]
        self.assertEqual(tiers, ["free-cloud", "free-cloud", "free-cloud", "free-cloud", "local"])

    @patch.dict("os.environ", {
        "OPENAI_API_KEY": "paid-key",
        "SMART_ROUTER_ALLOW_PAID_FALLBACK": "1",
    }, clear=True)
    def test_paid_fallback_requires_explicit_opt_in(self):
        routes = smart_router._routes_for("auto")
        self.assertEqual(routes[-1].cost_tier, "paid")

    @patch.dict("os.environ", {"GROQ_API_KEY": "gsk_test_key"}, clear=True)
    def test_simple_prompt_prefers_local_first_in_auto_mode(self):
        routes = smart_router._routes_for("auto", prompt="Please summarize this release note in one sentence.")
        tiers = [route.cost_tier for route in routes]
        self.assertEqual(tiers[0], "local")
        self.assertNotIn("paid", tiers)

    @patch.dict("os.environ", {"GROQ_API_KEY": "gsk_test_key"}, clear=True)
    def test_complex_prompt_prefers_free_cloud_before_local_in_auto_mode(self):
        routes = smart_router._routes_for(
            "auto",
            prompt="Refactor this stateful router to support multi-step retries, capability scoring, and edge-case handling.",
        )
        tiers = [route.cost_tier for route in routes]
        self.assertEqual(tiers[0], "free-cloud")
        self.assertIn("local", tiers)
        self.assertNotIn("paid", tiers)

    @patch.dict("os.environ", {"GROQ_API_KEY": "gsk_test_key"}, clear=True)
    def test_ask_smart_uses_first_available_free_route(self):
        result = smart_router.ask_smart("hello", "auto")
        self.assertEqual(result, "llama-3.3-70b-versatile: hello")
        self.assertEqual(MockOpenAI.calls, [("https://api.groq.com/openai/v1", "llama-3.3-70b-versatile")])

    @patch.dict("os.environ", {"GROQ_API_KEY": "gsk_test_key", "OPENAI_API_KEY": "paid-key", "SMART_ROUTER_ALLOW_PAID_FALLBACK": "1"}, clear=True)
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

    @patch.dict("os.environ", {"GROQ_API_KEY": "gsk_test_key", "OPENAI_API_KEY": "paid-key", "SMART_ROUTER_ALLOW_PAID_FALLBACK": "1"}, clear=True)
    def test_paid_task_type_still_requests_gpt_first_for_complex_prompts(self):
        result = smart_router.ask_smart("Refactor this module and explain the tradeoffs.", "paid")
        self.assertEqual(result, "gpt-4o-mini: Refactor this module and explain the tradeoffs.")
        self.assertEqual(MockOpenAI.calls, [(None, "gpt-4o-mini")])

    @patch.dict("os.environ", {"GEMINI_API_KEY": "gemini-key", "OPENAI_API_KEY": "paid-key", "SMART_ROUTER_ALLOW_PAID_FALLBACK": "1"}, clear=True)
    def test_provider_error_falls_back_to_next_route(self):
        smart_router.genai.Client.side_effect = RuntimeError("gemini unavailable")
        MockOpenAI.failing_models = {"gemma4:latest"}
        result = smart_router.ask_smart("provider down", "auto")
        self.assertEqual(result, "gpt-4o-mini: provider down")

    @patch.dict("os.environ", {"GROQ_API_KEY": "gsk_test_key", "OPENAI_API_KEY": "paid-key", "SMART_ROUTER_ALLOW_PAID_FALLBACK": "1"}, clear=True)
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

    @patch.dict("os.environ", {"GROQ_API_KEY": "gsk_test_key", "OPENAI_API_KEY": "paid-key", "SMART_ROUTER_ALLOW_PAID_FALLBACK": "1"}, clear=True)
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

    @patch.dict("os.environ", {"GROQ_API_KEY": "gsk_test_key", "OPENAI_API_KEY": "paid-key", "SMART_ROUTER_ALLOW_PAID_FALLBACK": "1"}, clear=True)
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
        "SMART_ROUTER_PAID_MODEL": "my-paid-gpt",
        "SMART_ROUTER_ALLOW_PAID_FALLBACK": "1",
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

    def test_lock_release_race_preserves_reacquired_lock(self):
        lock_file = Path(smart_router.bridge_state.LOCK_FILE_PATH)
        lock1 = smart_router.bridge_state.SimpleFileLock(str(lock_file))
        lock2 = smart_router.bridge_state.SimpleFileLock(str(lock_file))

        lock1.__enter__()
        self.assertTrue(lock_file.exists())

        original_rename = smart_router.bridge_state.os.rename

        def raced_rename(src, dst):
            if src == str(lock_file) and dst == f"{lock_file}.release.{lock1.owner_id}":
                lock_file.write_text(lock2.owner_id, encoding="utf-8")
            return original_rename(src, dst)

        with patch.object(smart_router.bridge_state.os, "rename", side_effect=raced_rename):
            lock1.__exit__(None, None, None)

        self.assertTrue(lock_file.exists())
        self.assertEqual(lock_file.read_text(encoding="utf-8").strip(), lock2.owner_id)

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

    def test_record_metric_trims_pre_existing_overflow_without_reordering(self):
        # Seed distinct values so we can prove the newest 10 survive in order.
        provider = "excess-provider"
        model = "excess-model"

        model_latency_history = [float(i) for i in range(15)]
        model_success_history = [i % 2 for i in range(15)]
        provider_latency_history = [float(100 + i) for i in range(15)]
        provider_success_history = [(i + 1) % 2 for i in range(15)]
        provider_rate_limit_history = [1 if i in (2, 5, 8, 11, 14) else 0 for i in range(15)]

        with smart_router.bridge_state.SimpleFileLock(smart_router.bridge_state.LOCK_FILE_PATH):
            state = smart_router.bridge_state.load_state()
            providers = state.setdefault("providers", {})
            provider_state = providers.setdefault(provider, {"status": "ok", "models": {}})
            model_state = provider_state.setdefault("models", {}).setdefault(model, {"status": "ok"})
            model_state["latency_history"] = model_latency_history[:]
            model_state["success_history"] = model_success_history[:]

            provider_state["latency_history"] = provider_latency_history[:]
            provider_state["success_history"] = provider_success_history[:]
            provider_state["rate_limit_history"] = provider_rate_limit_history[:]
            smart_router.bridge_state.save_state(state)

        new_latency = 99.5
        smart_router.bridge_state.record_metric(provider, model, latency=new_latency, success=False, is_rate_limit=True)

        expected_model_latency = model_latency_history[6:] + [new_latency]
        expected_model_success = model_success_history[6:] + [0]
        expected_provider_latency = provider_latency_history[6:] + [new_latency]
        expected_provider_success = provider_success_history[6:] + [0]
        expected_rate_limit_history = provider_rate_limit_history[6:] + [1]

        metrics = smart_router.bridge_state.get_metrics(provider, model)
        self.assertEqual(metrics["latency_history"], expected_model_latency)
        self.assertEqual(metrics["success_history"], expected_model_success)
        self.assertEqual(metrics["avg_latency"], sum(expected_model_latency) / 10)
        self.assertEqual(metrics["success_rate"], sum(expected_model_success) / 10)

        p_metrics = smart_router.bridge_state.get_provider_metrics(provider)
        self.assertEqual(p_metrics["latency_history"], expected_provider_latency)
        self.assertEqual(p_metrics["success_history"], expected_provider_success)
        self.assertEqual(p_metrics["rate_limit_history"], expected_rate_limit_history)
        self.assertEqual(p_metrics["avg_latency"], sum(expected_provider_latency) / 10)
        self.assertEqual(p_metrics["success_rate"], sum(expected_provider_success) / 10)

    def test_get_metrics_treats_missing_model_as_unseen_and_reliable_but_slow(self):
        metrics = smart_router.bridge_state.get_metrics("missing-provider", "missing-model")
        self.assertEqual(metrics["latency_history"], [])
        self.assertEqual(metrics["success_history"], [])
        self.assertEqual(metrics["avg_latency"], 9999.0)
        self.assertEqual(metrics["success_rate"], 1.0)

    def test_get_provider_metrics_treats_missing_provider_as_empty(self):
        metrics = smart_router.bridge_state.get_provider_metrics("missing-provider")
        self.assertEqual(metrics["latency_history"], [])
        self.assertEqual(metrics["success_history"], [])
        self.assertEqual(metrics["rate_limit_history"], [])
        self.assertEqual(metrics["avg_latency"], 9999.0)
        self.assertEqual(metrics["success_rate"], 1.0)

    @patch.dict("os.environ", {
        "GROQ_API_KEY": "gsk_test_key",
        "GEMINI_API_KEY": "gemini-key",
        "SMART_ROUTER_LATENCY_THRESHOLD": "2.0",
        "SMART_ROUTER_RATE_LIMIT_THRESHOLD": "99",
    }, clear=True)
    def test_provider_latency_degradation_is_strict_and_needs_three_samples(self):
        for _ in range(2):
            smart_router.bridge_state.record_metric(
                "groq-bridge",
                "llama-3.3-70b-versatile",
                latency=2.0,
                success=True,
            )

        health = smart_router.bridge_state.get_route_health("groq-bridge", "llama-3.3-70b-versatile")
        self.assertFalse(health["provider_is_degraded"])

        smart_router.bridge_state.record_metric(
            "groq-bridge",
            "llama-3.3-70b-versatile",
            latency=2.0,
            success=True,
        )
        health = smart_router.bridge_state.get_route_health("groq-bridge", "llama-3.3-70b-versatile")
        self.assertFalse(health["provider_is_degraded"])

        smart_router.bridge_state.record_metric(
            "groq-bridge",
            "llama-3.3-70b-versatile",
            latency=2.1,
            success=True,
        )
        health = smart_router.bridge_state.get_route_health("groq-bridge", "llama-3.3-70b-versatile")
        self.assertTrue(health["provider_is_degraded"])

    @patch.dict("os.environ", {
        "SMART_ROUTER_LATENCY_THRESHOLD": "1.0",
        "BRIDGE_SLOW_CALL_THRESHOLD": "3",
    }, clear=True)
    def test_sustained_high_latency_opens_provider_circuit_and_fast_probe_recovers(self):
        provider = "groq-bridge"
        model = "llama-3.3-70b-versatile"

        for _ in range(2):
            smart_router.bridge_state.record_metric(provider, model, latency=1.1, success=True)
        self.assertTrue(smart_router.bridge_state.is_available(provider, model))

        smart_router.bridge_state.record_metric(provider, model, latency=1.1, success=True)
        self.assertFalse(smart_router.bridge_state.is_available(provider, model))
        self.assertFalse(smart_router.bridge_state.is_available(provider, "another-model"))
        health = smart_router.bridge_state.get_route_health(provider, model)
        self.assertEqual(health["consecutive_slow_calls"], 3)

        smart_router.bridge_state.record_metric(provider, model, latency=0.2, success=True)
        self.assertTrue(smart_router.bridge_state.is_available(provider, model))
        health = smart_router.bridge_state.get_route_health(provider, model)
        self.assertEqual(health["consecutive_slow_calls"], 0)

    @patch.dict("os.environ", {}, clear=True)
    def test_latency_circuit_breaker_defaults(self):
        provider = "groq-bridge"
        model = "llama-3.3-70b-versatile"
        
        # Verify default latency threshold does not trip at <= 5.0s (default threshold is 5.0)
        for _ in range(3):
            smart_router.bridge_state.record_metric(provider, model, latency=5.0, success=True)
        self.assertTrue(smart_router.bridge_state.is_available(provider, model))
        
        # Verify default slow call threshold (3) trips after 3 consecutive > 5.0s calls
        for _ in range(2):
            smart_router.bridge_state.record_metric(provider, model, latency=5.1, success=True)
        self.assertTrue(smart_router.bridge_state.is_available(provider, model))
        
        smart_router.bridge_state.record_metric(provider, model, latency=5.1, success=True)
        self.assertFalse(smart_router.bridge_state.is_available(provider, model))

    @patch.dict("os.environ", {
        "SMART_ROUTER_LATENCY_THRESHOLD": "1.0",
        "BRIDGE_SLOW_CALL_THRESHOLD": "3",
    }, clear=True)
    def test_latency_circuit_breaker_non_slow_call_resets_count(self):
        provider = "groq-bridge"
        model = "llama-3.3-70b-versatile"
        
        # Two slow calls: consecutive_slow_calls should be 2
        smart_router.bridge_state.record_metric(provider, model, latency=1.2, success=True)
        smart_router.bridge_state.record_metric(provider, model, latency=1.2, success=True)
        health = smart_router.bridge_state.get_route_health(provider, model)
        self.assertEqual(health["consecutive_slow_calls"], 2)
        
        # One fast call: consecutive_slow_calls should reset to 0
        smart_router.bridge_state.record_metric(provider, model, latency=0.8, success=True)
        health = smart_router.bridge_state.get_route_health(provider, model)
        self.assertEqual(health["consecutive_slow_calls"], 0)
        self.assertTrue(smart_router.bridge_state.is_available(provider, model))

    @patch.dict("os.environ", {
        "SMART_ROUTER_LATENCY_THRESHOLD": "1.0",
        "BRIDGE_SLOW_CALL_THRESHOLD": "3",
    }, clear=True)
    def test_latency_circuit_breaker_failed_call_resets_count(self):
        provider = "groq-bridge"
        model = "llama-3.3-70b-versatile"
        
        # Two slow calls: consecutive_slow_calls should be 2
        smart_router.bridge_state.record_metric(provider, model, latency=1.2, success=True)
        smart_router.bridge_state.record_metric(provider, model, latency=1.2, success=True)
        health = smart_router.bridge_state.get_route_health(provider, model)
        self.assertEqual(health["consecutive_slow_calls"], 2)
        
        # A failed call should reset consecutive_slow_calls to 0
        smart_router.bridge_state.record_metric(provider, model, latency=0.1, success=False)
        health = smart_router.bridge_state.get_route_health(provider, model)
        self.assertEqual(health["consecutive_slow_calls"], 0)
        self.assertTrue(smart_router.bridge_state.is_available(provider, model))

    # ------------------------------------------------------------------
    # Adversarial coverage for the latency circuit breaker (bridge_state)
    # ------------------------------------------------------------------

    @patch.dict("os.environ", {
        "SMART_ROUTER_LATENCY_THRESHOLD": "1.0",
        "BRIDGE_SLOW_CALL_THRESHOLD": "3",
    }, clear=True)
    def test_latency_exactly_at_threshold_is_not_slow(self):
        """Boundary: `latency > threshold` is strict, so latency == threshold must never
        count as a slow call, even when repeated far past the slow-call threshold."""
        provider = "groq-bridge"
        model = "llama-3.3-70b-versatile"
        for _ in range(10):
            smart_router.bridge_state.record_metric(provider, model, latency=1.0, success=True)
        health = smart_router.bridge_state.get_route_health(provider, model)
        self.assertEqual(health["consecutive_slow_calls"], 0)
        self.assertTrue(smart_router.bridge_state.is_available(provider, model))

    @patch.dict("os.environ", {
        "SMART_ROUTER_LATENCY_THRESHOLD": "1.0",
        "BRIDGE_SLOW_CALL_THRESHOLD": "3",
    }, clear=True)
    def test_fast_success_does_not_reopen_error_tripped_circuit(self):
        """A fast successful call resets the slow-call counter but must NOT clear a circuit
        that was opened by the error-based breaker (reason without a 'high_latency:' prefix).
        The recovery guard keys off the reason string; if it were sloppy it would silently
        revive a provider still cooling off from real 429/5xx errors."""
        provider = "groq-bridge"
        model = "llama-3.3-70b-versatile"

        # Trip the error-based breaker at the provider level (default test threshold = 1).
        smart_router.bridge_state.mark_unavailable(
            provider, "server error 500", is_429_or_5xx=True
        )
        self.assertFalse(smart_router.bridge_state.is_available(provider, model))

        # A fast success arrives (e.g. a stray in-flight response). Must stay open.
        smart_router.bridge_state.record_metric(provider, model, latency=0.1, success=True)
        self.assertFalse(smart_router.bridge_state.is_available(provider, model))
        state = smart_router.bridge_state.load_state()
        pstate = state["providers"][provider]
        self.assertEqual(pstate["status"], "open")
        self.assertEqual(pstate["reason"], "server error 500")
        # But the slow-call counter is still reset.
        self.assertEqual(pstate.get("consecutive_slow_calls", 0), 0)

    @patch.dict("os.environ", {
        "SMART_ROUTER_LATENCY_THRESHOLD": "1.0",
        "BRIDGE_SLOW_CALL_THRESHOLD": "3",
    }, clear=True)
    def test_failed_call_does_not_recover_latency_tripped_circuit(self):
        """Only a fast *successful* probe recovers a latency-opened circuit. A failed call
        resets the counter but must leave the provider open/cooling off."""
        provider = "groq-bridge"
        model = "llama-3.3-70b-versatile"

        for _ in range(3):
            smart_router.bridge_state.record_metric(provider, model, latency=1.5, success=True)
        self.assertFalse(smart_router.bridge_state.is_available(provider, model))

        # Failure with tiny latency: resets counter but does NOT restore status.
        smart_router.bridge_state.record_metric(provider, model, latency=0.01, success=False)
        self.assertFalse(smart_router.bridge_state.is_available(provider, model))
        state = smart_router.bridge_state.load_state()
        pstate = state["providers"][provider]
        self.assertEqual(pstate["status"], "open")
        self.assertTrue(str(pstate["reason"]).startswith("high_latency:"))
        self.assertEqual(pstate.get("consecutive_slow_calls", 0), 0)

    @patch.dict("os.environ", {
        "SMART_ROUTER_LATENCY_THRESHOLD": "1.0",
        "BRIDGE_SLOW_CALL_THRESHOLD": "3",
    }, clear=True)
    def test_slow_calls_accumulate_across_models_of_same_provider(self):
        """The slow-call counter lives at the provider level, so slowness spread across
        different models of one provider still trips the provider-wide circuit and takes
        every model down with it."""
        provider = "groq-bridge"
        smart_router.bridge_state.record_metric(provider, "model-a", latency=1.5, success=True)
        smart_router.bridge_state.record_metric(provider, "model-b", latency=1.5, success=True)
        self.assertTrue(smart_router.bridge_state.is_available(provider, "model-a"))
        smart_router.bridge_state.record_metric(provider, "model-c", latency=1.5, success=True)

        self.assertFalse(smart_router.bridge_state.is_available(provider, "model-a"))
        self.assertFalse(smart_router.bridge_state.is_available(provider, "model-b"))
        self.assertFalse(smart_router.bridge_state.is_available(provider, "model-c"))

    @patch.dict("os.environ", {
        "SMART_ROUTER_LATENCY_THRESHOLD": "1.0",
        "BRIDGE_SLOW_CALL_THRESHOLD": "3",
    }, clear=True)
    def test_fast_success_on_other_model_recovers_provider_circuit(self):
        """A fast probe on any model of the provider recovers the shared provider circuit."""
        provider = "groq-bridge"
        for _ in range(3):
            smart_router.bridge_state.record_metric(provider, "model-a", latency=1.5, success=True)
        self.assertFalse(smart_router.bridge_state.is_available(provider, "model-a"))

        smart_router.bridge_state.record_metric(provider, "model-b", latency=0.1, success=True)
        self.assertTrue(smart_router.bridge_state.is_available(provider, "model-a"))
        self.assertTrue(smart_router.bridge_state.is_available(provider, "model-b"))

    @patch.dict("os.environ", {
        "SMART_ROUTER_LATENCY_THRESHOLD": "1.0",
        "BRIDGE_SLOW_CALL_THRESHOLD": "3",
    }, clear=True)
    def test_latency_trip_records_reason_and_cooldown_without_touching_failures(self):
        """Pin the reason format (recovery depends on the 'high_latency:' prefix) and that
        a latency trip sets a future cooldown while leaving consecutive_failures untouched —
        the two breakers must not cross-contaminate their counters."""
        provider = "groq-bridge"
        model = "llama-3.3-70b-versatile"
        for _ in range(3):
            smart_router.bridge_state.record_metric(provider, model, latency=2.0, success=True)

        state = smart_router.bridge_state.load_state()
        pstate = state["providers"][provider]
        self.assertEqual(pstate["status"], "open")
        self.assertEqual(pstate["reason"], "high_latency: 3 consecutive calls exceeded 1.000s")
        self.assertGreater(pstate["cooldown_until"], smart_router.bridge_state._now())
        self.assertEqual(pstate.get("consecutive_failures", 0), 0)
        health = smart_router.bridge_state.get_route_health(provider, model)
        self.assertEqual(health["consecutive_failures"], 0)

    @patch.dict("os.environ", {
        "SMART_ROUTER_LATENCY_THRESHOLD": "1.0",
        "BRIDGE_SLOW_CALL_THRESHOLD": "3",
    }, clear=True)
    def test_slow_calls_keep_climbing_while_already_open(self):
        """Once open, further slow calls should keep incrementing the counter and refresh the
        reason/cooldown rather than resetting — the breaker must not accidentally self-heal
        while the provider is still slow."""
        provider = "groq-bridge"
        model = "llama-3.3-70b-versatile"
        for _ in range(5):
            smart_router.bridge_state.record_metric(provider, model, latency=2.0, success=True)
        health = smart_router.bridge_state.get_route_health(provider, model)
        self.assertEqual(health["consecutive_slow_calls"], 5)
        self.assertFalse(smart_router.bridge_state.is_available(provider, model))
        state = smart_router.bridge_state.load_state()
        self.assertIn("5 consecutive calls", state["providers"][provider]["reason"])

    @patch.dict("os.environ", {}, clear=True)
    def test_malformed_slow_call_threshold_env_raises(self):
        """Adversarial input: a non-integer BRIDGE_SLOW_CALL_THRESHOLD is not sanitized, so
        record_metric raises ValueError. This pins current (fragile) behavior so a future
        guard/regression is visible."""
        import os as _os
        _os.environ["BRIDGE_SLOW_CALL_THRESHOLD"] = "not-a-number"
        with self.assertRaises(ValueError):
            smart_router.bridge_state.record_metric(
                "groq-bridge", "llama-3.3-70b-versatile", latency=0.1, success=True
            )

    @patch.dict("os.environ", {
        "GROQ_API_KEY": "gsk_test_key",
        "GEMINI_API_KEY": "gemini-key",
        "SMART_ROUTER_LATENCY_THRESHOLD": "5.0",
        "SMART_ROUTER_RATE_LIMIT_THRESHOLD": "2",
    }, clear=True)
    def test_rate_limit_window_deprioritizes_then_recovers_when_old_samples_roll_off(self):
        for _ in range(2):
            smart_router.bridge_state.record_metric(
                "groq-bridge",
                "llama-3.3-70b-versatile",
                latency=0.1,
                success=False,
                is_rate_limit=True,
            )
        smart_router.bridge_state.record_metric(
            "gemini-bridge",
            "gemini-2.5-flash",
            latency=0.5,
            success=True,
        )

        health = smart_router.bridge_state.get_route_health("groq-bridge", "llama-3.3-70b-versatile")
        self.assertTrue(health["provider_is_degraded"])

        routes = smart_router._routes_for("auto")
        free_cloud_providers = [r.provider for r in routes if r.cost_tier == "free-cloud" and r.provider in ("groq-bridge", "gemini-bridge")]
        self.assertEqual(free_cloud_providers[0], "gemini-bridge")
        self.assertEqual(free_cloud_providers[1], "groq-bridge")

        for _ in range(10):
            smart_router.bridge_state.record_metric(
                "groq-bridge",
                "llama-3.3-70b-versatile",
                latency=0.1,
                success=True,
                is_rate_limit=False,
            )

        provider_metrics = smart_router.bridge_state.get_provider_metrics("groq-bridge")
        self.assertEqual(len(provider_metrics["rate_limit_history"]), 10)
        self.assertEqual(sum(provider_metrics["rate_limit_history"]), 0)

        health = smart_router.bridge_state.get_route_health("groq-bridge", "llama-3.3-70b-versatile")
        self.assertFalse(health["provider_is_degraded"])

        routes = smart_router._routes_for("auto")
        free_cloud_providers = [r.provider for r in routes if r.cost_tier == "free-cloud" and r.provider in ("groq-bridge", "gemini-bridge")]
        self.assertEqual(free_cloud_providers[0], "groq-bridge")
        self.assertEqual(free_cloud_providers[1], "gemini-bridge")

    @patch.dict("os.environ", {"GROQ_API_KEY": "gsk_test_key", "OPENAI_API_KEY": "paid-key", "SMART_ROUTER_ALLOW_PAID_FALLBACK": "1"}, clear=True)
    def test_circuit_breaker_requires_repeated_errors_to_trip(self):
        with patch.dict("os.environ", {"BRIDGE_FAILURE_THRESHOLD": "3"}):
            MockOpenAI.failing_models = {"llama-3.3-70b-versatile"}

            # First call: groq fails, fallback to gpt-4o-mini
            result1 = smart_router.ask_smart("first call", "auto")
            self.assertEqual(result1, "gpt-4o-mini: first call")

            # Since threshold is 3, groq-bridge should still be available!
            self.assertTrue(smart_router.bridge_state.is_available("groq-bridge", "llama-3.3-70b-versatile"))

            # Second call: groq fails again, fallback to gpt-4o-mini
            MockOpenAI.calls = []
            result2 = smart_router.ask_smart("second call", "auto")
            self.assertEqual(result2, "gpt-4o-mini: second call")
            self.assertTrue(smart_router.bridge_state.is_available("groq-bridge", "llama-3.3-70b-versatile"))

            # Third call: groq fails a third time, fallback to gpt-4o-mini
            MockOpenAI.calls = []
            result3 = smart_router.ask_smart("third call", "auto")
            self.assertEqual(result3, "gpt-4o-mini: third call")

            # Now the circuit breaker should have tripped! Groq-bridge is unavailable.
            self.assertFalse(smart_router.bridge_state.is_available("groq-bridge", "llama-3.3-70b-versatile"))

            # Fourth call: groq-bridge is in cooldown/open state, so it shouldn't even be called!
            MockOpenAI.calls = []
            result4 = smart_router.ask_smart("fourth call", "auto")
            self.assertEqual(result4, "gpt-4o-mini: fourth call")
            called_models = [call[1] for call in MockOpenAI.calls]
            self.assertNotIn("llama-3.3-70b-versatile", called_models)

    @patch.dict("os.environ", {"GROQ_API_KEY": "gsk_test_key", "OPENAI_API_KEY": "paid-key", "SMART_ROUTER_ALLOW_PAID_FALLBACK": "1"}, clear=True)
    def test_circuit_breaker_resets_on_success(self):
        with patch.dict("os.environ", {"BRIDGE_FAILURE_THRESHOLD": "3"}):
            MockOpenAI.failing_models = {"llama-3.3-70b-versatile"}

            # Fail once
            smart_router.ask_smart("first call", "auto")
            # Fail twice
            smart_router.ask_smart("second call", "auto")

            # Now make it succeed
            MockOpenAI.failing_models = set()
            smart_router.ask_smart("third call", "auto")

            # The count should be reset to 0. Let's make it fail again.
            MockOpenAI.failing_models = {"llama-3.3-70b-versatile"}
            smart_router.ask_smart("fourth call", "auto")

            # Groq-bridge should still be available because the first two failures were reset by success!
            self.assertTrue(smart_router.bridge_state.is_available("groq-bridge", "llama-3.3-70b-versatile"))

    @patch.dict("os.environ", {"GROQ_API_KEY": "gsk_test_key", "GEMINI_API_KEY": "gemini-key", "OPENAI_API_KEY": "paid-key", "SMART_ROUTER_ALLOW_PAID_FALLBACK": "1"}, clear=True)
    def test_dynamic_fallback_chain_health_deprioritization(self):
        # 1. Initially all are healthy: Free (Groq, Gemini, Cerebras, OpenRouter) -> Local (hf-bridge) -> Paid (gpt-bridge)
        routes1 = smart_router._routes_for("auto")
        self.assertEqual(routes1[0].provider, "groq-bridge")
        self.assertEqual(routes1[4].provider, "hf-bridge")
        self.assertEqual(routes1[5].provider, "gpt-bridge")

        # 2. Under threshold=3, one failure on groq-bridge degrades its health (consecutive_failures=1) but keeps it available
        with patch.dict("os.environ", {"BRIDGE_FAILURE_THRESHOLD": "3"}):
            smart_router.bridge_state.mark_unavailable("groq-bridge", "temporary error", model="llama-3.3-70b-versatile", is_429_or_5xx=True)

            # Now groq-bridge has 1 failure, gemini-bridge has 0 failures.
            # groq-bridge should be deprioritized within the free-cloud tier. Gemini-bridge is first.
            routes2 = smart_router._routes_for("auto")
            self.assertEqual(routes2[0].provider, "gemini-bridge")

            # 3. Mark all free cloud routes as degraded (failures=1)
            smart_router.bridge_state.mark_unavailable("gemini-bridge", "temporary error", model="gemini-2.5-flash", is_429_or_5xx=True)
            smart_router.bridge_state.mark_unavailable("cerebras-bridge", "temporary error", model="gpt-oss-120b", is_429_or_5xx=True)
            smart_router.bridge_state.mark_unavailable("openrouter-bridge", "temporary error", model="nvidia/nemotron-3-super-120b-a12b:free", is_429_or_5xx=True)

            # 4. Now all Free models are degraded (failures=1). Local model (hf-bridge) is healthy (failures=0).
            # Cost priority is preserved: Free models (even degraded) should be tried before Local models.
            routes3 = smart_router._routes_for("auto")
            free_cloud_providers = [r.provider for r in routes3[:4]]
            self.assertIn("groq-bridge", free_cloud_providers)
            self.assertEqual(routes3[4].provider, "hf-bridge")

            # 5. Fail groq-bridge 2 more times to trip the circuit breaker (cooldown)
            smart_router.bridge_state.mark_unavailable("groq-bridge", "temporary error", model="llama-3.3-70b-versatile", is_429_or_5xx=True)
            smart_router.bridge_state.mark_unavailable("groq-bridge", "temporary error", model="llama-3.3-70b-versatile", is_429_or_5xx=True)

            # Now groq-bridge is unavailable (on cooldown). It must be sorted to the end of the list.
            routes4 = smart_router._routes_for("auto")
            self.assertEqual(routes4[-1].provider, "groq-bridge")

    @patch.dict("os.environ", {"GROQ_API_KEY": "gsk_test_key", "GEMINI_API_KEY": "gemini-key", "OPENAI_API_KEY": "paid-key", "SMART_ROUTER_ALLOW_PAID_FALLBACK": "1"}, clear=True)
    def test_auto_cost_tier_stays_free_then_local_then_paid_even_with_better_paid_metrics(self):
        # Give the paid route artificially strong health so the sort key has a chance to mis-rank it.
        for _ in range(4):
            smart_router.bridge_state.record_metric("gpt-bridge", "gpt-4o-mini", latency=0.01, success=True)

        # Make one free route look worse and another look better within the free tier.
        smart_router.bridge_state.record_metric("groq-bridge", "llama-3.3-70b-versatile", latency=1.5, success=False)
        smart_router.bridge_state.record_metric("gemini-bridge", "gemini-2.5-flash", latency=0.2, success=True)

        routes = smart_router._routes_for("auto")
        tiers = [route.cost_tier for route in routes]
        self.assertEqual(tiers[:4], ["free-cloud", "free-cloud", "free-cloud", "free-cloud"])
        self.assertEqual(tiers[4], "local")
        self.assertEqual(tiers[5], "paid")

        free_cloud_routes = [route.provider for route in routes if route.cost_tier == "free-cloud"]
        self.assertEqual(free_cloud_routes[0], "gemini-bridge")
        self.assertIn("groq-bridge", free_cloud_routes)

    @patch.dict("os.environ", {"GROQ_API_KEY": "gsk_test_key", "OPENAI_API_KEY": "paid-key", "SMART_ROUTER_ALLOW_PAID_FALLBACK": "1"}, clear=True)
    def test_empty_and_whitespace_task_types_default_to_auto_cost_order(self):
        auto_routes = smart_router._routes_for("auto")
        none_routes = smart_router._routes_for(None)
        empty_routes = smart_router._routes_for("")
        spaced_routes = smart_router._routes_for("   AuTo   ")

        auto_order = [route.provider for route in auto_routes]
        self.assertEqual([route.provider for route in none_routes], auto_order)
        self.assertEqual([route.provider for route in empty_routes], auto_order)
        self.assertEqual([route.provider for route in spaced_routes], auto_order)

    def test_provider_level_metric_accumulation(self):
        provider = "groq-bridge"
        model1 = "model-1"
        model2 = "model-2"

        smart_router.bridge_state.record_metric(provider, model1, latency=1.5, success=True, is_rate_limit=False)
        smart_router.bridge_state.record_metric(provider, model2, latency=2.5, success=False, is_rate_limit=True)

        p_metrics = smart_router.bridge_state.get_provider_metrics(provider)
        self.assertEqual(p_metrics["latency_history"], [1.5, 2.5])
        self.assertEqual(p_metrics["success_history"], [1, 0])
        self.assertEqual(p_metrics["rate_limit_history"], [0, 1])
        self.assertEqual(p_metrics["avg_latency"], 2.0)
        self.assertEqual(p_metrics["success_rate"], 0.5)

    @patch.dict("os.environ", {
        "GROQ_API_KEY": "gsk_test_key",
        "GEMINI_API_KEY": "gemini-key",
        "SMART_ROUTER_LATENCY_THRESHOLD": "1.0",
    }, clear=True)
    def test_latency_window_recovery_rolls_off_old_slow_samples(self):
        for _ in range(3):
            smart_router.bridge_state.record_metric(
                "groq-bridge",
                "llama-3.3-70b-versatile",
                latency=3.0,
                success=True,
            )

        smart_router.bridge_state.record_metric(
            "gemini-bridge",
            "gemini-2.5-flash",
            latency=0.5,
            success=True,
        )

        pre_recovery_routes = smart_router._routes_for("auto")
        pre_recovery_free = [
            r.provider for r in pre_recovery_routes
            if r.cost_tier == "free-cloud" and r.provider in ("groq-bridge", "gemini-bridge")
        ]
        self.assertEqual(pre_recovery_free[0], "gemini-bridge")
        self.assertEqual(pre_recovery_free[1], "groq-bridge")

        for _ in range(10):
            smart_router.bridge_state.record_metric(
                "groq-bridge",
                "llama-3.3-70b-versatile",
                latency=0.1,
                success=True,
            )

        health = smart_router.bridge_state.get_route_health("groq-bridge", "llama-3.3-70b-versatile")
        self.assertFalse(health["provider_is_degraded"])
        self.assertLess(health["provider_avg_latency"], 1.0)
        self.assertEqual(health["success_rate"], 1.0)

        post_recovery_routes = smart_router._routes_for("auto")
        post_recovery_free = [
            r.provider for r in post_recovery_routes
            if r.cost_tier == "free-cloud" and r.provider in ("groq-bridge", "gemini-bridge")
        ]
        self.assertEqual(post_recovery_free[0], "groq-bridge")
        self.assertEqual(post_recovery_free[1], "gemini-bridge")

    def test_malformed_state_file_falls_back_to_empty_state(self):
        state_path = Path(smart_router.bridge_state.STATE_FILE_PATH)
        state_path.write_text("{ not valid json", encoding="utf-8")

        state = smart_router.bridge_state.load_state()
        self.assertEqual(state, {"version": 1, "providers": {}})

        health = smart_router.bridge_state.get_route_health("groq-bridge", "llama-3.3-70b-versatile")
        self.assertTrue(health["is_available"])
        self.assertEqual(health["consecutive_failures"], 0)
        self.assertEqual(health["success_rate"], 1.0)
        self.assertEqual(health["avg_latency"], 9999.0)
        self.assertFalse(health["provider_is_degraded"])

    @patch.dict("os.environ", {
        "GROQ_API_KEY": "gsk_test_key",
        "GEMINI_API_KEY": "gemini-key",
        "SMART_ROUTER_LATENCY_THRESHOLD": "2.0",
        "SMART_ROUTER_RATE_LIMIT_THRESHOLD": "2"
    }, clear=True)
    def test_dynamic_deprioritization_due_to_high_latency(self):
        routes = smart_router._routes_for("auto")
        self.assertEqual(routes[0].provider, "groq-bridge")

        for _ in range(3):
            smart_router.bridge_state.record_metric("groq-bridge", "llama-3.3-70b-versatile", latency=3.0, success=True)

        smart_router.bridge_state.record_metric("gemini-bridge", "gemini-2.5-flash", latency=0.5, success=True)

        routes = smart_router._routes_for("auto")
        free_cloud_providers = [r.provider for r in routes if r.cost_tier == "free-cloud" and r.provider in ("groq-bridge", "gemini-bridge")]
        self.assertEqual(free_cloud_providers[0], "gemini-bridge")
        self.assertEqual(free_cloud_providers[1], "groq-bridge")

    @patch.dict("os.environ", {
        "GROQ_API_KEY": "gsk_test_key",
        "GEMINI_API_KEY": "gemini-key",
        "SMART_ROUTER_LATENCY_THRESHOLD": "5.0",
        "SMART_ROUTER_RATE_LIMIT_THRESHOLD": "2"
    }, clear=True)
    def test_dynamic_deprioritization_due_to_frequent_429s(self):
        smart_router.bridge_state.record_metric("groq-bridge", "llama-3.3-70b-versatile", latency=0.1, success=False, is_rate_limit=True)
        smart_router.bridge_state.record_metric("groq-bridge", "llama-3.3-70b-versatile", latency=0.1, success=False, is_rate_limit=True)

        smart_router.bridge_state.record_metric("gemini-bridge", "gemini-2.5-flash", latency=0.5, success=True)

        routes = smart_router._routes_for("auto")
        free_cloud_providers = [r.provider for r in routes if r.cost_tier == "free-cloud" and r.provider in ("groq-bridge", "gemini-bridge")]
        self.assertEqual(free_cloud_providers[0], "gemini-bridge")
        self.assertEqual(free_cloud_providers[1], "groq-bridge")

    @patch.dict("os.environ", {
        "GROQ_API_KEY": "gsk_test_key",
        "GEMINI_API_KEY": "gemini-key",
        "OPENAI_API_KEY": "paid-key",
        "SMART_ROUTER_ALLOW_PAID_FALLBACK": "1",
    }, clear=True)
    def test_capability_task_type_filters_to_capable_routes_only(self):
        routes = smart_router._routes_for("creative_writing")
        providers = [r.provider for r in routes]
        # Only gemini-bridge and gpt-bridge are tagged with creative_writing.
        self.assertEqual(set(providers), {"gemini-bridge", "gpt-bridge"})
        # Cost ordering is still respected: free before paid.
        self.assertEqual(providers[0], "gemini-bridge")
        self.assertEqual(providers[-1], "gpt-bridge")

    @patch.dict("os.environ", {
        "GROQ_API_KEY": "gsk_test_key",
        "GEMINI_API_KEY": "gemini-key",
        "OPENAI_API_KEY": "paid-key",
        "SMART_ROUTER_ALLOW_PAID_FALLBACK": "1",
    }, clear=True)
    def test_simple_extraction_capability_excludes_gemini_and_cerebras(self):
        routes = smart_router._routes_for("simple_extraction")
        providers = {r.provider for r in routes}
        self.assertIn("groq-bridge", providers)
        self.assertIn("gpt-bridge", providers)
        self.assertNotIn("gemini-bridge", providers)
        self.assertNotIn("cerebras-bridge", providers)

    @patch.dict("os.environ", {"GROQ_API_KEY": "gsk_test_key"}, clear=True)
    def test_unrecognized_task_type_is_unaffected_by_capability_filtering(self):
        # "auto" is not a capability task type, so all routes remain candidates.
        routes = smart_router._routes_for("auto")
        self.assertEqual(len(routes), 5)

    @patch.dict("os.environ", {
        "GROQ_API_KEY": "gsk_test_key",
        "OPENAI_API_KEY": "paid-key",
        "SMART_ROUTER_ALLOW_PAID_FALLBACK": "1",
    }, clear=True)
    def test_creative_writing_fallback_when_gemini_missing_env(self):
        # GEMINI_API_KEY is not set, so gemini-bridge is not available.
        # It should fall back to other free/local routes like groq-bridge first, then paid gpt-bridge.
        routes = smart_router._routes_for("creative_writing")
        providers = [r.provider for r in routes]
        self.assertIn("groq-bridge", providers)
        self.assertEqual(providers[-1], "gpt-bridge")

    @patch.dict("os.environ", {
        "GROQ_API_KEY": "gsk_test_key",
        "GEMINI_API_KEY": "gemini-key",
        "OPENAI_API_KEY": "paid-key",
        "SMART_ROUTER_ALLOW_PAID_FALLBACK": "1",
    }, clear=True)
    def test_creative_writing_fallback_when_gemini_cooling_down(self):
        # Mark gemini-bridge as unavailable (cooling down)
        smart_router.bridge_state.mark_unavailable("gemini-bridge", "429", model="gemini-2.5-flash")

        # It should fall back to other free/local routes like groq-bridge first, then paid gpt-bridge.
        routes = smart_router._routes_for("creative_writing")
        providers = [r.provider for r in routes]
        self.assertIn("groq-bridge", providers)
        self.assertLess(providers.index("groq-bridge"), providers.index("gpt-bridge"))
        self.assertGreater(providers.index("gemini-bridge"), providers.index("gpt-bridge"))

    @patch.dict("os.environ", {"GROQ_API_KEY": "gsk_test_key"}, clear=True)
    def test_task_complexity_heuristics_edge_cases(self):
        # 1. Empty, None, and whitespace-only prompts default to "medium"
        self.assertEqual(smart_router._task_complexity(None), "medium")
        self.assertEqual(smart_router._task_complexity(""), "medium")
        self.assertEqual(smart_router._task_complexity("   \n  \t "), "medium")

        # 2. Simple boundary: word counts around 40
        # Exactly 40 words with simple keyword -> "simple"
        prompt_40_simple = "summarize " + "word " * 39
        self.assertEqual(len(prompt_40_simple.split()), 40)
        self.assertEqual(smart_router._task_complexity(prompt_40_simple), "simple")

        # Exactly 41 words with simple keyword -> "medium"
        prompt_41_simple = "summarize " + "word " * 40
        self.assertEqual(len(prompt_41_simple.split()), 41)
        self.assertEqual(smart_router._task_complexity(prompt_41_simple), "medium")

        # 3. Complex boundary: word counts around 180
        # Exactly 179 words without complex keyword -> "medium"
        prompt_179_no_keyword = "word " * 179
        self.assertEqual(len(prompt_179_no_keyword.split()), 179)
        self.assertEqual(smart_router._task_complexity(prompt_179_no_keyword), "medium")

        # Exactly 180 words without complex keyword -> "complex"
        prompt_180_no_keyword = "word " * 180
        self.assertEqual(len(prompt_180_no_keyword.split()), 180)
        self.assertEqual(smart_router._task_complexity(prompt_180_no_keyword), "complex")

        # 4. Keyword casing and substring matching
        # Case insensitivity
        self.assertEqual(smart_router._task_complexity("REFACTOR code"), "complex")
        self.assertEqual(smart_router._task_complexity("SuMmArIzE this"), "simple")

        # Substring keyword matching (e.g., 'debugging' contains 'debug', 'proofreading' contains 'proofread')
        self.assertEqual(smart_router._task_complexity("debugging this session"), "complex")
        self.assertEqual(smart_router._task_complexity("proofreading this text"), "simple")

        # Both simple and complex keywords present -> complex takes precedence
        self.assertEqual(smart_router._task_complexity("summarize and refactor"), "complex")

    @patch.dict("os.environ", {
        "GROQ_API_KEY": "gsk_test_key",
        "GEMINI_API_KEY": "gemini-key",
        "OPENAI_API_KEY": "paid-key",
        "SMART_ROUTER_ALLOW_PAID_FALLBACK": "1",
    }, clear=True)
    def test_task_profile_aliases_reuse_existing_capability_filtering(self):
        # "refactor" and "unit_test" are task-profile aliases for the "coding"
        # capability, so they should filter to the same routes as "coding" itself.
        coding_routes = smart_router._routes_for("coding")
        for alias in ("refactor", "unit_test", "bug_fix", "code_review"):
            aliased_routes = smart_router._routes_for(alias)
            self.assertEqual(
                [r.provider for r in aliased_routes],
                [r.provider for r in coding_routes],
            )

        # "docs" aliases to "simple_extraction" and "story" aliases to "creative_writing".
        self.assertEqual(
            [r.provider for r in smart_router._routes_for("docs")],
            [r.provider for r in smart_router._routes_for("simple_extraction")],
        )
        self.assertEqual(
            [r.provider for r in smart_router._routes_for("story")],
            [r.provider for r in smart_router._routes_for("creative_writing")],
        )

    @patch.dict("os.environ", {"GROQ_API_KEY": "gsk_test_key", "OPENAI_API_KEY": "paid-key", "SMART_ROUTER_ALLOW_PAID_FALLBACK": "1"}, clear=True)
    def test_routing_fails_to_escalate_complex_prompt_to_paid_tier(self):
        # The user requested that the bridge "decide if it can safely stay on a free/local model or if it *must* escalate to a paid tier"
        # However, the current implementation maps "complex" tasks to priority: free-cloud=0, local=1, paid=2.
        # This test documents that a complex prompt DOES NOT escalate to paid tier (gpt-bridge), but stays on free-cloud.
        routes = smart_router._routes_for("auto", prompt="Refactor this large system and design a backup plan.")
        tiers = [route.cost_tier for route in routes]
        # It still prioritizes free-cloud over paid!
        self.assertEqual(tiers[0], "free-cloud")
        self.assertEqual(tiers[-1], "paid")

    @patch.dict("os.environ", {"GROQ_API_KEY": "gsk_test_key", "OPENAI_API_KEY": "paid-key", "SMART_ROUTER_ALLOW_PAID_FALLBACK": "1"}, clear=True)
    def test_truncated_free_response_escalates_to_next_route(self):
        # groq's answer hits the token limit, so the router should not stop there
        # and instead escalate through the remaining routes (local hf-bridge is
        # unreachable in this mock, so it lands on the paid tier).
        MockOpenAI.truncated_models = {"llama-3.3-70b-versatile"}
        result = smart_router.ask_smart("write something long", "auto")
        self.assertIn("llama-3.3-70b-versatile: write something long", result)
        self.assertIn("gpt-4o-mini: Continue the answer", result)
        called_models = [call[1] for call in MockOpenAI.calls]
        self.assertEqual(called_models, ["llama-3.3-70b-versatile", "gemma4:latest", "gpt-4o-mini"])

        # A truncated response is not a real failure, so the route should not be
        # cooling down afterwards.
        self.assertTrue(smart_router.bridge_state.is_available("groq-bridge", "llama-3.3-70b-versatile"))

    def test_continuation_merge_removes_repeated_boundary(self):
        self.assertEqual(
            smart_router._merge_continuation("alpha beta", "beta gamma"),
            "alpha beta gamma",
        )

    @patch.dict("os.environ", {"GROQ_API_KEY": "gsk_test_key"}, clear=True)
    def test_truncated_response_used_as_fallback_when_nothing_completes(self):
        # No paid route configured and the local route is unreachable, so the
        # truncated free-tier answer is returned rather than raising an error.
        MockOpenAI.truncated_models = {"llama-3.3-70b-versatile"}
        result = smart_router.ask_smart("write something long", "auto")
        self.assertEqual(result, "llama-3.3-70b-versatile: write something long")

    @patch.dict("os.environ", {"OPENAI_API_KEY": "paid-key"}, clear=True)
    def test_paid_route_truncation_is_returned_directly(self):
        # The paid tier is the last resort, so a truncated paid response is
        # simply returned rather than triggering further escalation.
        MockOpenAI.truncated_models = {"gpt-4o-mini"}
        result = smart_router.ask_smart("write something long", "paid")
        self.assertEqual(result, "gpt-4o-mini: write something long")


class TestSmartRouterMissingLibraries(unittest.TestCase):
    def setUp(self):
        self.sys_modules_backup = sys.modules.copy()

    def tearDown(self):
        sys.modules.clear()
        sys.modules.update(self.sys_modules_backup)

    def test_missing_dependencies_graceful_routing(self):
        if "openai" in sys.modules:
            del sys.modules["openai"]
        if "google" in sys.modules:
            del sys.modules["google"]
        if "google.genai" in sys.modules:
            del sys.modules["google.genai"]

        spec = importlib.util.spec_from_file_location("smart_router_no_libs", ROOT / "smart-router-bridge" / "server.py")
        router_no_libs = importlib.util.module_from_spec(spec)

        with patch.dict(sys.modules, {"openai": None, "google": None, "google.genai": None}):
            spec.loader.exec_module(router_no_libs)

            self.assertIsNone(router_no_libs.openai)
            self.assertIsNone(router_no_libs.OpenAI)
            self.assertIsNone(router_no_libs.genai)

            routes = router_no_libs._routes_for("auto")
            for route in routes:
                self.assertFalse(router_no_libs._library_available(route))

            with self.assertRaises(router_no_libs.bridge_state.ProviderUnavailableError) as ctx:
                router_no_libs.ask_smart("hello", "auto")
            self.assertIn("required library not installed", str(ctx.exception))

    def test_status_cache_bypass(self):
        import time
        import json
        cache_path = Path(smart_router.bridge_state.STATUS_CACHE_FILE)
        cache_data = {
            "timestamp": time.time(),
            "bridges": {
                "groq-bridge": {
                    "id": "groq-bridge",
                    "status": "🔴 Offline",
                    "details": "Connection failed"
                }
            }
        }

        if cache_path.exists():
            try:
                cache_path.unlink()
            except OSError:
                pass

        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_data, f)

            self.assertFalse(smart_router.bridge_state.is_available("groq-bridge", "llama-3.3-70b-versatile"))

            health = smart_router.bridge_state.get_route_health("groq-bridge", "llama-3.3-70b-versatile")
            self.assertFalse(health["is_available"])
        finally:
            if cache_path.exists():
                try:
                    cache_path.unlink()
                except OSError:
                    pass

    def test_mcp_resources_definition(self):
        self.assertIn("get_bridges_status", smart_router.mcp.resources)
        self.assertIn("get_smart_router_status", smart_router.mcp.resources)

        res = smart_router.mcp.resources["get_bridges_status"]()
        self.assertIn("Status cache file not found", res)

        res_router = smart_router.mcp.resources["get_smart_router_status"]()
        import json
        data = json.loads(res_router)
        self.assertIn("routes", data)
        self.assertGreater(len(data["routes"]), 0)


if __name__ == "__main__":
    unittest.main()

