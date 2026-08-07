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

    @patch.dict("os.environ", {"GROQ_API_KEY": "gsk_test_key"}, clear=True)
    def test_provider_heartbeat_records_free_and_local_metrics_without_paid_probe(self):
        routes = [
            smart_router.Route("groq-bridge", "groq-model", "free-cloud", "GROQ_API_KEY", lambda *_: "OK"),
            smart_router.Route("hf-bridge", "local-model", "local", None, lambda *_: "OK"),
            smart_router.Route("gpt-bridge", "paid-model", "paid", "OPENAI_API_KEY", lambda *_: "paid"),
        ]
        with patch.object(smart_router, "_routes_for", return_value=routes), \
             patch.object(smart_router.bridge_state, "is_available", return_value=True), \
             patch.object(smart_router.bridge_state, "mark_available") as mark_available, \
             patch.object(smart_router.bridge_state, "record_metric") as record_metric, \
             patch.object(smart_router.usage_tracker, "check_budget"):
            results = smart_router.run_provider_heartbeat()

        self.assertEqual([item["provider"] for item in results], ["groq-bridge", "hf-bridge"])
        self.assertEqual(mark_available.call_count, 2)
        self.assertEqual(record_metric.call_count, 2)
        self.assertTrue(all(call.kwargs["success"] for call in record_metric.call_args_list))

    def test_provider_heartbeat_marks_failed_route_unavailable(self):
        route = smart_router.Route(
            "hf-bridge", "local-model", "local", None,
            lambda *_: (_ for _ in ()).throw(RuntimeError("offline")),
        )
        with patch.object(smart_router, "_routes_for", return_value=[route]), \
             patch.object(smart_router.bridge_state, "is_available", return_value=True), \
             patch.object(smart_router.bridge_state, "record_metric") as record_metric, \
             patch.object(smart_router.bridge_state, "mark_unavailable") as mark_unavailable, \
             patch.object(smart_router.usage_tracker, "check_budget"):
            results = smart_router.run_provider_heartbeat()

        self.assertFalse(results[0]["available"])
        self.assertEqual(record_metric.call_args.kwargs["success"], False)
        mark_unavailable.assert_called_once_with(
            "hf-bridge", "RuntimeError", model="local-model",
            failure_class="transient", failure_category="unknown",
        )

    def test_fatal_authentication_errors_open_the_provider_circuit(self):
        error = RuntimeError("invalid API key")
        failure_class, category = smart_router._classify_provider_error(error)

        self.assertEqual((failure_class, category), ("fatal", "authentication"))
        smart_router.bridge_state.mark_unavailable(
            "groq-bridge", "RuntimeError",
            failure_class=failure_class, failure_category=category,
        )

        state = smart_router.bridge_state.load_state()
        provider_state = state["providers"]["groq-bridge"]
        self.assertEqual(provider_state["status"], "fatal")
        self.assertEqual(provider_state["failure_category"], "authentication")
        self.assertFalse(smart_router.bridge_state.is_available("groq-bridge", "groq-model"))

    def test_fatal_model_error_opens_only_the_model_circuit(self):
        smart_router.bridge_state.mark_unavailable(
            "groq-bridge", "NotFoundError", model="missing-model",
            failure_class="fatal", failure_category="model_not_found",
        )

        self.assertFalse(smart_router.bridge_state.is_available("groq-bridge", "missing-model"))
        self.assertTrue(smart_router.bridge_state.is_available("groq-bridge", "working-model"))

    def test_authentication_failure_skips_provider_routes_on_later_attempts(self):
        failing_ask = MagicMock(side_effect=RuntimeError("invalid API key"))
        sibling_ask = MagicMock(return_value="should not be called")
        fallback_ask = MagicMock(return_value="fallback")
        routes = [
            smart_router.Route("groq-bridge", "model-a", "free-cloud", None, failing_ask),
            smart_router.Route("groq-bridge", "model-b", "free-cloud", None, sibling_ask),
            smart_router.Route("hf-bridge", "local-model", "local", None, fallback_ask),
        ]

        with patch.object(smart_router, "_routes_for", return_value=routes), \
             patch.object(smart_router.usage_tracker, "check_budget"):
            self.assertEqual(smart_router.ask_smart("first"), "fallback")
            self.assertEqual(smart_router.ask_smart("second"), "fallback")

        failing_ask.assert_called_once()
        sibling_ask.assert_not_called()
        self.assertEqual(fallback_ask.call_count, 2)

    def test_classifier_marks_model_and_invalid_request_errors_fatal(self):
        self.assertEqual(
            smart_router._classify_provider_error(RuntimeError("model not found")),
            ("fatal", "model_not_found"),
        )
        self.assertEqual(
            smart_router._classify_provider_error(RuntimeError("invalid request parameter")),
            ("fatal", "invalid_parameters"),
        )

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

    @patch.dict("os.environ", {"GROQ_API_KEY": "gsk_test_key"}, clear=True)
    def test_preflight_skips_unavailable_primary_before_dispatch(self):
        smart_router.bridge_state.mark_unavailable("groq-bridge", "cooldown", model="llama-3.3-70b-versatile")
        route = smart_router.Route("hf-bridge", "local-model", "local", None, lambda *_: "fallback")
        with patch.object(smart_router, "_routes_for", return_value=[
            smart_router.Route("groq-bridge", "primary", "free-cloud", "GROQ_API_KEY", MagicMock()),
            route,
        ]), patch.object(smart_router.bridge_state, "get_route_health", side_effect=[
            {"is_available": False, "token_bucket_available": True},
            {"is_available": True, "token_bucket_available": True},
        ]):
            self.assertEqual(smart_router.ask_smart("hello"), "fallback")

    def test_preflight_reports_depleted_bucket_without_consuming_it(self):
        routes = [smart_router.Route("groq-bridge", "model", "free-cloud", None, MagicMock())]
        with patch.object(smart_router, "_library_available", return_value=True), \
             patch.object(smart_router, "_env_available", return_value=True), \
             patch.object(smart_router.bridge_state, "get_route_health", return_value={
                 "is_available": True, "token_bucket_available": False,
                 "token_bucket_wait_seconds": 2.5,
             }), patch.object(smart_router.bridge_state, "consume_provider_token") as consume:
            viable, errors = smart_router._preflight_routes(routes)
        self.assertEqual(viable, [])
        self.assertIn("token bucket throttled (refill in 2.5s)", errors[0])
        consume.assert_not_called()

    def test_preflight_handles_empty_routes_list(self):
        viable, errors = smart_router._preflight_routes([])
        self.assertEqual(viable, [])
        self.assertEqual(errors, [])

    def test_preflight_filters_multiple_routes_with_diverse_failures(self):
        route1 = smart_router.Route("uninstalled-bridge", "m1", "free-cloud", None, MagicMock())
        route2 = smart_router.Route("missing-env-bridge", "m2", "free-cloud", "MISSING_ENV_KEY", MagicMock())
        route3 = smart_router.Route("cooling-bridge", "m3", "free-cloud", None, MagicMock())
        route4 = smart_router.Route("throttled-bridge", "m4", "free-cloud", None, MagicMock())
        route5 = smart_router.Route("healthy-bridge", "m5", "local", None, MagicMock())

        def mock_health(provider, model):
            if provider == "cooling-bridge":
                return {"is_available": False, "token_bucket_available": True}
            elif provider == "throttled-bridge":
                return {"is_available": True, "token_bucket_available": False, "token_bucket_wait_seconds": 3.7}
            return {"is_available": True, "token_bucket_available": True}

        def mock_lib(route):
            return route.provider != "uninstalled-bridge"

        with patch.object(smart_router, "_library_available", side_effect=mock_lib), \
             patch.object(smart_router.bridge_state, "get_route_health", side_effect=mock_health):
            viable, errors = smart_router._preflight_routes([route1, route2, route3, route4, route5])

        self.assertEqual(viable, [route5])
        self.assertEqual(len(errors), 4)
        self.assertIn("uninstalled-bridge/m1: required library not installed", errors[0])
        self.assertIn("missing-env-bridge/m2: missing MISSING_ENV_KEY", errors[1])
        self.assertIn("cooling-bridge/m3: cooling down", errors[2])
        self.assertIn("throttled-bridge/m4: token bucket throttled (refill in 3.7s)", errors[3])

    def test_preflight_all_routes_unviable_raises_provider_unavailable(self):
        routes = [
            smart_router.Route("bridge1", "m1", "free-cloud", None, MagicMock()),
            smart_router.Route("bridge2", "m2", "free-cloud", None, MagicMock()),
        ]
        with patch.object(smart_router, "_routes_for", return_value=routes), \
             patch.object(smart_router.bridge_state, "get_route_health", return_value={"is_available": False}):
            with self.assertRaises(smart_router.bridge_state.ProviderUnavailableError) as ctx:
                smart_router.ask_smart("hello")
            err_msg = str(ctx.exception)
            self.assertIn("bridge1/m1: cooling down", err_msg)
            self.assertIn("bridge2/m2: cooling down", err_msg)

    def test_preflight_with_large_and_complex_input(self):
        huge_prompt = "Explain quantum computing: " + ("A" * 100000)
        route_healthy = smart_router.Route("hf-bridge", "local-model", "local", None, lambda p, r: "huge_response")
        with patch.object(smart_router, "_routes_for", return_value=[route_healthy]), \
             patch.object(smart_router.bridge_state, "get_route_health", return_value={"is_available": True, "token_bucket_available": True}):
            res = smart_router.ask_smart(huge_prompt, "auto")
            self.assertEqual(res, "huge_response")

    def test_preflight_concurrency_under_contention(self):
        import threading
        primary_route = smart_router.Route("groq-bridge", "llama-3.3-70b-versatile", "free-cloud", None, lambda p, r: "ok")
        fallback_route = smart_router.Route("hf-bridge", "local-model", "local", None, lambda p, r: "fallback_ok")

        results = []
        errors = []

        def worker():
            try:
                res = smart_router._preflight_routes([primary_route, fallback_route])
                results.append(res)
            except Exception as e:
                errors.append(e)

        with patch.object(smart_router.bridge_state, "get_route_health", return_value={"is_available": True, "token_bucket_available": True}):
            threads = [threading.Thread(target=worker) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        self.assertEqual(len(errors), 0)
        self.assertEqual(len(results), 10)
        for viable, errs in results:
            self.assertEqual(viable, [primary_route, fallback_route])
            self.assertEqual(errs, [])


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

    @patch.dict("os.environ", {
        "GROQ_API_KEY": "gsk_test_key",
        "GEMINI_API_KEY": "gemini-key",
        "CEREBRAS_API_KEY": "cerebras-key",
    }, clear=True)
    def test_provider_capability_score_prioritizes_stronger_coding_route_within_free_tier(self):
        routes = smart_router._routes_for("coding")
        free_cloud_providers = [route.provider for route in routes if route.cost_tier == "free-cloud"]
        self.assertEqual(free_cloud_providers[:3], ["cerebras-bridge", "gemini-bridge", "groq-bridge"])

        score_by_provider = {
            route.provider: smart_router._route_capability_score(route, "coding")["total"]
            for route in routes
            if route.provider in ("cerebras-bridge", "gemini-bridge", "groq-bridge")
        }
        self.assertGreater(score_by_provider["cerebras-bridge"], score_by_provider["gemini-bridge"])
        self.assertGreater(score_by_provider["gemini-bridge"], score_by_provider["groq-bridge"])

    @patch.dict("os.environ", {"GROQ_API_KEY": "gsk_test_key", "GEMINI_API_KEY": "gemini-key"}, clear=True)
    def test_capability_score_includes_historical_latency_component(self):
        routes = smart_router._routes_for("auto")
        groq_route = next(route for route in routes if route.provider == "groq-bridge")
        gemini_route = next(route for route in routes if route.provider == "gemini-bridge")

        initial_groq_score = smart_router._route_capability_score(groq_route, "auto")["total"]
        initial_gemini_score = smart_router._route_capability_score(gemini_route, "auto")["total"]
        self.assertEqual(initial_groq_score, initial_gemini_score)

        smart_router.bridge_state.record_metric("groq-bridge", "llama-3.3-70b-versatile", latency=2.0, success=True)
        smart_router.bridge_state.record_metric("gemini-bridge", "gemini-2.5-flash", latency=0.5, success=True)

        groq_score = smart_router._route_capability_score(groq_route, "auto")
        gemini_score = smart_router._route_capability_score(gemini_route, "auto")
        self.assertGreater(gemini_score["latency"], groq_score["latency"])
        self.assertGreater(gemini_score["total"], groq_score["total"])

    def test_cost_efficiency_score_uses_requested_composite_formula(self):
        route = smart_router.Route(
            "test-bridge", "test-model", "free-cloud", None, MagicMock(),
            cost_weight=0.25,
        )
        health = {"success_rate": 0.8, "avg_latency": 1.0}

        with patch.dict("os.environ", {"SMART_ROUTER_LATENCY_THRESHOLD": "5"}):
            score = smart_router._route_cost_efficiency_score(route, health)

        # 0.8 * 0.5 + 0.9 * 0.3 - 0.25 * 0.2 = 0.62
        self.assertEqual(score, {
            "total": 0.62,
            "reliability": 0.8,
            "latency": 0.9,
            "cost_weight": 0.25,
        })

    def test_cost_efficiency_score_trades_reliability_for_latency(self):
        reliable_slow = smart_router.Route(
            "reliable-slow", "model", "free-cloud", None, MagicMock()
        )
        responsive = smart_router.Route(
            "responsive", "model", "free-cloud", None, MagicMock()
        )
        health_map = {
            "reliable-slow": {
                "is_available": True, "avg_latency": 10.0,
                "consecutive_failures": 0, "success_rate": 0.95,
                "is_soft_capped": False, "token_bucket_available": True,
                "provider_is_degraded": False,
            },
            "responsive": {
                "is_available": True, "avg_latency": 0.0,
                "consecutive_failures": 0, "success_rate": 0.70,
                "is_soft_capped": False, "token_bucket_available": True,
                "provider_is_degraded": False,
            },
        }

        with patch.dict("os.environ", {"SMART_ROUTER_LATENCY_THRESHOLD": "5"}), \
             patch.object(
                 smart_router.bridge_state,
                 "get_route_health",
                 side_effect=lambda provider, model: health_map[provider],
             ):
            ordered = smart_router._sort_routes_by_cost_and_health(
                [reliable_slow, responsive], "auto"
            )

        self.assertEqual([route.provider for route in ordered], ["responsive", "reliable-slow"])

    def test_cost_efficiency_score_prefers_lower_provider_cost_weight(self):
        expensive = smart_router.Route(
            "expensive", "model", "free-cloud", None, MagicMock(), cost_weight=0.8
        )
        inexpensive = smart_router.Route(
            "inexpensive", "model", "free-cloud", None, MagicMock(), cost_weight=0.1
        )
        health = {
            "is_available": True, "avg_latency": 1.0,
            "consecutive_failures": 0, "success_rate": 0.9,
            "is_soft_capped": False, "token_bucket_available": True,
            "provider_is_degraded": False,
        }

        with patch.object(
            smart_router.bridge_state, "get_route_health", return_value=health
        ):
            ordered = smart_router._sort_routes_by_cost_and_health(
                [expensive, inexpensive], "auto"
            )

        self.assertEqual([route.provider for route in ordered], ["inexpensive", "expensive"])

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
        # Verify that aliases map to their expected capability tags; their cost
        # policy may intentionally change ordering within that capability.
        coding_aliases = ("refactor", "bug_fix", "bugfix", "debug", "unit_test", "test", "code_review")
        for alias in coding_aliases:
            aliased_routes = smart_router._routes_for(alias)
            self.assertTrue(all("coding" in r.capabilities or r.cost_tier == "paid" for r in aliased_routes))
            # Test case insensitivity and whitespace trim
            mixed_alias = f"  {alias.upper()}  "
            mixed_routes = smart_router._routes_for(mixed_alias)
            self.assertEqual([r.provider for r in mixed_routes], [r.provider for r in aliased_routes])

        simple_ext_routes = smart_router._routes_for("simple_extraction")
        simple_ext_aliases = ("docs", "documentation")
        for alias in simple_ext_aliases:
            aliased_routes = smart_router._routes_for(alias)
            self.assertTrue(all("simple_extraction" in r.capabilities or r.cost_tier == "paid" for r in aliased_routes))
            # Test case insensitivity and whitespace trim
            mixed_alias = f" \t {alias.upper()} \n "
            mixed_routes = smart_router._routes_for(mixed_alias)
            self.assertEqual([r.provider for r in mixed_routes], [r.provider for r in aliased_routes])

        creative_routes = smart_router._routes_for("creative_writing")
        creative_aliases = ("story", "copywriting")
        for alias in creative_aliases:
            aliased_routes = smart_router._routes_for(alias)
            self.assertTrue(all("creative_writing" in r.capabilities or r.cost_tier == "paid" for r in aliased_routes))
            # Test case insensitivity and whitespace trim
            mixed_alias = f"\n{alias.upper()}\r"
            mixed_routes = smart_router._routes_for(mixed_alias)
            self.assertEqual([r.provider for r in mixed_routes], [r.provider for r in aliased_routes])

        # Ensure all returned routes for aliases contain the target capability
        for alias in coding_aliases:
            for r in smart_router._routes_for(alias):
                # If there are available capable free routes, the route list should be restricted
                # to only routes supporting coding (plus paid_routes since paid_fallback is enabled).
                self.assertTrue("coding" in r.capabilities or r.cost_tier == "paid")

        # Verify that cost prioritization behaves consistently for aliased tasks
        # Compare "refactor" vs "coding" - they must result in the same cost ordering
        refactor_routes = smart_router._routes_for("refactor")
        coding_routes_prio = smart_router._routes_for("coding")
        self.assertEqual({r.provider for r in refactor_routes}, {r.provider for r in coding_routes_prio})

    @patch.dict("os.environ", {
        "GROQ_API_KEY": "gsk_test_key",
        "GEMINI_API_KEY": "gemini-key",
        "OPENAI_API_KEY": "paid-key",
        "SMART_ROUTER_ALLOW_PAID_FALLBACK": "1",
    }, clear=True)
    def test_task_profile_aliases_are_canonicalized_before_sorting(self):
        original_sort = smart_router._sort_routes_by_cost_and_health
        sorted_task_types = []

        def capture_sort(routes, task_type, prompt=None):
            sorted_task_types.append(task_type)
            return original_sort(routes, task_type, prompt)

        with patch.object(smart_router, "_sort_routes_by_cost_and_health", side_effect=capture_sort):
            smart_router._routes_for("  REFACTOR  ", prompt="Refactor this module.")
            smart_router._routes_for("\tdocs\n", prompt="Summarize these docs.")
            smart_router._routes_for("Story", prompt="Write a launch story.")

        self.assertEqual(sorted_task_types, ["refactor", "docs", "story"])

        # Aliases retain canonical capability filtering but apply their profile
        # cost policy: lightweight work stays local, heavier coding uses free cloud.
        with patch.dict("os.environ", {"GROQ_API_KEY": "gsk_test_key"}, clear=True):
            self.assertEqual(smart_router._get_cost_priority("local", "test"), 0)
            self.assertEqual(smart_router._get_cost_priority("free-cloud", "refactor"), 0)
            self.assertLess(
                [r.cost_tier for r in smart_router._routes_for("docs")].index("local"),
                [r.cost_tier for r in smart_router._routes_for("docs")].index("free-cloud"),
            )

    @patch.dict("os.environ", {
        "OPENAI_API_KEY": "paid-key",
        "SMART_ROUTER_ALLOW_PAID_FALLBACK": "1",
    }, clear=True)
    def test_task_profile_aliases_match_canonical_fallback_when_capable_free_missing(self):
        # With no capable free provider configured, capability filtering should relax in the
        # same way for aliases and canonical task types instead of jumping straight to paid.
        alias_routes = smart_router._routes_for("story")
        canonical_routes = smart_router._routes_for("creative_writing")
        alias_providers = [r.provider for r in alias_routes]

        self.assertEqual(set(alias_providers), {r.provider for r in canonical_routes})
        self.assertIn("groq-bridge", alias_providers)
        self.assertLess(alias_providers.index("hf-bridge"), alias_providers.index("gpt-bridge"))

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

    @patch.dict("os.environ", {"SMART_ROUTER_MAX_CONTEXT_HF_BRIDGE": "8"}, clear=True)
    def test_route_context_limit_uses_provider_override(self):
        route = smart_router.Route("hf-bridge", "local-model", "local", None, MagicMock())
        self.assertEqual(smart_router.get_route_max_context_tokens(route), 8)

    @patch.dict("os.environ", {
        "GROQ_API_KEY": "gsk_test_key",
        "OPENAI_API_KEY": "paid-key",
        "SMART_ROUTER_ALLOW_PAID_FALLBACK": "1",
        "SMART_ROUTER_MAX_CONTEXT_HF_BRIDGE": "8",
    }, clear=True)
    def test_truncated_response_does_not_call_context_overflow_route(self):
        MockOpenAI.truncated_models = {"llama-3.3-70b-versatile"}
        result = smart_router.ask_smart("write something long", "auto")
        self.assertIn("gpt-4o-mini: Continue the answer", result)
        self.assertEqual([call[1] for call in MockOpenAI.calls], [
            "llama-3.3-70b-versatile", "gpt-4o-mini",
        ])

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
        first_route = data["routes"][0]
        self.assertIn("capability_score", first_route)
        self.assertIn("cost_efficiency_score", first_route)
        self.assertEqual(
            set(first_route["cost_efficiency_components"]),
            {"reliability", "latency", "cost_weight"},
        )
        self.assertEqual(
            data["cost_efficiency_weights"],
            {"reliability": 0.5, "latency": 0.3, "cost_weight": 0.2},
        )
        self.assertIn("score_components", first_route)
        self.assertEqual(
            set(first_route["score_components"]),
            {"cost", "latency", "capability"},
        )

    def test_bridges_status_resource_reads_status_cache(self):
        import json
        import time

        cache_path = Path(smart_router.bridge_state.STATUS_CACHE_FILE)
        cache_data = {
            "timestamp": time.time(),
            "bridges": {
                "groq-bridge": {
                    "id": "groq-bridge",
                    "status": "🟢 Online",
                    "latency": "12 ms",
                    "details": "Cache hit",
                }
            },
        }

        if cache_path.exists():
            try:
                cache_path.unlink()
            except OSError:
                pass

        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_data, f)

            res = smart_router.mcp.resources["get_bridges_status"]()
            data = json.loads(res)
            self.assertEqual(data["bridges"]["groq-bridge"]["status"], "🟢 Online")
            self.assertEqual(data["bridges"]["groq-bridge"]["details"], "Cache hit")
        finally:
            if cache_path.exists():
                try:
                    cache_path.unlink()
                except OSError:
                    pass

    def test_smart_router_soft_cap_routing_priority(self):
        import os
        import usage_tracker

        if os.path.exists(usage_tracker.USAGE_FILE_PATH):
            try:
                os.remove(usage_tracker.USAGE_FILE_PATH)
            except OSError:
                pass

        with patch.dict("os.environ", {
            "GROQ_API_KEY": "gsk_test_key",
            "GEMINI_API_KEY": "gemini-key",
            "PROVIDER_DAILY_TOKEN_BUDGET_GROQ_BRIDGE": "100",
            "PROVIDER_SOFT_CAP_RATIO_GROQ_BRIDGE": "0.5"
        }):
            usage_tracker.record_usage("groq-bridge", "llama-3.3-70b-versatile", 60, 0)

            status = usage_tracker.check_provider_budget("groq-bridge")
            self.assertTrue(status["is_soft_capped"])
            self.assertFalse(status["is_exceeded"])

            health = smart_router.bridge_state.get_route_health("groq-bridge", "llama-3.3-70b-versatile")
            self.assertTrue(health["is_soft_capped"])
            self.assertTrue(health["is_available"])

            routes = [
                smart_router.Route(
                    "groq-bridge",
                    "llama-3.3-70b-versatile",
                    "free-cloud",
                    None,
                    lambda *args, **kwargs: "soft",
                    ("general",),
                    {"general": 0.72},
                ),
                smart_router.Route(
                    "gemini-bridge",
                    "gemini-2.5-flash",
                    "free-cloud",
                    None,
                    lambda *args, **kwargs: "healthy",
                    ("general",),
                    {"general": 0.72},
                ),
            ]
            health_map = {
                "groq-bridge": {
                    "is_available": True,
                    "is_soft_capped": True,
                    "provider_is_degraded": False,
                    "consecutive_failures": 0,
                    "success_rate": 1.0,
                    "avg_latency": 0.1,
                },
                "gemini-bridge": {
                    "is_available": True,
                    "is_soft_capped": False,
                    "provider_is_degraded": False,
                    "consecutive_failures": 0,
                    "success_rate": 1.0,
                    "avg_latency": 0.1,
                },
            }
            with patch.object(smart_router.bridge_state, "get_route_health", side_effect=lambda provider, model: health_map[provider]):
                ordered_auto = smart_router._sort_routes_by_cost_and_health(routes, "auto", prompt="summarize this text")
                self.assertEqual([route.provider for route in ordered_auto], ["gemini-bridge", "groq-bridge"])

                ordered_paid = smart_router._sort_routes_by_cost_and_health(routes, "paid", prompt="summarize this text")
                self.assertEqual([route.provider for route in ordered_paid], ["groq-bridge", "gemini-bridge"])

                score_non_critical = smart_router._route_capability_score(
                    routes[0],
                    "auto",
                    "summarize this text",
                )
                score_critical = smart_router._route_capability_score(
                    routes[0],
                    "paid",
                    "summarize this text",
                )
                self.assertAlmostEqual(score_critical["total"], round(score_non_critical["total"] * 2, 4))

    def test_latency_heatmap_generation(self):
        smart_router.bridge_state.record_metric("groq-bridge", "llama-3.3-70b-versatile", latency=0.2, success=True)
        smart_router.bridge_state.record_metric("gemini-bridge", "gemini-2.5-flash", latency=2.5, success=True)
        heatmap = smart_router.bridge_state.get_latency_heatmap()
        self.assertIn("groq-bridge", heatmap)
        self.assertIn("gemini-bridge", heatmap)
        self.assertEqual(heatmap["groq-bridge"]["llama-3.3-70b-versatile"]["latency_tier"], "fast")
        self.assertEqual(heatmap["gemini-bridge"]["gemini-2.5-flash"]["latency_tier"], "moderate")

    def test_fast_and_reliable_routing_profile_weightings(self):
        fast_weights = smart_router._get_routing_weights("fast")
        reliable_weights = smart_router._get_routing_weights("reliable")
        balanced_weights = smart_router._get_routing_weights("auto")

        self.assertEqual(fast_weights["latency"], 0.50)
        self.assertEqual(reliable_weights["capability"], 0.50)
        self.assertEqual(balanced_weights["latency"], 0.25)

    def test_qos_profile_classifies_throttled_before_degraded(self):
        self.assertEqual(smart_router.bridge_state.qos_profile({}), "Optimal")
        self.assertEqual(
            smart_router.bridge_state.qos_profile({"provider_is_degraded": True}),
            "Degraded",
        )
        self.assertEqual(
            smart_router.bridge_state.qos_profile({
                "provider_is_degraded": True,
                "token_bucket_available": False,
            }),
            "Throttled",
        )

    def test_smart_router_status_includes_qos_profile(self):
        import json
        status = json.loads(smart_router.mcp.resources["get_smart_router_status"]())
        self.assertIn(status["routes"][0]["qos_profile"], {"Optimal", "Degraded", "Throttled"})

    def test_get_smart_router_status_includes_heatmap_and_profiles(self):
        import json
        status_raw = smart_router.mcp.resources["get_smart_router_status"]()
        status = json.loads(status_raw)
        self.assertIn("latency_heatmap", status)
        self.assertIn("routing_profiles", status)
        self.assertIn("fast", status["routing_profiles"])

    def test_speculative_prewarm_routes(self):
        mock_route = smart_router.Route("hf-bridge", "local-model", "local", None, MagicMock(return_value="OK"))
        with patch.object(smart_router, "_library_available", return_value=True), \
             patch.object(smart_router, "_env_available", return_value=True), \
             patch.object(smart_router.bridge_state, "is_available", return_value=True), \
             patch.object(smart_router.bridge_state, "record_metric") as mock_record:
            smart_router._speculative_prewarm_routes([mock_route])
            import time
            time.sleep(0.1)
            mock_route.ask.assert_called_with(smart_router.HEARTBEAT_PROMPT, mock_route)
            mock_record.assert_called()

    def test_prewarm_high_priority_bridges_mcp_tool(self):
        mock_route = smart_router.Route("hf-bridge", "local-model", "local", None, MagicMock(return_value="OK"))
        with patch.object(smart_router, "_routes_for", return_value=[mock_route]), \
             patch.object(smart_router, "_library_available", return_value=True), \
             patch.object(smart_router, "_env_available", return_value=True), \
             patch.object(smart_router.bridge_state, "is_available", return_value=True):
            res = smart_router.prewarm_high_priority_bridges("coding")
            self.assertEqual(res["task_type"], "coding")
            self.assertEqual(len(res["warmed_routes"]), 1)
            self.assertEqual(res["warmed_routes"][0]["provider"], "hf-bridge")
            self.assertEqual(res["warmed_routes"][0]["status"], "warmed")

    def test_ask_smart_triggers_speculative_prewarm_on_high_priority_task(self):
        route1 = smart_router.Route("groq-bridge", "m1", "free-cloud", None, MagicMock(return_value="primary_res"))
        route2 = smart_router.Route("hf-bridge", "m2", "local", None, MagicMock(return_value="OK"))
        with patch.object(smart_router, "_routes_for", return_value=[route1, route2]), \
             patch.object(smart_router, "_library_available", return_value=True), \
             patch.object(smart_router, "_env_available", return_value=True), \
             patch.object(smart_router.bridge_state, "is_available", return_value=True), \
             patch.object(smart_router, "_speculative_prewarm_routes") as mock_prewarm:
            res = smart_router.ask_smart("refactor this function", "coding")
            self.assertEqual(res, "primary_res")
            mock_prewarm.assert_called_once_with([route2])

    def test_task_profile_aliases_hyphenated_variants_and_custom_capability_scores(self):
        # Verify hyphenated aliases map identically to underscore variants
        for hyphenated, underscore in [
            ("unit-test", "unit_test"),
            ("code-review", "code_review"),
            ("bug-fix", "bug_fix"),
            ("high-reliability", "high_reliability"),
        ]:
            self.assertEqual(
                smart_router.TASK_PROFILE_ALIASES.get(hyphenated),
                smart_router.TASK_PROFILE_ALIASES.get(underscore)
            )
            self.assertEqual(
                smart_router._get_cost_priority("local", hyphenated),
                smart_router._get_cost_priority("local", underscore)
            )

        # Verify route with custom raw capability score takes precedence over generic alias score
        custom_route = smart_router.Route(
            "custom-bridge", "custom-model", "free-cloud", None, MagicMock(),
            capabilities=("unit-test",),
            capability_scores={"unit-test": 0.95, "coding": 0.70}
        )
        fit_raw = smart_router._route_capability_fit(custom_route, "unit-test")
        self.assertEqual(fit_raw, 0.95)

        # Verify route with raw capability tag without alias score still matches
        tag_route = smart_router.Route(
            "tag-bridge", "tag-model", "free-cloud", None, MagicMock(),
            capabilities=("unit-test",)
        )
        fit_tag = smart_router._route_capability_fit(tag_route, "unit-test")
        self.assertEqual(fit_tag, 0.75)


class TestTaskProfileAliasesAdversarial(unittest.TestCase):
    """Adversarial tests probing edge cases, boundaries, and routing behavior of TASK_PROFILE_ALIASES."""

    def test_task_profile_aliases_mapping_completeness(self):
        """Pin intended alias mappings to canonical capability tags or routing profile weights."""
        canonical_tags = {"coding", "simple_extraction", "creative_writing", "fast", "reliable"}
        for alias, canonical in smart_router.TASK_PROFILE_ALIASES.items():
            self.assertIn(canonical, canonical_tags, f"Alias '{alias}' maps to unexpected target '{canonical}'")
            self.assertTrue(len(alias.strip()) > 0)

        # Every entry in HIGH_PRIORITY_TASK_TYPES must be valid and non-empty
        for task_type in smart_router.HIGH_PRIORITY_TASK_TYPES:
            self.assertTrue(len(task_type) > 0)
            alias = smart_router.TASK_PROFILE_ALIASES.get(task_type, task_type)
            self.assertIn(alias, {"coding", "reliable"})

    def test_edge_case_and_whitespace_task_type_inputs(self):
        """Probe empty, missing, whitespace, mixed-case, and malformed task_type inputs."""
        edge_inputs = [None, "", "   ", "\t\n", "  Refactor  ", "BUG_FIX", "  Paid  ", "  Local  ", "FAST"]

        for task_type in edge_inputs:
            # 1. Routing weights should not throw
            weights = smart_router._get_routing_weights(task_type)
            self.assertIn("cost", weights)
            self.assertIn("latency", weights)
            self.assertIn("capability", weights)

            # 2. Cost priority should return valid integer
            prio = smart_router._get_cost_priority("free-cloud", task_type)
            self.assertIsInstance(prio, int)

            # 3. Capability fit on a dummy route should return float in [0.0, 1.0]
            dummy_route = smart_router.Route("groq-bridge", "m1", "free-cloud", None, MagicMock(), ("coding",))
            fit = smart_router._route_capability_fit(dummy_route, task_type)
            self.assertGreaterEqual(fit, 0.0)
            self.assertLessEqual(fit, 1.0)

            # 4. _routes_for should complete and return non-empty route list
            routes = smart_router._routes_for(task_type, "test prompt")
            self.assertGreater(len(routes), 0)

    def test_cost_priority_specialization_vs_capability_alias(self):
        """Verify that task types like unit_test use specialized cost priority while aliasing capability to coding."""
        # unit_test capability aliases to coding, but cost priority prefers local first
        unit_test_local = smart_router._get_cost_priority("local", "unit_test")
        unit_test_free = smart_router._get_cost_priority("free-cloud", "unit_test")
        unit_test_paid = smart_router._get_cost_priority("paid", "unit_test")
        self.assertEqual((unit_test_local, unit_test_free, unit_test_paid), (0, 1, 2))

        # generic coding task has no cost priority override, defaulting free-cloud first for medium complexity
        coding_free = smart_router._get_cost_priority("free-cloud", "coding")
        coding_local = smart_router._get_cost_priority("local", "coding")
        coding_paid = smart_router._get_cost_priority("paid", "coding")
        self.assertEqual((coding_free, coding_local, coding_paid), (0, 1, 2))

        # Verify route ordering reflects local priority for unit-test vs free-cloud priority for generic coding
        local_route = smart_router.Route("local-b", "m1", "local", None, MagicMock(), capabilities=("coding",))
        free_route = smart_router.Route("free-b", "m2", "free-cloud", None, MagicMock(), capabilities=("coding",))

        with patch.object(smart_router, "_routes_for") as mock_rf, \
             patch.object(smart_router, "_library_available", return_value=True), \
             patch.object(smart_router, "_env_available", return_value=True), \
             patch.object(smart_router.bridge_state, "is_available", return_value=True), \
             patch.object(smart_router.bridge_state, "get_route_health", return_value={
                 "is_available": True, "avg_latency": 0.5, "consecutive_failures": 0,
                 "success_rate": 1.0, "is_soft_capped": False, "token_bucket_available": True
             }):

            sorted_unit_test = smart_router._sort_routes_by_cost_and_health([free_route, local_route], "unit-test")
            self.assertEqual(sorted_unit_test[0].cost_tier, "local")

            sorted_coding = smart_router._sort_routes_by_cost_and_health([free_route, local_route], "coding")
            self.assertEqual(sorted_coding[0].cost_tier, "free-cloud")

    def test_raw_score_overrides_alias_score_and_clamping(self):
        """Verify raw capability score takes precedence over alias score, and scores are clamped to [0,1]."""
        route_custom = smart_router.Route(
            "custom", "m", "free-cloud", None, MagicMock(),
            capabilities=("coding", "bug-fix"),
            capability_scores={"bug-fix": 0.98, "coding": 0.65}
        )
        # Specific raw task_type 'bug-fix' gets 0.98
        self.assertEqual(smart_router._route_capability_fit(route_custom, "bug-fix"), 0.98)
        # Generic task_type 'coding' gets 0.65
        self.assertEqual(smart_router._route_capability_fit(route_custom, "coding"), 0.65)

        # Test score clamping for out-of-bound raw capability scores
        route_clamped = smart_router.Route(
            "clamped", "m", "free-cloud", None, MagicMock(),
            capability_scores={"out_of_bounds_high": 1.75, "out_of_bounds_low": -0.50}
        )
        self.assertEqual(smart_router._route_capability_fit(route_clamped, "out_of_bounds_high"), 1.0)
        self.assertEqual(smart_router._route_capability_fit(route_clamped, "out_of_bounds_low"), 0.0)

    def test_malformed_and_unknown_task_profile_types(self):
        """Verify graceful fallback for completely unknown task profile strings."""
        unknown_inputs = ["unknown_task_type_xyz", "!!!invalid_chars!!!", "a" * 1000]

        for unknown in unknown_inputs:
            # Routing weights fall back to balanced
            weights = smart_router._get_routing_weights(unknown)
            self.assertEqual(weights, smart_router.ROUTING_PROFILE_WEIGHTS["balanced"])

            # Cost priority falls back to standard medium complexity mapping (free-cloud priority = 0)
            cost_prio = smart_router._get_cost_priority("free-cloud", unknown, "simple prompt")
            self.assertEqual(cost_prio, 0)

            # Capability fit falls back to general capability score
            dummy_route = smart_router.Route("r", "m", "free-cloud", None, MagicMock(), capability_scores={"general": 0.62})
            fit = smart_router._route_capability_fit(dummy_route, unknown)
            self.assertEqual(fit, 0.62)

    def test_soft_cap_criticality_with_aliases(self):
        """Verify soft-cap penalty logic correctly recognizes critical tasks via raw and alias names."""
        soft_capped_route = smart_router.Route("groq-bridge", "m", "free-cloud", None, MagicMock())

        with patch.object(smart_router.bridge_state, "get_route_health", return_value={
            "is_available": True, "avg_latency": 0.1, "consecutive_failures": 0,
            "success_rate": 1.0, "is_soft_capped": True, "token_bucket_available": True
        }):
            # Non-critical task ("docs", aliased to "simple_extraction") gets soft-cap penalty (total score halved)
            score_non_critical = smart_router._route_capability_score(soft_capped_route, "docs", "simple prompt")

            # Critical task ("paid", or complex prompt) bypasses soft-cap penalty
            score_critical = smart_router._route_capability_score(soft_capped_route, "paid", "simple prompt")
            score_complex = smart_router._route_capability_score(soft_capped_route, "docs", "refactor and architecture plan " * 30)

            self.assertGreater(score_critical["total"], score_non_critical["total"])
            self.assertGreater(score_complex["total"], score_non_critical["total"])

    def test_prewarm_high_priority_aliased_tasks(self):
        """Verify speculative prewarming triggers correctly for aliased high-priority tasks."""
        route1 = smart_router.Route("groq-bridge", "m1", "free-cloud", None, MagicMock(return_value="res"))
        route2 = smart_router.Route("hf-bridge", "m2", "local", None, MagicMock(return_value="OK"))

        for high_prio_alias in ["unit-test", "bugfix", "code-review", "high-reliability"]:
            with patch.object(smart_router, "_routes_for", return_value=[route1, route2]), \
                 patch.object(smart_router, "_library_available", return_value=True), \
                 patch.object(smart_router, "_env_available", return_value=True), \
                 patch.object(smart_router.bridge_state, "is_available", return_value=True):

                res = smart_router.prewarm_high_priority_bridges(high_prio_alias)
                self.assertEqual(res["task_type"], high_prio_alias)
                self.assertEqual(len(res["warmed_routes"]), 2)
                self.assertEqual(res["warmed_routes"][0]["provider"], "groq-bridge")
                self.assertEqual(res["warmed_routes"][1]["provider"], "hf-bridge")


if __name__ == "__main__":
    unittest.main()





