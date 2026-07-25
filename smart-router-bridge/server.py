import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Mapping

from mcp.server.fastmcp import FastMCP

try:
    import openai
    from openai import OpenAI
except Exception:
    openai = None
    OpenAI = None

try:
    from google import genai
except Exception:
    genai = None

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bridge_state
import usage_tracker


mcp = FastMCP("smart-router-bridge")
PROVIDER = "smart-router-bridge"
HEARTBEAT_PROMPT = "Reply with OK only."
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 60.0
_heartbeat_thread = None
_heartbeat_stop = None


@dataclass(frozen=True)
class Route:
    provider: str
    model: str
    cost_tier: str
    required_env: str | None
    ask: Callable[..., str]
    capabilities: tuple[str, ...] = ()
    capability_scores: Mapping[str, float] | None = None


# Capability tags per route, used to filter candidates for capability-aware
# task types (e.g. task_type="coding") before applying cost/health ordering.
CAPABILITY_TASK_TYPES = {"coding", "creative_writing", "simple_extraction"}
SIMPLE_INTENT_KEYWORDS = (
    "summary",
    "summarize",
    "extract",
    "bullet",
    "rewrite",
    "translate",
    "proofread",
    "typo",
    "title",
    "format",
    "brief",
)
COMPLEX_INTENT_KEYWORDS = (
    "refactor",
    "architecture",
    "design",
    "debug",
    "implement",
    "migrate",
    "optimize",
    "build",
    "plan",
    "analyze",
    "comparison",
    "multi-step",
)

# Maps caller-facing task profile names to the canonical capability tags
# already defined on each Route, so requests like task_type="unit_test" get
# routed through the same capability-aware filtering as task_type="coding".
TASK_PROFILE_ALIASES = {
    "refactor": "coding",
    "bug_fix": "coding",
    "bugfix": "coding",
    "debug": "coding",
    "unit_test": "coding",
    "test": "coding",
    "code_review": "coding",
    "docs": "simple_extraction",
    "documentation": "simple_extraction",
    "story": "creative_writing",
    "copywriting": "creative_writing",
    "fast": "fast",
    "quick": "fast",
    "reliable": "reliable",
    "high_reliability": "reliable",
}

HIGH_PRIORITY_TASK_TYPES = {
    "coding",
    "refactor",
    "bug_fix",
    "bugfix",
    "debug",
    "unit_test",
    "test",
    "code_review",
    "reliable",
    "high_reliability",
}

# Keep capability filtering canonical, while allowing caller-facing profiles
# to express a useful cost preference within that capability.
TASK_PROFILE_COST_PRIORITIES = {
    "unit_test": ("local", "free-cloud", "paid"),
    "test": ("local", "free-cloud", "paid"),
    "docs": ("local", "free-cloud", "paid"),
    "documentation": ("local", "free-cloud", "paid"),
    "story": ("local", "free-cloud", "paid"),
    "copywriting": ("local", "free-cloud", "paid"),
    "refactor": ("free-cloud", "local", "paid"),
    "bug_fix": ("free-cloud", "local", "paid"),
    "bugfix": ("free-cloud", "local", "paid"),
    "debug": ("free-cloud", "local", "paid"),
    "code_review": ("free-cloud", "local", "paid"),
    "fast": ("free-cloud", "local", "paid"),
    "quick": ("free-cloud", "local", "paid"),
    "reliable": ("free-cloud", "local", "paid"),
    "high_reliability": ("free-cloud", "local", "paid"),
}

ROUTING_PROFILE_WEIGHTS = {
    "fast": {"cost": 0.25, "latency": 0.50, "capability": 0.25},
    "quick": {"cost": 0.25, "latency": 0.50, "capability": 0.25},
    "reliable": {"cost": 0.35, "latency": 0.15, "capability": 0.50},
    "high_reliability": {"cost": 0.35, "latency": 0.15, "capability": 0.50},
    "balanced": {"cost": 0.45, "latency": 0.25, "capability": 0.30},
}

CAPABILITY_SCORE_WEIGHTS = ROUTING_PROFILE_WEIGHTS["balanced"]


def _get_routing_weights(task_type: str) -> dict:
    normalized = (task_type or "auto").lower().strip()
    alias = TASK_PROFILE_ALIASES.get(normalized, normalized)
    if alias in ROUTING_PROFILE_WEIGHTS:
        return ROUTING_PROFILE_WEIGHTS[alias]
    if normalized in ROUTING_PROFILE_WEIGHTS:
        return ROUTING_PROFILE_WEIGHTS[normalized]
    return ROUTING_PROFILE_WEIGHTS["balanced"]


DEFAULT_CAPABILITY_SCORE = 0.55


def _openai_client(api_key_env: str, base_url: str | None = None, default_headers: dict | None = None):
    if OpenAI is None:
        raise RuntimeError("openai package is not installed")
    api_key = os.environ.get(api_key_env)
    if api_key_env == "OLLAMA_API_KEY":
        api_key = api_key or "ollama"
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        default_headers=default_headers,
    )


class TruncatedResponseError(Exception):
    """Raised when a free/local route's response was cut off by a token limit.

    Carries the partial content so the caller can still use it if every
    subsequent route also fails to produce a complete answer.
    """

    def __init__(self, partial_content: str):
        super().__init__("response truncated due to token limit")
        self.partial_content = partial_content


def _continuation_prompt(partial: str) -> str:
    """Ask the next route for only the missing tail of a truncated answer."""
    return (
        "Continue the answer below from exactly where it stops. Return only the "
        "missing continuation; do not repeat any existing text.\n\n"
        f"Existing answer:\n{partial}"
    )


def _merge_continuation(partial: str, continuation: str) -> str:
    """Join a continuation while removing a repeated boundary."""
    if not partial:
        return continuation
    if not continuation:
        return partial
    if continuation.startswith(partial):
        return continuation
    limit = min(len(partial), len(continuation), 2000)
    for size in range(limit, 0, -1):
        if partial[-size:] == continuation[:size]:
            return partial + continuation[size:]
    return partial + continuation


def _ask_openai_compatible(prompt: str, route: Route, base_url: str | None = None, api_key_env: str = "OPENAI_API_KEY",
                           default_headers: dict | None = None, max_tokens: int | None = None) -> str:
    client = _openai_client(api_key_env, base_url, default_headers)
    kwargs = {
        "model": route.model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    response = client.chat.completions.create(**kwargs)
    usage = getattr(response, "usage", None)
    if usage is not None:
        usage_tracker.record_usage(
            provider=route.provider,
            model=route.model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0),
            completion_tokens=getattr(usage, "completion_tokens", 0),
        )
    choice = response.choices[0]
    content = choice.message.content
    finish_reason = getattr(choice, "finish_reason", None)
    if finish_reason == "length" and route.cost_tier != "paid":
        raise TruncatedResponseError(content)
    return content


def _ask_groq(prompt: str, route: Route) -> str:
    return _ask_openai_compatible(prompt, route, "https://api.groq.com/openai/v1", "GROQ_API_KEY")


def _ask_cerebras(prompt: str, route: Route) -> str:
    return _ask_openai_compatible(prompt, route, "https://api.cerebras.ai/v1", "CEREBRAS_API_KEY")


def _ask_openrouter(prompt: str, route: Route) -> str:
    return _ask_openai_compatible(
        prompt,
        route,
        "https://openrouter.ai/api/v1",
        "OPENROUTER_API_KEY",
        {
            "HTTP-Referer": os.environ.get("OPENROUTER_REFERER", "https://github.com/yourusername/ai-bridges"),
            "X-Title": os.environ.get("OPENROUTER_TITLE", "AI Bridges for Claude Code"),
        },
    )


def _ask_hf(prompt: str, route: Route) -> str:
    if "/" in route.model:
        return _ask_openai_compatible(prompt, route, "https://router.huggingface.co/v1", "HF_TOKEN", max_tokens=4096)
    return _ask_openai_compatible(prompt, route, "http://localhost:11434/v1", "OLLAMA_API_KEY", max_tokens=4096)


def _ask_gpt(prompt: str, route: Route) -> str:
    usage_tracker.check_budget(route.provider, route.model)
    return _ask_openai_compatible(prompt, route, api_key_env="OPENAI_API_KEY")


def _ask_gemini(prompt: str, route: Route) -> str:
    if genai is None:
        raise RuntimeError("google-genai is not installed")
    response = genai.Client(api_key=os.environ.get("GEMINI_API_KEY")).models.generate_content(
        model=route.model,
        contents=prompt,
    )
    return response.text


def _task_complexity(prompt: str) -> str:
    normalized = (prompt or "").lower().strip()
    word_count = len(normalized.split())
    if not normalized:
        return "medium"
    if word_count >= 180 or any(keyword in normalized for keyword in COMPLEX_INTENT_KEYWORDS):
        return "complex"
    if word_count <= 40 and any(keyword in normalized for keyword in SIMPLE_INTENT_KEYWORDS):
        return "simple"
    return "medium"


def _get_cost_priority(cost_tier: str, task_type: str, prompt: str | None = None) -> int:
    normalized = (task_type or "auto").lower().strip()
    profile_priority = TASK_PROFILE_COST_PRIORITIES.get(normalized)
    if profile_priority:
        return profile_priority.index(cost_tier) if cost_tier in profile_priority else 99
    complexity = _task_complexity(prompt or "") if normalized == "auto" else "medium"
    if normalized in ("paid", "openai", "gpt"):
        mapping = {
            "paid": 0,
            "free-cloud": 1,
            "local": 2
        }
    elif normalized in ("local", "offline", "private"):
        mapping = {
            "local": 0,
            "free-cloud": 1,
            "paid": 2
        }
    else:
        if complexity == "simple":
            mapping = {
                "local": 0,
                "free-cloud": 1,
                "paid": 2,
            }
        else:
            mapping = {
                "free-cloud": 0,
                "local": 1,
                "paid": 2,
            }
    return mapping.get(cost_tier, 99)


def _score_from_cost_priority(cost_priority: int) -> float:
    if cost_priority <= 0:
        return 1.0
    if cost_priority == 1:
        return 0.65
    if cost_priority == 2:
        return 0.30
    return 0.0


def _score_from_latency(avg_latency: float) -> float:
    if avg_latency >= 9999.0:
        return 0.50
    if avg_latency <= 0:
        return 1.0
    latency_threshold = float(os.environ.get(
        "SMART_ROUTER_LATENCY_THRESHOLD",
        str(bridge_state.DEFAULT_LATENCY_THRESHOLD_SECONDS),
    ))
    # Treat the latency breaker threshold as the lower edge of "slow". Routes
    # at or above 2x that threshold have no latency advantage, while faster
    # routes scale linearly up to 1.0.
    slow_edge = max(latency_threshold * 2, 0.001)
    return max(0.0, min(1.0, 1.0 - (avg_latency / slow_edge)))


def _route_capability_fit(route: Route, task_type: str) -> float:
    normalized = TASK_PROFILE_ALIASES.get((task_type or "auto").lower().strip(), (task_type or "auto").lower().strip())
    scores = route.capability_scores or {}
    if normalized in scores:
        return max(0.0, min(1.0, float(scores[normalized])))
    if normalized in route.capabilities:
        return 0.75
    if normalized in ("auto", "simple", "fast", "quick", "paid", "openai", "gpt", "local", "offline", "private"):
        return max(0.0, min(1.0, float(scores.get("general", DEFAULT_CAPABILITY_SCORE))))
    return max(0.0, min(1.0, float(scores.get("general", DEFAULT_CAPABILITY_SCORE))))


def _route_capability_score(route: Route, task_type: str, prompt: str | None = None) -> dict:
    health = bridge_state.get_route_health(route.provider, route.model)
    cost_priority = _get_cost_priority(route.cost_tier, task_type, prompt)
    cost_score = _score_from_cost_priority(cost_priority)
    latency_score = _score_from_latency(health["avg_latency"])
    capability_score = _route_capability_fit(route, task_type)
    weights = _get_routing_weights(task_type)
    total = (
        cost_score * weights["cost"]
        + latency_score * weights["latency"]
        + capability_score * weights["capability"]
    )
    if not health["is_available"]:
        total *= 0.25
    if health.get("provider_is_degraded", False):
        total *= 0.75

    # Penalize soft-capped providers for non-critical tasks
    normalized = TASK_PROFILE_ALIASES.get((task_type or "auto").lower().strip(), (task_type or "auto").lower().strip())
    is_critical = normalized in ("paid", "openai", "gpt") or _task_complexity(prompt or "") == "complex"
    if health.get("is_soft_capped", False) and not is_critical:
        total *= 0.50

    if not health.get("token_bucket_available", True):
        total *= 0.50

    total *= health["success_rate"]
    return {
        "total": round(total, 4),
        "cost": round(cost_score, 4),
        "latency": round(latency_score, 4),
        "capability": round(capability_score, 4),
    }


def _sort_routes_by_cost_and_health(routes: List[Route], task_type: str, prompt: str | None = None) -> List[Route]:
    def sort_key(route):
        health = bridge_state.get_route_health(route.provider, route.model)
        score = _route_capability_score(route, task_type, prompt)["total"]
        # 1. Availability status (available first, i.e. 0 for available, 1 for cooling down/exceeded budget)
        is_avail_val = 0 if health["is_available"] else 1

        # Soft-cap penalty: if provider is soft-capped and this is a non-critical task, penalize it by pushing it after non-soft-capped ones.
        normalized = TASK_PROFILE_ALIASES.get((task_type or "auto").lower().strip(), (task_type or "auto").lower().strip())
        is_critical = normalized in ("paid", "openai", "gpt") or _task_complexity(prompt or "") == "complex"
        soft_cap_prio = 1 if (health.get("is_soft_capped", False) and not is_critical) else 0

        # 2. Cost tier priority (lower value first)
        cost_prio = _get_cost_priority(route.cost_tier, task_type, prompt)
        # Token bucket status (un-throttled first)
        is_throttled = 0 if health.get("token_bucket_available", True) else 1
        # 3. Provider degradation status (healthy first, i.e. 0 for healthy, 1 for degraded)
        is_degraded = 1 if health.get("provider_is_degraded", False) else 0
        # 4. Health: consecutive failures count (lower failures first)
        failures = health["consecutive_failures"]
        # 5. Health: success rate (higher rate first -> negative rate)
        neg_success_rate = -health["success_rate"]
        # 6. Weighted capability score (higher score first)
        neg_score = -score
        # 7. Latency (lower latency first)
        latency = health["avg_latency"]

        return (is_avail_val, soft_cap_prio, cost_prio, is_throttled, is_degraded, failures, neg_success_rate, neg_score, latency)

    return sorted(routes, key=sort_key)



def _routes_for(task_type: str, prompt: str | None = None) -> List[Route]:
    requested = (task_type or "auto").lower().strip()
    normalized = TASK_PROFILE_ALIASES.get(requested, requested)
    simple_model = "llama-3.1-8b-instant" if normalized in ("simple", "fast", "quick") else "llama-3.3-70b-versatile"
    local_model = os.environ.get("SMART_ROUTER_LOCAL_MODEL", "gemma4:latest")
    paid_model = os.environ.get("SMART_ROUTER_PAID_MODEL", "gpt-4o-mini")

    free_routes = [
        Route("groq-bridge", simple_model, "free-cloud", "GROQ_API_KEY", _ask_groq,
              ("coding", "general", "simple_extraction"),
              {"general": 0.72, "coding": 0.70, "simple_extraction": 0.84}),
        Route("gemini-bridge", "gemini-2.5-flash", "free-cloud", "GEMINI_API_KEY", _ask_gemini,
              ("coding", "creative_writing", "general"),
              {"general": 0.72, "coding": 0.76, "creative_writing": 0.86}),
        Route("cerebras-bridge", "gpt-oss-120b", "free-cloud", "CEREBRAS_API_KEY", _ask_cerebras,
              ("coding", "general"),
              {"general": 0.72, "coding": 0.82}),
        Route("openrouter-bridge", "nvidia/nemotron-3-super-120b-a12b:free", "free-cloud", "OPENROUTER_API_KEY", _ask_openrouter,
              ("general", "simple_extraction"),
              {"general": 0.70, "simple_extraction": 0.78}),
        Route("hf-bridge", local_model, "local", None, _ask_hf,
              ("general", "simple_extraction"),
              {"general": 0.58, "simple_extraction": 0.68}),
    ]
    paid_routes = [Route("gpt-bridge", paid_model, "paid", "OPENAI_API_KEY", _ask_gpt,
                          ("coding", "creative_writing", "general", "simple_extraction"),
                          {"general": 0.86, "coding": 0.90, "creative_writing": 0.88, "simple_extraction": 0.86})]

    paid_requested = normalized in ("paid", "openai", "gpt")
    allow_paid_fallback = os.environ.get("SMART_ROUTER_ALLOW_PAID_FALLBACK", "").lower() in ("1", "true", "yes", "on")

    if paid_requested:
        routes = paid_routes + free_routes
    elif normalized in ("local", "offline", "private"):
        routes = [free_routes[-1]] + free_routes[:-1]
        if allow_paid_fallback:
            routes += paid_routes
    else:
        routes = free_routes
        if allow_paid_fallback:
            routes += paid_routes

    if normalized in CAPABILITY_TASK_TYPES:
        capable_routes = [r for r in routes if normalized in r.capabilities]
        # Only restrict to capable_routes if at least one capable free/local route is fully available.
        # Otherwise, keep non-capable free/local routes in the list as fallback before using paid GPT.
        has_available_capable_free = any(
            r.cost_tier != "paid" and
            _library_available(r) and
            _env_available(r) and
            bridge_state.is_available(r.provider, r.model)
            for r in capable_routes
        )
        if has_available_capable_free:
            routes = capable_routes

    return _sort_routes_by_cost_and_health(routes, requested, prompt)


def _env_available(route: Route) -> bool:
    if route.required_env is None:
        return True
    return bool(os.environ.get(route.required_env))


def _library_available(route: Route) -> bool:
    if route.ask in (_ask_groq, _ask_cerebras, _ask_openrouter, _ask_hf, _ask_gpt):
        return OpenAI is not None
    if route.ask == _ask_gemini:
        return genai is not None
    return True


def _preflight_routes(routes: List[Route]) -> tuple[List[Route], list[str]]:
    """Select routes that are viable before dispatching the primary request.

    This is a read-only telemetry snapshot.  Token consumption and the final
    availability check remain in ``ask_smart`` so a concurrent request cannot
    bypass the existing rate limiter or circuit breaker.
    """
    viable = []
    errors = []
    for route in routes:
        target = f"{route.provider}/{route.model}"
        if not _library_available(route):
            errors.append(f"{target}: required library not installed")
            continue
        if not _env_available(route):
            errors.append(f"{target}: missing {route.required_env}")
            continue
        health = bridge_state.get_route_health(route.provider, route.model)
        if not health["is_available"]:
            errors.append(f"{target}: cooling down")
            continue
        if not health.get("token_bucket_available", True):
            wait_sec = health.get("token_bucket_wait_seconds", 0.0)
            errors.append(f"{target}: token bucket throttled (refill in {wait_sec:.1f}s)")
            continue
        viable.append(route)
    return viable, errors


def _is_retryable_error(exc: Exception) -> bool:
    retryable_names = {
        "APIConnectionError",
        "APITimeoutError",
        "ConnectError",
        "ReadTimeout",
        "RateLimitError",
    }
    rate_limit_err = getattr(openai, "RateLimitError", tuple()) if openai is not None else tuple()
    return isinstance(exc, rate_limit_err) or exc.__class__.__name__ in retryable_names


def _classify_provider_error(exc: Exception) -> tuple[str, str]:
    """Classify an upstream failure without coupling routing to SDK internals."""
    response = getattr(exc, "response", None)
    status = getattr(exc, "status_code", None) or getattr(response, "status_code", None)
    name = exc.__class__.__name__.lower()
    message = str(exc).lower()
    if status in (401, 403) or "authentication" in name or "permission" in name or "invalid api key" in message or "unauthorized" in message or "forbidden" in message:
        return "fatal", "authentication"
    if status == 404 or "model not found" in message or "unknown model" in message or "no such model" in message:
        return "fatal", "model_not_found"
    if status in (400, 422) or "invalid parameter" in message or "invalid request" in message or "bad request" in message:
        return "fatal", "invalid_parameters"
    if status == 429 or (isinstance(status, int) and status >= 500) or _is_retryable_error(exc):
        return "transient", "upstream_or_network"
    return "transient", "unknown"


def _should_fallback(exc: Exception) -> bool:
    if isinstance(exc, ValueError) and "budget cap" in str(exc).lower():
        return False
    return True


def run_provider_heartbeat() -> list[dict]:
    """Probe configured free/local routes and refresh their live metrics.

    Heartbeats deliberately use the existing route call functions and never
    include paid routes.  A failed probe updates the same circuit-breaker
    state used by normal requests, so routing immediately reflects both
    availability and latency without adding a second health model.
    """
    results = []
    for route in _routes_for("auto"):
        if route.cost_tier == "paid" or not _library_available(route) or not _env_available(route):
            continue
        if not bridge_state.is_available(route.provider, route.model):
            continue
        try:
            usage_tracker.check_budget(route.provider, route.model)
        except ValueError:
            # Budget/quota state is already reflected by bridge_state health;
            # do not convert it into a transport-failure cooldown.
            continue
        started = time.monotonic()
        try:
            route.ask(HEARTBEAT_PROMPT, route)
        except Exception as exc:
            latency = time.monotonic() - started
            bridge_state.record_metric(route.provider, route.model, latency, success=False)
            failure_class, failure_category = _classify_provider_error(exc)
            failure_model = None if failure_category == "authentication" else route.model
            bridge_state.mark_unavailable(route.provider, exc.__class__.__name__, model=failure_model,
                                          failure_class=failure_class, failure_category=failure_category)
            results.append({"provider": route.provider, "model": route.model, "available": False, "latency": latency})
        else:
            latency = time.monotonic() - started
            bridge_state.mark_available(route.provider, route.model)
            bridge_state.record_metric(route.provider, route.model, latency, success=True)
            results.append({"provider": route.provider, "model": route.model, "available": True, "latency": latency})
    return results


def start_provider_heartbeat() -> threading.Event:
    """Start the idempotent background heartbeat worker for the bridge server."""
    global _heartbeat_thread, _heartbeat_stop
    if _heartbeat_thread is not None and _heartbeat_thread.is_alive():
        return _heartbeat_stop
    try:
        interval = max(5.0, float(os.environ.get(
            "SMART_ROUTER_HEARTBEAT_INTERVAL_SECONDS",
            str(DEFAULT_HEARTBEAT_INTERVAL_SECONDS),
        )))
    except ValueError:
        interval = DEFAULT_HEARTBEAT_INTERVAL_SECONDS
    _heartbeat_stop = threading.Event()

    def worker():
        while not _heartbeat_stop.is_set():
            run_provider_heartbeat()
            _heartbeat_stop.wait(interval)

    _heartbeat_thread = threading.Thread(target=worker, name="smart-router-heartbeat", daemon=True)
    _heartbeat_thread.start()
    return _heartbeat_stop


def _speculative_prewarm_routes(routes: List[Route]) -> None:
    """Speculatively probe fallback candidate routes in background for high-priority tasks.

    Performs non-blocking background probes on secondary free/local routes to ensure
    connections and health metrics are fresh if the primary route fails.
    """
    def worker():
        for route in routes:
            if route.cost_tier == "paid" or not _library_available(route) or not _env_available(route):
                continue
            if not bridge_state.is_available(route.provider, route.model):
                continue
            try:
                usage_tracker.check_budget(route.provider, route.model)
            except ValueError:
                continue
            started = time.monotonic()
            try:
                route.ask(HEARTBEAT_PROMPT, route)
            except Exception as exc:
                latency = time.monotonic() - started
                bridge_state.record_metric(route.provider, route.model, latency, success=False)
                failure_class, failure_category = _classify_provider_error(exc)
                failure_model = None if failure_category == "authentication" else route.model
                bridge_state.mark_unavailable(
                    route.provider, exc.__class__.__name__, model=failure_model,
                    failure_class=failure_class, failure_category=failure_category
                )
            else:
                latency = time.monotonic() - started
                bridge_state.mark_available(route.provider, route.model)
                bridge_state.record_metric(route.provider, route.model, latency, success=True)

    thread = threading.Thread(target=worker, name="speculative-prewarm", daemon=True)
    thread.start()


@mcp.tool()
def prewarm_high_priority_bridges(task_type: str = "coding") -> dict:
    """
    Speculatively pre-warm high-priority routes for critical or high-priority task types.
    Returns status and latency of pre-warmed candidate routes.
    """
    routes, _ = _preflight_routes(_routes_for(task_type))
    results = []
    for route in routes:
        if route.cost_tier == "paid" or not _library_available(route) or not _env_available(route):
            continue
        if not bridge_state.is_available(route.provider, route.model):
            continue
        try:
            usage_tracker.check_budget(route.provider, route.model)
        except ValueError:
            continue
        started = time.monotonic()
        try:
            route.ask(HEARTBEAT_PROMPT, route)
            latency = time.monotonic() - started
            bridge_state.mark_available(route.provider, route.model)
            bridge_state.record_metric(route.provider, route.model, latency, success=True)
            results.append({"provider": route.provider, "model": route.model, "status": "warmed", "latency": latency})
        except Exception as exc:
            latency = time.monotonic() - started
            failure_class, failure_category = _classify_provider_error(exc)
            failure_model = None if failure_category == "authentication" else route.model
            bridge_state.mark_unavailable(
                route.provider, exc.__class__.__name__, model=failure_model,
                failure_class=failure_class, failure_category=failure_category
            )
            results.append({"provider": route.provider, "model": route.model, "status": "failed", "error": str(exc), "latency": latency})
    return {"task_type": task_type, "warmed_routes": results}


@mcp.tool()
def ask_smart(prompt: str, task_type: str = "auto") -> str:
    """
    Ask the cheapest suitable provider first, then fall back through free/local routes
    before using an explicitly enabled paid last resort. Use task_type='paid' to
    request GPT first, or SMART_ROUTER_ALLOW_PAID_FALLBACK=1 to allow fallback.
    """
    errors = []
    best_partial = None
    request = prompt
    routes, preflight_errors = _preflight_routes(_routes_for(task_type, prompt))
    errors.extend(preflight_errors)

    normalized_type = TASK_PROFILE_ALIASES.get((task_type or "auto").lower().strip(), (task_type or "auto").lower().strip())
    is_explicit_paid = normalized_type in ("paid", "openai", "gpt")
    is_high_priority = not is_explicit_paid and (normalized_type in HIGH_PRIORITY_TASK_TYPES or _task_complexity(prompt) == "complex")
    if is_high_priority and len(routes) > 1:
        _speculative_prewarm_routes(routes[1:2])

    for route in routes:
        if not _library_available(route):
            errors.append(f"{route.provider}/{route.model}: required library not installed")
            continue
        if not _env_available(route):
            errors.append(f"{route.provider}/{route.model}: missing {route.required_env}")
            continue
        if not bridge_state.is_available(route.provider, route.model):
            errors.append(f"{route.provider}/{route.model}: cooling down")
            continue

        consumed, wait_sec = bridge_state.consume_provider_token(route.provider)
        if not consumed:
            errors.append(f"{route.provider}/{route.model}: token bucket throttled (refill in {wait_sec:.1f}s)")
            continue

        called = False
        start_time = time.time()

        try:
            usage_tracker.check_budget(route.provider, route.model)
            called = True
            result = route.ask(request, route)
            latency = time.time() - start_time
            bridge_state.mark_available(route.provider)
            bridge_state.mark_available(route.provider, route.model)
            bridge_state.record_metric(route.provider, route.model, latency, success=True)
            return _merge_continuation(best_partial, result) if best_partial else result
        except TruncatedResponseError as e:
            # The route responded successfully but hit its token limit. It is
            # healthy, so don't penalize it; ask the next route for the missing tail.
            latency = time.time() - start_time
            bridge_state.mark_available(route.provider)
            bridge_state.mark_available(route.provider, route.model)
            bridge_state.record_metric(route.provider, route.model, latency, success=True)
            errors.append(f"{route.provider}/{route.model}: truncated by token limit")
            best_partial = _merge_continuation(best_partial, e.partial_content) if best_partial else e.partial_content
            request = _continuation_prompt(best_partial)
            continue
        except Exception as e:
            failure_class, failure_category = _classify_provider_error(e)
            is_429_or_5xx = False
            is_rate_limit = False
            api_status_err = getattr(openai, "APIStatusError", None) if openai is not None else None
            rate_limit_err = getattr(openai, "RateLimitError", None) if openai is not None else None
            if isinstance(api_status_err, type) and isinstance(e, api_status_err):
                status_code = getattr(e, "status_code", None)
                if status_code == 429:
                    is_rate_limit = True
                    is_429_or_5xx = True
                elif status_code and 500 <= status_code < 600:
                    is_429_or_5xx = True
            elif isinstance(rate_limit_err, type) and isinstance(e, rate_limit_err):
                is_rate_limit = True
                is_429_or_5xx = True
            else:
                err_msg = str(e).lower()
                if "429" in err_msg or "rate limit" in err_msg:
                    is_rate_limit = True
                    is_429_or_5xx = True
                elif "500" in err_msg or "502" in err_msg or "503" in err_msg or "504" in err_msg:
                    is_429_or_5xx = True
                elif "internal server error" in err_msg or "bad gateway" in err_msg or "service unavailable" in err_msg or "gateway timeout" in err_msg:
                    is_429_or_5xx = True

            if called:
                latency = time.time() - start_time
                bridge_state.record_metric(route.provider, route.model, latency, success=False, is_rate_limit=is_rate_limit)
            errors.append(f"{route.provider}/{route.model}: {e.__class__.__name__}: {e}")
            if _is_retryable_error(e) or _should_fallback(e):
                reason = e.__class__.__name__
                headers = getattr(getattr(e, "response", None), "headers", None)
                recorded_quota_reset = (
                    is_rate_limit
                    and bridge_state.record_rate_limit_headers(route.provider, route.model, headers)
                )
                if not recorded_quota_reset:
                    failure_model = None if failure_category == "authentication" else route.model
                    bridge_state.mark_unavailable(
                        route.provider, reason, model=failure_model, is_429_or_5xx=is_429_or_5xx,
                        failure_class=failure_class, failure_category=failure_category,
                    )
                continue
            raise

    if best_partial is not None:
        return best_partial
    raise bridge_state.ProviderUnavailableError("No smart-router routes succeeded. " + " | ".join(errors))


@mcp.resource("status://bridges")
def get_bridges_status() -> str:
    """
    Get the cached health and diagnostics status of all AI bridges.
    """
    import json
    cache = bridge_state.load_status_cache()
    if cache is not None:
        return json.dumps(cache, indent=2)
    return json.dumps({"error": "Status cache file not found. Run check_bridges.py to generate it."}, indent=2)


@mcp.resource("status://smart-router")
def get_smart_router_status() -> str:
    """
    Get live route health, latency metrics, and failure counts for the smart router.
    """
    import json
    routes = _routes_for("auto")
    route_details = []
    for r in routes:
        health = bridge_state.get_route_health(r.provider, r.model)
        score = _route_capability_score(r, "auto")
        route_details.append({
            "provider": r.provider,
            "model": r.model,
            "cost_tier": r.cost_tier,
            "capabilities": list(r.capabilities),
            "capability_score": score["total"],
            "score_components": {
                "cost": score["cost"],
                "latency": score["latency"],
                "capability": score["capability"],
            },
            "env_configured": _env_available(r),
            "library_installed": _library_available(r),
            "is_available": health["is_available"],
            "consecutive_failures": health["consecutive_failures"],
            "success_rate": health["success_rate"],
            "avg_latency": health["avg_latency"],
            "provider_is_degraded": health.get("provider_is_degraded", False),
            "consecutive_slow_calls": health.get("consecutive_slow_calls", 0),
            "is_soft_capped": health.get("is_soft_capped", False),
            "token_bucket_available": health.get("token_bucket_available", True),
            "token_bucket_tokens": health.get("token_bucket_tokens", 0.0),
            "token_bucket_wait_seconds": health.get("token_bucket_wait_seconds", 0.0),
        })

    return json.dumps({
        "timestamp": time.time(),
        "routes": route_details,
        "routing_profiles": ROUTING_PROFILE_WEIGHTS,
        "latency_heatmap": bridge_state.get_latency_heatmap(),
    }, indent=2)


if __name__ == "__main__":
    start_provider_heartbeat()
    mcp.run(transport="stdio")
