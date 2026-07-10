import os
import sys
import time
from dataclasses import dataclass
from typing import Callable, List

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


@dataclass(frozen=True)
class Route:
    provider: str
    model: str
    cost_tier: str
    required_env: str | None
    ask: Callable[..., str]
    capabilities: tuple[str, ...] = ()


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
}


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


def _sort_routes_by_cost_and_health(routes: List[Route], task_type: str, prompt: str | None = None) -> List[Route]:
    def sort_key(route):
        health = bridge_state.get_route_health(route.provider, route.model)
        # 1. Availability status (available first, i.e. 0 for available, 1 for cooling down)
        is_avail_val = 0 if health["is_available"] else 1
        # 2. Cost tier priority (lower value first)
        cost_prio = _get_cost_priority(route.cost_tier, task_type, prompt)
        # 3. Provider degradation status (healthy first, i.e. 0 for healthy, 1 for degraded)
        is_degraded = 1 if health.get("provider_is_degraded", False) else 0
        # 4. Health: consecutive failures count (lower failures first)
        failures = health["consecutive_failures"]
        # 5. Health: success rate (higher rate first -> negative rate)
        neg_success_rate = -health["success_rate"]
        # 6. Latency (lower latency first)
        latency = health["avg_latency"]

        return (is_avail_val, cost_prio, is_degraded, failures, neg_success_rate, latency)

    return sorted(routes, key=sort_key)


def _routes_for(task_type: str, prompt: str | None = None) -> List[Route]:
    normalized = (task_type or "auto").lower().strip()
    normalized = TASK_PROFILE_ALIASES.get(normalized, normalized)
    simple_model = "llama-3.1-8b-instant" if normalized in ("simple", "fast", "quick") else "llama-3.3-70b-versatile"
    local_model = os.environ.get("SMART_ROUTER_LOCAL_MODEL", "gemma4:latest")
    paid_model = os.environ.get("SMART_ROUTER_PAID_MODEL", "gpt-4o-mini")

    free_routes = [
        Route("groq-bridge", simple_model, "free-cloud", "GROQ_API_KEY", _ask_groq,
              ("coding", "general", "simple_extraction")),
        Route("gemini-bridge", "gemini-2.5-flash", "free-cloud", "GEMINI_API_KEY", _ask_gemini,
              ("coding", "creative_writing", "general")),
        Route("cerebras-bridge", "gpt-oss-120b", "free-cloud", "CEREBRAS_API_KEY", _ask_cerebras,
              ("coding", "general")),
        Route("openrouter-bridge", "nvidia/nemotron-3-super-120b-a12b:free", "free-cloud", "OPENROUTER_API_KEY", _ask_openrouter,
              ("general", "simple_extraction")),
        Route("hf-bridge", local_model, "local", None, _ask_hf,
              ("general", "simple_extraction")),
    ]
    paid_routes = [Route("gpt-bridge", paid_model, "paid", "OPENAI_API_KEY", _ask_gpt,
                          ("coding", "creative_writing", "general", "simple_extraction"))]

    if normalized in ("paid", "openai", "gpt"):
        routes = paid_routes + free_routes
    elif normalized in ("local", "offline", "private"):
        routes = [free_routes[-1]] + free_routes[:-1] + paid_routes
    else:
        routes = free_routes + paid_routes

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

    return _sort_routes_by_cost_and_health(routes, task_type, prompt)


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


def _should_fallback(exc: Exception) -> bool:
    if isinstance(exc, ValueError) and "budget cap" in str(exc).lower():
        return False
    return True


@mcp.tool()
def ask_smart(prompt: str, task_type: str = "auto") -> str:
    """
    Ask the cheapest suitable provider first, then fall back through free/local routes
    before using GPT as the paid last resort. Use task_type='paid' to request GPT first.
    """
    errors = []
    best_partial = None
    for route in _routes_for(task_type, prompt):
        if not _library_available(route):
            errors.append(f"{route.provider}/{route.model}: required library not installed")
            continue
        if not _env_available(route):
            errors.append(f"{route.provider}/{route.model}: missing {route.required_env}")
            continue
        if not bridge_state.is_available(route.provider, route.model):
            errors.append(f"{route.provider}/{route.model}: cooling down")
            continue
        
        called = False
        start_time = time.time()
        try:
            usage_tracker.check_budget(route.provider, route.model)
            called = True
            result = route.ask(prompt, route)
            latency = time.time() - start_time
            bridge_state.record_metric(route.provider, route.model, latency, success=True)
            bridge_state.mark_available(route.provider)
            bridge_state.mark_available(route.provider, route.model)
            return result
        except TruncatedResponseError as e:
            # The route responded successfully but hit its token limit. It is
            # healthy, so don't penalize it — just prefer a route that can
            # return a complete answer (which may escalate to the paid tier).
            latency = time.time() - start_time
            bridge_state.record_metric(route.provider, route.model, latency, success=True)
            bridge_state.mark_available(route.provider)
            bridge_state.mark_available(route.provider, route.model)
            errors.append(f"{route.provider}/{route.model}: truncated by token limit")
            best_partial = e.partial_content
            continue
        except Exception as e:
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
                bridge_state.mark_unavailable(route.provider, reason, model=route.model, is_429_or_5xx=is_429_or_5xx)
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
    status_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".bridge_status.json")
    if os.path.exists(status_file):
        try:
            with open(status_file, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            import json
            return json.dumps({"error": f"Failed to read cache: {str(e)}"}, indent=2)
    import json
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
        route_details.append({
            "provider": r.provider,
            "model": r.model,
            "cost_tier": r.cost_tier,
            "env_configured": _env_available(r),
            "library_installed": _library_available(r),
            "is_available": health["is_available"],
            "consecutive_failures": health["consecutive_failures"],
            "success_rate": health["success_rate"],
            "avg_latency": health["avg_latency"],
            "provider_is_degraded": health.get("provider_is_degraded", False),
        })
    return json.dumps({
        "timestamp": time.time(),
        "routes": route_details
    }, indent=2)


if __name__ == "__main__":
    mcp.run(transport="stdio")
