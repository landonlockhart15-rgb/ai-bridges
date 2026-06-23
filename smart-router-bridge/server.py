import os
import sys
import time
from dataclasses import dataclass
from typing import Callable, List

from mcp.server.fastmcp import FastMCP
import openai
from openai import OpenAI

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


def _openai_client(api_key_env: str, base_url: str | None = None, default_headers: dict | None = None):
    api_key = os.environ.get(api_key_env)
    if api_key_env == "OLLAMA_API_KEY":
        api_key = api_key or "ollama"
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        default_headers=default_headers,
    )


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
    return response.choices[0].message.content


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


def _sort_by_latency_within_tiers(routes: List[Route]) -> List[Route]:
    from collections import defaultdict
    tier_order = []
    routes_by_tier = defaultdict(list)
    for r in routes:
        if r.cost_tier not in routes_by_tier:
            tier_order.append(r.cost_tier)
        routes_by_tier[r.cost_tier].append(r)
        
    sorted_routes = []
    for tier in tier_order:
        def sort_key(route):
            metrics = bridge_state.get_metrics(route.provider, route.model)
            return (-metrics["success_rate"], metrics["avg_latency"])
        sorted_routes.extend(sorted(routes_by_tier[tier], key=sort_key))
    return sorted_routes


def _routes_for(task_type: str) -> List[Route]:
    normalized = (task_type or "auto").lower().strip()
    simple_model = "llama-3.1-8b-instant" if normalized in ("simple", "fast", "quick") else "llama-3.3-70b-versatile"
    local_model = os.environ.get("SMART_ROUTER_LOCAL_MODEL", "gemma4:latest")
    paid_model = os.environ.get("SMART_ROUTER_PAID_MODEL", "gpt-4o-mini")

    free_routes = [
        Route("groq-bridge", simple_model, "free-cloud", "GROQ_API_KEY", _ask_groq),
        Route("gemini-bridge", "gemini-2.5-flash", "free-cloud", "GEMINI_API_KEY", _ask_gemini),
        Route("cerebras-bridge", "gpt-oss-120b", "free-cloud", "CEREBRAS_API_KEY", _ask_cerebras),
        Route("openrouter-bridge", "nvidia/nemotron-3-super-120b-a12b:free", "free-cloud", "OPENROUTER_API_KEY", _ask_openrouter),
        Route("hf-bridge", local_model, "local", None, _ask_hf),
    ]
    paid_routes = [Route("gpt-bridge", paid_model, "paid", "OPENAI_API_KEY", _ask_gpt)]

    if normalized in ("paid", "openai", "gpt"):
        routes = paid_routes + free_routes
    elif normalized in ("local", "offline", "private"):
        routes = [free_routes[-1]] + free_routes[:-1] + paid_routes
    else:
        routes = free_routes + paid_routes
        
    return _sort_by_latency_within_tiers(routes)


def _env_available(route: Route) -> bool:
    if route.required_env is None:
        return True
    return bool(os.environ.get(route.required_env))


def _is_retryable_error(exc: Exception) -> bool:
    retryable_names = {
        "APIConnectionError",
        "APITimeoutError",
        "ConnectError",
        "ReadTimeout",
        "RateLimitError",
    }
    return isinstance(exc, getattr(openai, "RateLimitError", tuple())) or exc.__class__.__name__ in retryable_names


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
    for route in _routes_for(task_type):
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
        except Exception as e:
            if called:
                latency = time.time() - start_time
                bridge_state.record_metric(route.provider, route.model, latency, success=False)
            errors.append(f"{route.provider}/{route.model}: {e.__class__.__name__}: {e}")
            if _is_retryable_error(e) or _should_fallback(e):
                reason = e.__class__.__name__
                is_429_or_5xx = False
                api_status_err = getattr(openai, "APIStatusError", None)
                rate_limit_err = getattr(openai, "RateLimitError", None)
                if isinstance(api_status_err, type) and isinstance(e, api_status_err):
                    status_code = getattr(e, "status_code", None)
                    if status_code == 429 or (status_code and 500 <= status_code < 600):
                        is_429_or_5xx = True
                elif isinstance(rate_limit_err, type) and isinstance(e, rate_limit_err):
                    is_429_or_5xx = True
                else:
                    err_msg = str(e).lower()
                    if "429" in err_msg or "rate limit" in err_msg or "500" in err_msg or "502" in err_msg or "503" in err_msg or "504" in err_msg:
                        is_429_or_5xx = True
                    elif "internal server error" in err_msg or "bad gateway" in err_msg or "service unavailable" in err_msg or "gateway timeout" in err_msg:
                        is_429_or_5xx = True
                
                bridge_state.mark_unavailable(route.provider, reason, model=route.model, is_429_or_5xx=is_429_or_5xx)
                continue
            raise

    raise bridge_state.ProviderUnavailableError("No smart-router routes succeeded. " + " | ".join(errors))


if __name__ == "__main__":
    mcp.run(transport="stdio")
