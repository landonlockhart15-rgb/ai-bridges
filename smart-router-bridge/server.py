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
    max_context_tokens: int = 32768


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
    "bug-fix": "coding",
    "bugfix": "coding",
    "debug": "coding",
    "unit_test": "coding",
    "unit-test": "coding",
    "test": "coding",
    "code_review": "coding",
    "code-review": "coding",
    "docs": "simple_extraction",
    "documentation": "simple_extraction",
    "story": "creative_writing",
    "copywriting": "creative_writing",
    "fast": "fast",
    "quick": "fast",
    "reliable": "reliable",
    "high_reliability": "reliable",
    "high-reliability": "reliable",
}

HIGH_PRIORITY_TASK_TYPES = {
    "coding",
    "refactor",
    "bug_fix",
    "bug-fix",
    "bugfix",
    "debug",
    "unit_test",
    "unit-test",
    "test",
    "code_review",
    "code-review",
    "reliable",
    "high_reliability",
    "high-reliability",
}

# Keep capability filtering canonical, while allowing caller-facing profiles
# to express a useful cost preference within that capability.
TASK_PROFILE_COST_PRIORITIES = {
    "unit_test": ("local", "free-cloud", "paid"),
    "unit-test": ("local", "free-cloud", "paid"),
    "test": ("local", "free-cloud", "paid"),
    "docs": ("local", "free-cloud", "paid"),
    "documentation": ("local", "free-cloud", "paid"),
    "story": ("local", "free-cloud", "paid"),
    "copywriting": ("local", "free-cloud", "paid"),
    "refactor": ("free-cloud", "local", "paid"),
    "bug_fix": ("free-cloud", "local", "paid"),
    "bug-fix": ("free-cloud", "local", "paid"),
    "bugfix": ("free-cloud", "local", "paid"),
    "debug": ("free-cloud", "local", "paid"),
    "code_review": ("free-cloud", "local", "paid"),
    "code-review": ("free-cloud", "local", "paid"),
    "fast": ("free-cloud", "local", "paid"),
    "quick": ("free-cloud", "local", "paid"),
    "reliable": ("free-cloud", "local", "paid"),
    "high_reliability": ("free-cloud", "local", "paid"),
    "high-reliability": ("free-cloud", "local", "paid"),
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


def _estimate_tokens(val: str | list | dict | None) -> int:
    """Return a fast, provider-neutral estimate for context budgeting."""
    if not val:
        return 0
    if isinstance(val, str):
        return max(1, (len(val) + 3) // 4)
    if isinstance(val, dict):
        role = str(val.get("role", "") or "")
        c_val = val.get("content")
        content = str(c_val) if c_val is not None else ""
        return _estimate_tokens(role) + _estimate_tokens(content) + 4
    if isinstance(val, list):
        return sum(_estimate_tokens(item) for item in val)
    return max(1, (len(str(val)) + 3) // 4)


def _negotiate_prompt_string(prompt: str, capacity: int) -> str:
    if capacity <= 0:
        return ""
    current_tokens = _estimate_tokens(prompt)
    if current_tokens <= capacity:
        return prompt

    max_chars = capacity * 4
    if max_chars <= 0:
        return ""
    prune_notice = "\n[... Preceding context dynamically pruned to fit bridge context limit ...]\n"
    notice_chars = len(prune_notice)
    available_chars = max_chars - notice_chars
    if available_chars < 40:
        return prompt[-max_chars:] if max_chars > 0 else ""

    head_chars = max(20, available_chars // 4)
    tail_chars = available_chars - head_chars

    head = prompt[:head_chars]
    tail = prompt[-tail_chars:]
    return f"{head}{prune_notice}{tail}"


def _negotiate_messages(messages: list[dict], capacity: int) -> list[dict]:
    if capacity <= 0:
        return []
    current_tokens = _estimate_tokens(messages)
    if current_tokens <= capacity:
        return list(messages)

    if not messages:
        return []

    system_msgs = [m for m in messages if isinstance(m, dict) and m.get("role") == "system"]
    other_msgs = [m for m in messages if isinstance(m, dict) and m.get("role") != "system"]

    sys_tokens = _estimate_tokens(system_msgs)

    if sys_tokens >= capacity:
        pruned_sys = []
        avail_sys = capacity
        for sm in system_msgs:
            sm_c = sm.get("content")
            sm_str = str(sm_c) if sm_c is not None else ""
            c = _negotiate_prompt_string(sm_str, max(10, avail_sys - 8))
            pruned_sys.append({"role": "system", "content": c})
            avail_sys -= _estimate_tokens(pruned_sys[-1])
            if avail_sys <= 0:
                break
        return pruned_sys

    rem_capacity = capacity - sys_tokens
    if not other_msgs:
        return system_msgs

    last_msg = other_msgs[-1]
    last_tokens = _estimate_tokens(last_msg)
    middle_msgs = other_msgs[:-1]

    dummy_notice = {
        "role": "system",
        "content": f"[Context Negotiation: {len(middle_msgs)} earlier turns pruned to fit target bridge context limit]"
    }
    notice_overhead = _estimate_tokens(dummy_notice)

    selected_middle = []
    current_middle_tokens = 0
    if _estimate_tokens(middle_msgs) + last_tokens <= rem_capacity:
        selected_middle = list(middle_msgs)
        current_middle_tokens = _estimate_tokens(selected_middle)
    else:
        avail_for_middle = rem_capacity - notice_overhead
        for msg in reversed(middle_msgs):
            msg_tokens = _estimate_tokens(msg)
            if current_middle_tokens + msg_tokens + last_tokens <= avail_for_middle:
                selected_middle.insert(0, msg)
                current_middle_tokens += msg_tokens
            else:
                break

    pruned_count = len(middle_msgs) - len(selected_middle)
    result = list(system_msgs)
    notice_tokens = 0
    if pruned_count > 0:
        notice_msg = {
            "role": "system",
            "content": f"[Context Negotiation: {pruned_count} earlier turns pruned to fit target bridge context limit]"
        }
        result.append(notice_msg)
        notice_tokens = _estimate_tokens(notice_msg)

    result.extend(selected_middle)

    used_so_far = sys_tokens + notice_tokens + current_middle_tokens
    if used_so_far + last_tokens > capacity:
        avail_for_last = max(10, capacity - used_so_far - 8)
        last_c = last_msg.get("content")
        last_str = str(last_c) if last_c is not None else ""
        last_content = _negotiate_prompt_string(last_str, avail_for_last)
        new_last = dict(last_msg)
        new_last["content"] = last_content
        result.append(new_last)
    else:
        result.append(last_msg)

    return result


def negotiate_context_window(
    prompt_or_messages: str | list | dict,
    target_max_tokens: int,
    reserve_tokens: int = 1024,
) -> str | list | dict:
    """Dynamically prune or summarize conversation context to fit target bridge context window.

    Ensures prompt or structured message history stays within target_max_tokens - reserve_tokens.
    Preserves system instructions and the latest turns while summarizing/pruning older context.
    """
    reserve_limit = max(16, int(target_max_tokens * 0.25))
    effective_reserve = min(reserve_tokens, reserve_limit)
    capacity = max(16, target_max_tokens - effective_reserve)
    if _estimate_tokens(prompt_or_messages) <= capacity:
        return prompt_or_messages

    if isinstance(prompt_or_messages, list):
        return _negotiate_messages(prompt_or_messages, capacity)
    if isinstance(prompt_or_messages, str):
        return _negotiate_prompt_string(prompt_or_messages, capacity)
    if isinstance(prompt_or_messages, dict):
        c = _negotiate_prompt_string(prompt_or_messages.get("content", ""), capacity - 4)
        res = dict(prompt_or_messages)
        res["content"] = c
        return res
    return prompt_or_messages


def get_route_max_context_tokens(route: Route) -> int:
    env_name = "SMART_ROUTER_MAX_CONTEXT_" + route.provider.upper().replace("-", "_")
    configured = os.environ.get(env_name)
    if configured:
        try:
            return max(1, int(configured))
        except ValueError:
            pass
    return max(1, route.max_context_tokens)


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


def _task_complexity(prompt: str | list | dict) -> str:
    if isinstance(prompt, list):
        text = " ".join(str(m.get("content", "") or "") for m in prompt if isinstance(m, dict))
    elif isinstance(prompt, dict):
        text = str(prompt.get("content", "") or "")
    else:
        text = str(prompt or "")
    normalized = text.lower().strip()
    word_count = len(normalized.split())
    if not normalized:
        return "medium"
    if word_count >= 180 or any(keyword in normalized for keyword in COMPLEX_INTENT_KEYWORDS):
        return "complex"
    if word_count <= 40 and any(keyword in normalized for keyword in SIMPLE_INTENT_KEYWORDS):
        return "simple"
    return "medium"


def _get_cost_priority(cost_tier: str, task_type: str, prompt: str | None = None) -> int:
    raw = (task_type or "auto").lower().strip()
    alias = TASK_PROFILE_ALIASES.get(raw, raw)
    profile_priority = TASK_PROFILE_COST_PRIORITIES.get(raw) or TASK_PROFILE_COST_PRIORITIES.get(alias)
    if profile_priority:
        return profile_priority.index(cost_tier) if cost_tier in profile_priority else 99
    complexity = _task_complexity(prompt or "") if raw in ("auto", "simple", "fast", "quick") or alias in ("auto", "simple", "fast", "quick") else "medium"
    if raw in ("paid", "openai", "gpt") or alias in ("paid", "openai", "gpt"):
        mapping = {
            "paid": 0,
            "free-cloud": 1,
            "local": 2
        }
    elif raw in ("local", "offline", "private") or alias in ("local", "offline", "private"):
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
    raw = (task_type or "auto").lower().strip()
    alias = TASK_PROFILE_ALIASES.get(raw, raw)
    scores = route.capability_scores or {}
    if raw in scores:
        return max(0.0, min(1.0, float(scores[raw])))
    if alias in scores:
        return max(0.0, min(1.0, float(scores[alias])))
    if raw in route.capabilities or alias in route.capabilities:
        return 0.75
    if raw in ("auto", "simple", "fast", "quick", "paid", "openai", "gpt", "local", "offline", "private") or alias in ("auto", "simple", "fast", "quick", "paid", "openai", "gpt", "local", "offline", "private"):
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
    raw = (task_type or "auto").lower().strip()
    alias = TASK_PROFILE_ALIASES.get(raw, raw)
    is_critical = raw in ("paid", "openai", "gpt") or alias in ("paid", "openai", "gpt") or _task_complexity(prompt or "") == "complex"
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
        raw = (task_type or "auto").lower().strip()
        alias = TASK_PROFILE_ALIASES.get(raw, raw)
        is_critical = raw in ("paid", "openai", "gpt") or alias in ("paid", "openai", "gpt") or _task_complexity(prompt or "") == "complex"
        soft_cap_prio = 1 if (health.get("is_soft_capped", False) and not is_critical) else 0

        # 2. Cost tier priority (lower value first)
        cost_prio = _get_cost_priority(route.cost_tier, task_type, prompt)
        # Token bucket status (un-throttled first)
        is_throttled = 0 if health.get("token_bucket_available", True) else 1
        # 3. Provider degradation status (healthy first, i.e. 0 for healthy, 1 for degraded)
        is_degraded = 1 if health.get("provider_is_degraded", False) else 0
        # QoS is a canonical provider profile; use it as a tie-breaker inside
        # the existing cost tier so the free/local/paid policy remains intact.
        qos_rank = bridge_state.qos_priority(
            health.get("qos_profile") or bridge_state.qos_profile(health)
        )
        # 4. Health: consecutive failures count (lower failures first)
        failures = health["consecutive_failures"]
        # 5. Health: success rate (higher rate first -> negative rate)
        neg_success_rate = -health["success_rate"]
        # 6. Weighted capability score (higher score first)
        neg_score = -score
        # 7. Latency (lower latency first)
        latency = health["avg_latency"]

        return (is_avail_val, soft_cap_prio, cost_prio, qos_rank, is_throttled, is_degraded, failures, neg_success_rate, neg_score, latency)

    return sorted(routes, key=sort_key)



def _routes_for(task_type: str, prompt: str | None = None) -> List[Route]:
    requested = (task_type or "auto").lower().strip()
    normalized = TASK_PROFILE_ALIASES.get(requested, requested)
    simple_model = "llama-3.1-8b-instant" if normalized in ("simple", "fast", "quick") or requested in ("simple", "fast", "quick") else "llama-3.3-70b-versatile"
    local_model = os.environ.get("SMART_ROUTER_LOCAL_MODEL", "gemma4:latest")
    paid_model = os.environ.get("SMART_ROUTER_PAID_MODEL", "gpt-4o-mini")

    free_routes = [
        Route("groq-bridge", simple_model, "free-cloud", "GROQ_API_KEY", _ask_groq,
              ("coding", "general", "simple_extraction"),
              {"general": 0.72, "coding": 0.70, "simple_extraction": 0.84}, 32768),
        Route("gemini-bridge", "gemini-2.5-flash", "free-cloud", "GEMINI_API_KEY", _ask_gemini,
              ("coding", "creative_writing", "general"),
              {"general": 0.72, "coding": 0.76, "creative_writing": 0.86}, 1048576),
        Route("cerebras-bridge", "gpt-oss-120b", "free-cloud", "CEREBRAS_API_KEY", _ask_cerebras,
              ("coding", "general"),
              {"general": 0.72, "coding": 0.82}, 131072),
        Route("openrouter-bridge", "nvidia/nemotron-3-super-120b-a12b:free", "free-cloud", "OPENROUTER_API_KEY", _ask_openrouter,
              ("general", "simple_extraction"),
              {"general": 0.70, "simple_extraction": 0.78}, 131072),
        Route("hf-bridge", local_model, "local", None, _ask_hf,
              ("general", "simple_extraction"),
              {"general": 0.58, "simple_extraction": 0.68}, 32768),
    ]
    paid_routes = [Route("gpt-bridge", paid_model, "paid", "OPENAI_API_KEY", _ask_gpt,
                          ("coding", "creative_writing", "general", "simple_extraction"),
                          {"general": 0.86, "coding": 0.90, "creative_writing": 0.88, "simple_extraction": 0.86}, 128000)]

    paid_requested = normalized in ("paid", "openai", "gpt") or requested in ("paid", "openai", "gpt")
    allow_paid_fallback = os.environ.get("SMART_ROUTER_ALLOW_PAID_FALLBACK", "").lower() in ("1", "true", "yes", "on")

    if paid_requested:
        routes = paid_routes + free_routes
    elif normalized in ("local", "offline", "private") or requested in ("local", "offline", "private"):
        routes = [free_routes[-1]] + free_routes[:-1]
        if allow_paid_fallback:
            routes += paid_routes
    else:
        routes = free_routes
        if allow_paid_fallback:
            routes += paid_routes

    if normalized in CAPABILITY_TASK_TYPES or requested in CAPABILITY_TASK_TYPES:
        capable_routes = [r for r in routes if normalized in r.capabilities or requested in r.capabilities]
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
                latency = time.monotonic() - started
                bridge_state.record_metric(route.provider, route.model, latency, success=True)
            except Exception:
                latency = time.monotonic() - started
                # Probe-only: record metric but never mark_unavailable — a transient
                # probe failure must not poison the fallback right before ask_smart needs it.
                bridge_state.record_metric(route.provider, route.model, latency, success=False)

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
def negotiate_context(prompt: str, target_max_tokens: int = 32768) -> dict:
    """
    Negotiate and prune prompt/context history to fit a target bridge context window.
    Returns negotiated prompt and token statistics.
    """
    initial_tokens = _estimate_tokens(prompt)
    negotiated = negotiate_context_window(prompt, target_max_tokens)
    final_tokens = _estimate_tokens(negotiated)
    return {
        "initial_tokens": initial_tokens,
        "final_tokens": final_tokens,
        "target_max_tokens": target_max_tokens,
        "was_pruned": final_tokens < initial_tokens,
        "negotiated_prompt": str(negotiated),
    }


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
    original_prompt = prompt
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

        target_max_context = get_route_max_context_tokens(route)
        route_request = request
        if best_partial:
            continuation = _continuation_prompt(best_partial)
            est_total = _estimate_tokens(original_prompt) + _estimate_tokens(continuation)
            if est_total > target_max_context:
                avail_for_orig = max(64, target_max_context - _estimate_tokens(continuation) - 256)
                negotiated_orig = negotiate_context_window(original_prompt, avail_for_orig, reserve_tokens=0)
                if _estimate_tokens(negotiated_orig) + _estimate_tokens(continuation) > target_max_context:
                    errors.append(f"{route.provider}/{route.model}: continuation exceeds context limit")
                    continue
            route_request = continuation
        else:
            if _estimate_tokens(route_request) > target_max_context:
                route_request = negotiate_context_window(route_request, target_max_context, reserve_tokens=256)
                if _estimate_tokens(route_request) > target_max_context:
                    errors.append(f"{route.provider}/{route.model}: prompt exceeds context limit after negotiation")
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
            result = route.ask(route_request, route)
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
            "qos_profile": health.get("qos_profile", bridge_state.qos_profile(health)),
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
