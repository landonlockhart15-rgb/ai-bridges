import os
import sys
from mcp.server.fastmcp import FastMCP
import openai
from openai import OpenAI

# Add repository root to path so we can import usage_tracker
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bridge_state
import usage_tracker

mcp = FastMCP("openrouter-bridge")
PROVIDER = "openrouter-bridge"

# Free models on OpenRouter (append :free to get zero-cost tier).
# When one hits rate limits, switch model — they each have independent quotas.
FREE_MODELS = [
    "nvidia/nemotron-3-super-120b-a12b:free",         # 120B, strongest free
    "google/gemma-4-31b-it:free",                     # Google Gemma 4 31B
    "google/gemma-4-26b-a4b-it:free",                 # Google Gemma 4 MoE
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",  # reasoning
    "poolside/laguna-m.1:free",                       # medium, capable
    "poolside/laguna-xs.2:free",                      # fast, small
]

def _client():
    return OpenAI(
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": os.environ.get("OPENROUTER_REFERER", "https://github.com/yourusername/ai-bridges"),
            "X-Title": os.environ.get("OPENROUTER_TITLE", "AI Bridges for Claude Code"),
        },
    )

def _ask_openrouter_with_fallback(messages: list, model: str) -> str:
    # Determine the fallback chain of models to try.
    if model in FREE_MODELS:
        start_idx = FREE_MODELS.index(model)
        models_to_try = FREE_MODELS[start_idx:]
    else:
        # If the requested model is not in FREE_MODELS, try it first,
        # then fall back to the FREE_MODELS in order.
        models_to_try = [model] + FREE_MODELS

    available_models = bridge_state.filter_available_models(PROVIDER, models_to_try)
    if not available_models:
        bridge_state.raise_if_unavailable(PROVIDER)
        raise bridge_state.ProviderUnavailableError(f"No OpenRouter models are currently available")

    client = _client()
    for i, current_model in enumerate(available_models):
        usage_tracker.check_budget(PROVIDER, current_model)
        try:
            response = client.chat.completions.create(
                model=current_model,
                messages=messages,
            )
            bridge_state.mark_available(PROVIDER)
            bridge_state.mark_available(PROVIDER, current_model)
            usage_tracker.record_usage(
                provider=PROVIDER,
                model=current_model,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens
            )
            return response.choices[0].message.content
        except openai.RateLimitError as e:
            bridge_state.mark_unavailable(PROVIDER, "rate_limit", model=current_model, is_429_or_5xx=True)
            if i < len(available_models) - 1:
                next_model = available_models[i + 1]
                print(
                    f"Rate limit hit for model '{current_model}' (HTTP 429). "
                    f"Falling back to '{next_model}'...",
                    file=sys.stderr
                )
            else:
                bridge_state.mark_unavailable(PROVIDER, "rate_limit", is_429_or_5xx=True)
                raise
        except Exception as e:
            is_5xx = False
            api_status_err = getattr(openai, "APIStatusError", None)
            if isinstance(api_status_err, type) and isinstance(e, api_status_err):
                status_code = getattr(e, "status_code", None)
                if status_code and 500 <= status_code < 600:
                    is_5xx = True
            else:
                err_msg = str(e).lower()
                if "500" in err_msg or "502" in err_msg or "503" in err_msg or "504" in err_msg:
                    is_5xx = True
                elif "internal server error" in err_msg or "bad gateway" in err_msg or "service unavailable" in err_msg or "gateway timeout" in err_msg:
                    is_5xx = True

            if is_5xx:
                bridge_state.mark_unavailable(PROVIDER, "server_error", model=current_model, is_429_or_5xx=True)
                bridge_state.mark_unavailable(PROVIDER, "server_error", is_429_or_5xx=True)
            elif e.__class__.__name__ in ("APIConnectionError", "APITimeoutError", "ConnectError", "ReadTimeout"):
                bridge_state.mark_unavailable(PROVIDER, "connection_error", is_429_or_5xx=False)
            raise

@mcp.tool()
def ask_openrouter(prompt: str, model: str = "nvidia/nemotron-3-super-120b-a12b:free") -> str:
    """
    Send a prompt to OpenRouter (many free models) and return the response.
    Free models: nvidia/nemotron-3-super-120b-a12b:free, google/gemma-4-31b-it:free,
    google/gemma-4-26b-a4b-it:free, nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free,
    poolside/laguna-m.1:free, poolside/laguna-xs.2:free
    If one model hits rate limits, switch to another — each has independent quotas.
    """
    return _ask_openrouter_with_fallback(
        messages=[{"role": "user", "content": prompt}],
        model=model,
    )

@mcp.tool()
def ask_openrouter_with_context(system: str, prompt: str, model: str = "nvidia/nemotron-3-super-120b-a12b:free") -> str:
    """Send a prompt to OpenRouter with a system prompt."""
    return _ask_openrouter_with_fallback(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        model=model,
    )

@mcp.tool()
def get_bridge_costs(timeframe: str = "all") -> str:
    """
    Get a summary of the tokens used and costs incurred across all AI bridges.
    timeframe: 'all', 'today', or 'month'
    """
    return usage_tracker.get_bridge_costs(timeframe)

if __name__ == "__main__":
    mcp.run(transport="stdio")
