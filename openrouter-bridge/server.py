import os
import sys
from mcp.server.fastmcp import FastMCP
import openai
from openai import OpenAI

mcp = FastMCP("openrouter-bridge")

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

    client = _client()
    for i, current_model in enumerate(models_to_try):
        try:
            response = client.chat.completions.create(
                model=current_model,
                messages=messages,
            )
            return response.choices[0].message.content
        except openai.RateLimitError as e:
            if i < len(models_to_try) - 1:
                next_model = models_to_try[i + 1]
                print(
                    f"Rate limit hit for model '{current_model}' (HTTP 429). "
                    f"Falling back to '{next_model}'...",
                    file=sys.stderr
                )
            else:
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

if __name__ == "__main__":
    mcp.run(transport="stdio")
