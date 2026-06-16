import os
from mcp.server.fastmcp import FastMCP
from openai import OpenAI

mcp = FastMCP("cerebras-bridge")

# Cerebras exposes an OpenAI-compatible endpoint. Valid free-tier model IDs are
# "gpt-oss-120b" (default) and "zai-glm-4.7"; the free tier allows 1M tokens/day
# at 60K tokens/min with very low latency inference.
def _client():
    return OpenAI(
        api_key=os.environ.get("CEREBRAS_API_KEY"),
        base_url="https://api.cerebras.ai/v1",
    )

@mcp.tool()
def ask_cerebras(prompt: str, model: str = "gpt-oss-120b") -> str:
    """
    Send a prompt to Cerebras (free tier) and return the response.
    Fast models: gpt-oss-120b, zai-glm-4.7
    Free tier: 1M tokens/day, 60K tokens/min — very fast inference.
    """
    response = _client().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

@mcp.tool()
def ask_cerebras_with_context(system: str, prompt: str, model: str = "gpt-oss-120b") -> str:
    """Send a prompt to Cerebras with a system prompt."""
    response = _client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    mcp.run(transport="stdio")
