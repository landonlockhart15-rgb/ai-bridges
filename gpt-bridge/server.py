import os
import sys
from mcp.server.fastmcp import FastMCP
from openai import OpenAI

# Add repository root to path so we can import usage_tracker
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import usage_tracker

mcp = FastMCP("gpt-bridge")
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

@mcp.tool()
def ask_gpt(prompt: str, model: str = "gpt-5.5") -> str:
    """
    Send a prompt to GPT and return its response.
    Use gpt-4o-mini for simple/cheap tasks, gpt-5.5 for complex reasoning.
    """
    usage_tracker.check_budget("gpt-bridge", model)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    usage_tracker.record_usage(
        provider="gpt-bridge",
        model=model,
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens
    )
    return response.choices[0].message.content

@mcp.tool()
def ask_gpt_with_context(system: str, prompt: str, model: str = "gpt-5.5") -> str:
    """
    Send a prompt to GPT with a custom system prompt.
    Useful for giving GPT a specific role or persona before asking.
    """
    usage_tracker.check_budget("gpt-bridge", model)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ]
    )
    usage_tracker.record_usage(
        provider="gpt-bridge",
        model=model,
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens
    )
    return response.choices[0].message.content

@mcp.tool()
def get_bridge_costs(timeframe: str = "all") -> str:
    """
    Get a summary of the tokens used and costs incurred across all AI bridges.
    timeframe: 'all', 'today', or 'month'
    """
    return usage_tracker.get_bridge_costs(timeframe)

if __name__ == "__main__":
    mcp.run(transport="stdio")
