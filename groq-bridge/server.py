import os
from mcp.server.fastmcp import FastMCP
from openai import OpenAI

mcp = FastMCP("groq-bridge")

def _client():
    return OpenAI(
        api_key=os.environ.get("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
    )

@mcp.tool()
def ask_groq(prompt: str, model: str = "llama-3.3-70b-versatile") -> str:
    """
    Send a prompt to Groq (free tier) and return the response.
    Fast models: llama-3.3-70b-versatile, llama-3.1-8b-instant, qwen-qwq-32b, gemma2-9b-it
    """
    response = _client().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

@mcp.tool()
def ask_groq_with_context(system: str, prompt: str, model: str = "llama-3.3-70b-versatile") -> str:
    """Send a prompt to Groq with a system prompt."""
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
