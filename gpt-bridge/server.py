import os
import sys
from mcp.server.fastmcp import FastMCP
from openai import OpenAI

mcp = FastMCP("gpt-bridge")
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

@mcp.tool()
def ask_gpt(prompt: str, model: str = "gpt-5.5") -> str:
    """
    Send a prompt to GPT and return its response.
    Use gpt-4o-mini for simple/cheap tasks, gpt-5.5 for complex reasoning.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

@mcp.tool()
def ask_gpt_with_context(system: str, prompt: str, model: str = "gpt-5.5") -> str:
    """
    Send a prompt to GPT with a custom system prompt.
    Useful for giving GPT a specific role or persona before asking.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    mcp.run(transport="stdio")
