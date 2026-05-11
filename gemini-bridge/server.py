import os
from mcp.server.fastmcp import FastMCP
from google import genai
from google.genai import types

mcp = FastMCP("gem-bridge")
_client = None

def get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    return _client

@mcp.tool()
def ask_gemini(prompt: str, model: str = "gemini-2.5-flash") -> str:
    """
    Send a prompt to Gemini and return its response.
    Use gemini-2.5-flash for most tasks; gemini-2.5-pro requires a paid API key.
    """
    response = get_client().models.generate_content(model=model, contents=prompt)
    return response.text

@mcp.tool()
def ask_gemini_with_context(system: str, prompt: str, model: str = "gemini-2.5-flash") -> str:
    """
    Send a prompt to Gemini with a custom system prompt.
    Useful for giving Gemini a specific role or persona before asking.
    """
    response = get_client().models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=system)
    )
    return response.text

if __name__ == "__main__":
    mcp.run(transport="stdio")
