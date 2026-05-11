import os
from mcp.server.fastmcp import FastMCP
from openai import OpenAI

mcp = FastMCP("hf-bridge")

HF_TOKEN = os.environ.get("HF_TOKEN", "")
OLLAMA_BASE_URL = "http://localhost:11434/v1"
HF_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_MODEL = "gemma4:latest"  # local Ollama — free, on-device

def get_client(model: str = DEFAULT_MODEL):
    # Route local Ollama models vs HuggingFace cloud
    if "/" not in model:
        return OpenAI(api_key="ollama", base_url=OLLAMA_BASE_URL)
    return OpenAI(api_key=HF_TOKEN, base_url=HF_BASE_URL)

@mcp.tool()
def ask_hf(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """
    Send a prompt to a local Ollama model or HuggingFace cloud model.
    Default: gemma4 (local Ollama, on-device, free)
    Local models (no slash):
      gemma4:latest                 — Gemma 4 4B, fast, on-device, FREE
      llama31-8b-abliterated:q4km   — Llama 3.1 8B, uncensored
      qwen2.5:3b                    — Qwen 2.5 3B, tiny/fast
    Cloud models (use HF router, has rate limits):
      google/gemma-4-31B-it         — Gemma 4 31B cloud
      Qwen/Qwen2.5-Coder-32B-Instruct — best for coding
      deepseek-ai/DeepSeek-V3-0324  — deep reasoning
    """
    response = get_client(model).chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4096,
    )
    return response.choices[0].message.content

@mcp.tool()
def ask_hf_with_context(system: str, prompt: str, model: str = DEFAULT_MODEL) -> str:
    """Send a prompt with a system prompt. Default: gemma4 (local, free). Same model options as ask_hf."""
    response = get_client(model).chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        max_tokens=4096,
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    mcp.run(transport="stdio")
