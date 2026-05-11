# AI Bridges for Claude Code

A collection of MCP (Model Context Protocol) servers that give Claude Code the ability to call other AI models and control smart home devices — all from inside your Claude Code conversation.

Out of the box, Claude Code can only use itself. These bridges change that. Install one or all of them and Claude Code can delegate tasks to free AI models, call GPT when it needs a second opinion, run inference locally on your own machine, or turn your lights on and off — without ever leaving the conversation.

---

## What's Included

| Bridge | What it does | Cost |
|--------|-------------|------|
| **gpt-bridge** | Call OpenAI GPT models (GPT-4o, GPT-4o-mini, etc.) | Pay per token |
| **groq-bridge** | Call Llama, Qwen, and Gemma models via Groq's cloud | Free |
| **gemini-bridge** | Call Google Gemini models (2.5 Flash, 2.5 Pro) | Free tier available |
| **hf-bridge** | Run local Ollama models OR call HuggingFace cloud models | Free |
| **openrouter-bridge** | Access 50+ models through one API, including many free tiers | Free models available |
| **kasa-bridge** | Discover and control TP-Link Kasa smart devices by name | Free (local network) |

---

## Why Use This?

Claude Code is powerful, but it's one model at one price point. With these bridges you can:

- **Cut costs** — let a free Groq or Gemini model handle routine tasks; only use Claude/GPT for hard reasoning
- **Get second opinions** — ask two different models the same question and compare
- **Run fully offline** — hf-bridge connects to local Ollama models, no internet required
- **Automate your home** — tell Claude "turn off the office lights" and it just works

---

## Requirements

- [Claude Code](https://claude.ai/code) installed and running
- Python 3.10+
- A terminal / command prompt

---

## Installation

Each bridge is a standalone Python MCP server. Pick the ones you want.

### 1. Clone the repo

```bash
git clone https://github.com/landonlockhart15-rgb/ai-bridges.git
cd ai-bridges
```

### 2. Set up a virtual environment for each bridge you want

```bash
cd gpt-bridge          # or groq-bridge, gemini-bridge, etc.
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Mac/Linux
pip install -r requirements.txt
```

Repeat for each bridge you want to use.

### 3. Set your API keys

Each bridge reads its key from an environment variable. The easiest way is to add them to your system environment variables, or put them in a `.env` file in each bridge folder.

| Bridge | Environment Variable | Where to get the key |
|--------|---------------------|----------------------|
| gpt-bridge | `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com) |
| groq-bridge | `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) |
| gemini-bridge | `GEMINI_API_KEY` | [aistudio.google.com](https://aistudio.google.com) |
| hf-bridge | `HF_TOKEN` | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| openrouter-bridge | `OPENROUTER_API_KEY` | [openrouter.ai/keys](https://openrouter.ai/keys) |
| kasa-bridge | *(none — uses local network)* | — |

**hf-bridge** can also run local models via [Ollama](https://ollama.com) without any API key. Just install Ollama and pull a model (`ollama pull qwen2.5:3b`) and it will automatically route local models to your machine.

### 4. Register each bridge with Claude Code

Open your terminal and run one `claude mcp add` command per bridge. Replace the path with wherever you cloned the repo.

```bash
# GPT bridge
claude mcp add gpt-bridge -- "C:/path/to/ai-bridges/gpt-bridge/venv/Scripts/python.exe" "C:/path/to/ai-bridges/gpt-bridge/server.py"

# Groq bridge
claude mcp add groq-bridge -- "C:/path/to/ai-bridges/groq-bridge/venv/Scripts/python.exe" "C:/path/to/ai-bridges/groq-bridge/server.py"

# Gemini bridge
claude mcp add gem-bridge -- "C:/path/to/ai-bridges/gemini-bridge/venv/Scripts/python.exe" "C:/path/to/ai-bridges/gemini-bridge/server.py"

# HuggingFace / Ollama bridge
claude mcp add hf-bridge -- "C:/path/to/ai-bridges/hf-bridge/venv/Scripts/python.exe" "C:/path/to/ai-bridges/hf-bridge/server.py"

# OpenRouter bridge
claude mcp add openrouter-bridge -- "C:/path/to/ai-bridges/openrouter-bridge/venv/Scripts/python.exe" "C:/path/to/ai-bridges/openrouter-bridge/server.py"

# Kasa smart home bridge
claude mcp add kasa-bridge -- "C:/path/to/ai-bridges/kasa-bridge/venv/Scripts/python.exe" "C:/path/to/ai-bridges/kasa-bridge/server.py"
```

> **Mac/Linux:** Replace `venv/Scripts/python.exe` with `venv/bin/python`

After registering, restart Claude Code. Each bridge will appear as a set of available tools.

---

## Available Tools

### gpt-bridge
| Tool | Description |
|------|-------------|
| `ask_gpt(prompt, model)` | Send a prompt to GPT. Default model: `gpt-4o-mini` |
| `ask_gpt_with_context(system, prompt, model)` | Send with a custom system prompt |

### groq-bridge
| Tool | Description |
|------|-------------|
| `ask_groq(prompt, model)` | Send a prompt to Groq. Default: `llama-3.3-70b-versatile` |
| `ask_groq_with_context(system, prompt, model)` | Send with a system prompt |

**Available Groq models:** `llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, `qwen-qwq-32b`, `gemma2-9b-it`

### gemini-bridge
| Tool | Description |
|------|-------------|
| `ask_gemini(prompt, model)` | Send a prompt to Gemini. Default: `gemini-2.5-flash` |
| `ask_gemini_with_context(system, prompt, model)` | Send with a system prompt |

### hf-bridge
| Tool | Description |
|------|-------------|
| `ask_hf(prompt, model)` | Send to a local Ollama model or HuggingFace cloud model |
| `ask_hf_with_context(system, prompt, model)` | Send with a system prompt |

**Local models** (no API key, requires Ollama): `qwen2.5:3b`, `llama3.2:latest`, any model you've pulled  
**Cloud models** (requires HF token): `Qwen/Qwen2.5-Coder-32B-Instruct`, `deepseek-ai/DeepSeek-V3-0324`, `google/gemma-3-27b-it`

### openrouter-bridge
| Tool | Description |
|------|-------------|
| `ask_openrouter(prompt, model)` | Send to any OpenRouter model |
| `ask_openrouter_with_context(system, prompt, model)` | Send with a system prompt |

**Free models:** `nvidia/nemotron-3-super-120b-a12b:free`, `google/gemma-4-31b-it:free`, `google/gemma-4-26b-a4b-it:free`, `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free`  
Each free model has its own independent rate limit — if one is busy, switch to another.

### kasa-bridge
| Tool | Description |
|------|-------------|
| `discover_devices()` | Find all Kasa devices on your network |
| `turn_on(device_name)` | Turn a device on by name |
| `turn_off(device_name)` | Turn a device off by name |
| `set_brightness(device_name, brightness)` | Set brightness 0–100 on a smart bulb |
| `get_status(device_name)` | Get current state and brightness of a device |

---

## Optional: Configure OpenRouter Headers

By default openrouter-bridge identifies itself generically. You can customize it with environment variables:

```bash
OPENROUTER_REFERER=https://github.com/yourusername/your-project
OPENROUTER_TITLE=My Claude Setup
```

---

## How It Works

Each bridge is a tiny Python server that speaks the MCP protocol over `stdio`. When Claude Code starts, it launches the server as a subprocess and keeps it running in the background. Any time Claude Code calls one of the tools (like `ask_groq`), the bridge receives the request, calls the appropriate API, and returns the result — all transparently inside your conversation.

This means Claude Code can use these tools the same way it uses any built-in tool: automatically, in parallel, and without you having to do anything once it's set up.

---

## Project Structure

```
ai-bridges/
├── gpt-bridge/
│   ├── server.py          # MCP server — GPT via OpenAI SDK
│   └── requirements.txt
├── groq-bridge/
│   └── server.py          # MCP server — Groq cloud (free)
├── gemini-bridge/
│   ├── server.py          # MCP server — Google Gemini
│   └── requirements.txt
├── hf-bridge/
│   └── server.py          # MCP server — local Ollama + HuggingFace cloud
├── openrouter-bridge/
│   └── server.py          # MCP server — 50+ models via OpenRouter
└── kasa-bridge/
    └── server.py          # MCP server — TP-Link Kasa smart devices
```

---

## Contributing

Pull requests welcome. If you build a bridge for another provider (Anthropic direct, Cohere, Mistral, etc.), feel free to open a PR.

---

## License

MIT
