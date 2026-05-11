# AI Bridges for Claude Code

MCP servers that give Claude Code the ability to call other AI models and control TP-Link Kasa smart home devices — all from inside your Claude Code conversation.

Out of the box, Claude Code can only use itself. These bridges change that. Install one or all of them and Claude Code can delegate tasks to free AI models, call GPT when it needs a second opinion, run inference locally on your own machine, or turn your lights on and off — without ever leaving the conversation.

---

## What's Included

| Bridge | What it does | Cost |
|--------|-------------|------|
| **gpt-bridge** | Call OpenAI GPT models | Pay per token |
| **groq-bridge** | Call Llama, Qwen, and Gemma models via Groq | **Free** |
| **gemini-bridge** | Call Google Gemini 2.5 Flash / Pro | **Free tier available** |
| **hf-bridge** | Run local Ollama models OR call HuggingFace cloud models | **Free** |
| **openrouter-bridge** | Access 50+ models through one API, many with free tiers | **Free models available** |
| **kasa-bridge** | Discover and control TP-Link Kasa smart devices by name | **Free (local network)** |

---

## Why Use This?

Claude Code is powerful but it's one model at one price point. With these bridges you can:

- **Cut costs** — route simple tasks to a free Groq or Gemini model and save Claude for hard reasoning
- **Get second opinions** — ask two different models the same question and compare answers
- **Run fully offline** — hf-bridge connects to local Ollama models, no internet required
- **Automate your home** — tell Claude "turn off the office lights" and it just works

---

## Quick Setup

### 1. Clone the repo

```bash
git clone https://github.com/landonlockhart15-rgb/ai-bridges.git
cd ai-bridges
```

### 2. Run the setup script

**Windows (PowerShell):**
```powershell
.\setup.ps1
```

**Mac / Linux:**
```bash
bash setup.sh
```

This creates a virtual environment for each bridge and installs its dependencies. The script also prints the exact `claude mcp add` commands to register each bridge with Claude Code, using the correct paths for your machine.

### 3. Set your API keys

Each bridge reads its key from an environment variable. Set the ones for the bridges you want to use.

**Windows (PowerShell) — sets permanently for your user account:**
```powershell
[System.Environment]::SetEnvironmentVariable("OPENAI_API_KEY",      "your-key", "User")
[System.Environment]::SetEnvironmentVariable("GROQ_API_KEY",        "your-key", "User")
[System.Environment]::SetEnvironmentVariable("GEMINI_API_KEY",      "your-key", "User")
[System.Environment]::SetEnvironmentVariable("HF_TOKEN",            "your-key", "User")
[System.Environment]::SetEnvironmentVariable("OPENROUTER_API_KEY",  "your-key", "User")
```

**Mac / Linux — add to your `~/.zshrc` or `~/.bashrc`:**
```bash
export OPENAI_API_KEY="your-key"
export GROQ_API_KEY="your-key"
export GEMINI_API_KEY="your-key"
export HF_TOKEN="your-key"
export OPENROUTER_API_KEY="your-key"
```

| Bridge | Environment Variable | Where to get the key |
|--------|---------------------|----------------------|
| gpt-bridge | `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com) |
| groq-bridge | `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) — free |
| gemini-bridge | `GEMINI_API_KEY` | [aistudio.google.com](https://aistudio.google.com) — free |
| hf-bridge | `HF_TOKEN` | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) — free |
| openrouter-bridge | `OPENROUTER_API_KEY` | [openrouter.ai/keys](https://openrouter.ai/keys) — free account |
| kasa-bridge | *(no key needed)* | Communicates directly over your local network |

### 4. Register with Claude Code

Run the `claude mcp add` commands printed by the setup script, then restart Claude Code. Each bridge will appear as a new set of tools.

If you skipped the setup script, the manual commands look like this:

**Windows:**
```powershell
claude mcp add gpt-bridge         -- "C:/ai-bridges/gpt-bridge/venv/Scripts/python.exe"         "C:/ai-bridges/gpt-bridge/server.py"
claude mcp add groq-bridge        -- "C:/ai-bridges/groq-bridge/venv/Scripts/python.exe"        "C:/ai-bridges/groq-bridge/server.py"
claude mcp add gem-bridge         -- "C:/ai-bridges/gemini-bridge/venv/Scripts/python.exe"      "C:/ai-bridges/gemini-bridge/server.py"
claude mcp add hf-bridge          -- "C:/ai-bridges/hf-bridge/venv/Scripts/python.exe"          "C:/ai-bridges/hf-bridge/server.py"
claude mcp add openrouter-bridge  -- "C:/ai-bridges/openrouter-bridge/venv/Scripts/python.exe"  "C:/ai-bridges/openrouter-bridge/server.py"
claude mcp add kasa-bridge        -- "C:/ai-bridges/kasa-bridge/venv/Scripts/python.exe"        "C:/ai-bridges/kasa-bridge/server.py"
```

**Mac / Linux:** Replace `venv/Scripts/python.exe` with `venv/bin/python` in every command above.

---

## Available Tools

### gpt-bridge
| Tool | Description |
|------|-------------|
| `ask_gpt(prompt, model)` | Send a prompt to GPT. Default: `gpt-4o-mini` |
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

**Local models — requires [Ollama](https://ollama.com) installed and running:**
Pull a model first with `ollama pull qwen2.5:3b`, then pass it by name (no slash).
Examples: `qwen2.5:3b`, `llama3.2:latest`, `gemma3:4b`

**Cloud models — requires HF_TOKEN, pass the full repo ID with a slash:**
Examples: `Qwen/Qwen2.5-Coder-32B-Instruct`, `deepseek-ai/DeepSeek-V3-0324`, `google/gemma-3-27b-it`

> **Don't have Ollama?** Pass a cloud model ID (with a `/` in the name) and the bridge will route to HuggingFace automatically.

### openrouter-bridge
| Tool | Description |
|------|-------------|
| `ask_openrouter(prompt, model)` | Send to any OpenRouter model |
| `ask_openrouter_with_context(system, prompt, model)` | Send with a system prompt |

**Free models (append `:free` to use the no-cost tier):**
`nvidia/nemotron-3-super-120b-a12b:free`, `google/gemma-4-31b-it:free`, `google/gemma-4-26b-a4b-it:free`, `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free`

Each free model has its own independent rate limit — if one is busy, switch to another.

### kasa-bridge
| Tool | Description |
|------|-------------|
| `discover_devices()` | Find all Kasa devices on your local network |
| `turn_on(device_name)` | Turn a device on by name |
| `turn_off(device_name)` | Turn a device off by name |
| `set_brightness(device_name, brightness)` | Set brightness 0–100 on a smart bulb |
| `get_status(device_name)` | Get current on/off state and brightness |

Device names are matched loosely — "office light" will match a device named "Office Light Strip".

---

## How It Works

Each bridge is a small Python server that speaks the MCP (Model Context Protocol) over `stdio`. When Claude Code starts, it launches each registered bridge as a background subprocess. Any time Claude Code calls a tool like `ask_groq`, the bridge receives the request, calls the appropriate API, and returns the result — transparently, inside your conversation.

This means Claude Code can use these bridges the same way it uses any built-in tool: automatically, in parallel, and without you having to do anything once setup is complete.

---

## Optional: Customize OpenRouter Headers

OpenRouter accepts optional headers that identify your app. Set these environment variables to customize them:

```bash
OPENROUTER_REFERER=https://github.com/yourusername/your-project
OPENROUTER_TITLE=My Claude Setup
```

---

## Project Structure

```
ai-bridges/
├── setup.ps1                  # One-shot setup script (Windows)
├── setup.sh                   # One-shot setup script (Mac/Linux)
├── gpt-bridge/
│   ├── server.py              # MCP server — GPT via OpenAI SDK
│   └── requirements.txt
├── groq-bridge/
│   ├── server.py              # MCP server — Groq cloud (free)
│   └── requirements.txt
├── gemini-bridge/
│   ├── server.py              # MCP server — Google Gemini
│   └── requirements.txt
├── hf-bridge/
│   ├── server.py              # MCP server — local Ollama + HuggingFace cloud
│   └── requirements.txt
├── openrouter-bridge/
│   ├── server.py              # MCP server — 50+ models via OpenRouter
│   └── requirements.txt
└── kasa-bridge/
    ├── server.py              # MCP server — TP-Link Kasa smart devices
    └── requirements.txt
```

---

## Contributing

Pull requests welcome. If you build a bridge for another provider, feel free to open a PR.

---

## License

MIT
