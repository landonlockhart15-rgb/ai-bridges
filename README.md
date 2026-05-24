# AI Bridges for Claude Code

**Status:** Usable personal tooling / experimental bridge collection  
**Audience:** Claude Code users who want to route tasks to other AI providers, local models, or local smart-home tools without leaving the coding session.  
**Design goal:** Give Claude Code small, inspectable MCP servers for delegating work to the right model or local device.

AI Bridges is a collection of MCP servers that let Claude Code call other AI models and control TP-Link Kasa smart-home devices from inside a Claude Code conversation.

Out of the box, Claude Code uses one model. These bridges add optional routes to GPT, Groq, Gemini, Hugging Face/Ollama, OpenRouter, and local Kasa devices.

## Fastest path: try one bridge first

Start with **Groq Bridge** because it is simple, fast, and has a free tier.

```bash
git clone https://github.com/landonlockhart15-rgb/ai-bridges.git
cd ai-bridges
bash setup.sh
```

On Windows PowerShell:

```powershell
git clone https://github.com/landonlockhart15-rgb/ai-bridges.git
cd ai-bridges
.\setup.ps1
```

Then set one key:

```bash
export GROQ_API_KEY="your-key"
```

Windows PowerShell:

```powershell
[System.Environment]::SetEnvironmentVariable("GROQ_API_KEY", "your-key", "User")
```

Register only the Groq bridge first using the `claude mcp add` command printed by the setup script. Restart Claude Code and try a simple prompt such as:

```text
Use Groq to summarize what this repository does.
```

Once one bridge works, add the others you actually need.

## Why this exists

Different tasks deserve different tools. Some work needs stronger reasoning, some needs a cheaper/free model, some should stay local, and some should control a device on your LAN. AI Bridges makes those options available as Claude Code tools.

## What's included

| Bridge | What it does | Cost profile |
|--------|--------------|--------------|
| **gpt-bridge** | Calls OpenAI GPT models | Pay per token |
| **groq-bridge** | Calls Llama, Qwen, and Gemma models through Groq | Free tier available |
| **gemini-bridge** | Calls Google Gemini models | Free tier available |
| **hf-bridge** | Calls local Ollama models or Hugging Face cloud models | Local/free options |
| **openrouter-bridge** | Routes to many models through OpenRouter | Free and paid models |
| **kasa-bridge** | Discovers and controls TP-Link Kasa devices by name | Local network |

## Use cases

- Route simple tasks to cheaper or free models.
- Ask a second model for another perspective.
- Use local Ollama models for offline/private tasks.
- Control local Kasa smart devices from a Claude Code workflow.
- Compare providers without manually switching tools.

## Full setup

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

The setup script creates a virtual environment for each bridge and installs its dependencies. It also prints the `claude mcp add` commands needed to register each bridge with Claude Code.

### 3. Set API keys for the bridges you want

Each bridge reads its key from an environment variable.

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

| Bridge | Environment variable | Where to get the key |
|--------|----------------------|----------------------|
| gpt-bridge | `OPENAI_API_KEY` | OpenAI Platform |
| groq-bridge | `GROQ_API_KEY` | Groq Console |
| gemini-bridge | `GEMINI_API_KEY` | Google AI Studio |
| hf-bridge | `HF_TOKEN` | Hugging Face tokens |
| openrouter-bridge | `OPENROUTER_API_KEY` | OpenRouter keys |
| kasa-bridge | No key needed | Local network |

### 4. Register with Claude Code

Run the `claude mcp add` commands printed by the setup script, then restart Claude Code. Each bridge should appear as a new set of tools.

Manual registration examples:

**Windows:**

```powershell
claude mcp add gpt-bridge         -- "C:/ai-bridges/gpt-bridge/venv/Scripts/python.exe"         "C:/ai-bridges/gpt-bridge/server.py"
claude mcp add groq-bridge        -- "C:/ai-bridges/groq-bridge/venv/Scripts/python.exe"        "C:/ai-bridges/groq-bridge/server.py"
claude mcp add gem-bridge         -- "C:/ai-bridges/gemini-bridge/venv/Scripts/python.exe"      "C:/ai-bridges/gemini-bridge/server.py"
claude mcp add hf-bridge          -- "C:/ai-bridges/hf-bridge/venv/Scripts/python.exe"          "C:/ai-bridges/hf-bridge/server.py"
claude mcp add openrouter-bridge  -- "C:/ai-bridges/openrouter-bridge/venv/Scripts/python.exe"  "C:/ai-bridges/openrouter-bridge/server.py"
claude mcp add kasa-bridge        -- "C:/ai-bridges/kasa-bridge/venv/Scripts/python.exe"        "C:/ai-bridges/kasa-bridge/server.py"
```

**Mac / Linux:** Replace `venv/Scripts/python.exe` with `venv/bin/python`.

## Available tools

### gpt-bridge

| Tool | Description |
|------|-------------|
| `ask_gpt(prompt, model)` | Send a prompt to GPT. |
| `ask_gpt_with_context(system, prompt, model)` | Send with a custom system prompt. |

### groq-bridge

| Tool | Description |
|------|-------------|
| `ask_groq(prompt, model)` | Send a prompt to Groq. |
| `ask_groq_with_context(system, prompt, model)` | Send with a system prompt. |

Example Groq models: `llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, `qwen-qwq-32b`, `gemma2-9b-it`.

### gemini-bridge

| Tool | Description |
|------|-------------|
| `ask_gemini(prompt, model)` | Send a prompt to Gemini. |
| `ask_gemini_with_context(system, prompt, model)` | Send with a system prompt. |

### hf-bridge

| Tool | Description |
|------|-------------|
| `ask_hf(prompt, model)` | Send to a local Ollama model or Hugging Face cloud model. |
| `ask_hf_with_context(system, prompt, model)` | Send with a system prompt. |

Local models require Ollama. Pull a model first, for example:

```bash
ollama pull qwen2.5:3b
```

Cloud Hugging Face models require `HF_TOKEN`.

### openrouter-bridge

| Tool | Description |
|------|-------------|
| `ask_openrouter(prompt, model)` | Send to an OpenRouter model. |
| `ask_openrouter_with_context(system, prompt, model)` | Send with a system prompt. |

### kasa-bridge

| Tool | Description |
|------|-------------|
| `discover_devices()` | Find Kasa devices on your local network. |
| `turn_on(device_name)` | Turn a device on by name. |
| `turn_off(device_name)` | Turn a device off by name. |
| `set_brightness(device_name, brightness)` | Set brightness 0–100 on a smart bulb. |
| `get_status(device_name)` | Get current on/off state and brightness. |

Device names are matched loosely, so `office light` can match a device named `Office Light Strip`.

## Project structure

```text
ai-bridges/
├── setup.ps1
├── setup.sh
├── gpt-bridge/
├── groq-bridge/
├── gemini-bridge/
├── hf-bridge/
├── openrouter-bridge/
└── kasa-bridge/
```

## Security notes

- Do not commit API keys, `.env` files, tokens, or local credentials.
- Prefer environment variables for provider keys.
- The Kasa bridge is for devices on networks you own or administer.
- Review each bridge before giving it access to paid APIs or local devices.

## Contributing

Pull requests are welcome. Provider-specific bridges should stay small, inspectable, and easy to run locally.

## License

MIT
