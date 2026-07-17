import os
import sys
import json
import time
import socket
import urllib.request
import urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import bridge_state

IS_TEST = "unittest" in sys.modules or "pytest" in sys.modules or os.environ.get("BRIDGE_TESTING") == "1"


def encrypt_kasa(string):
    """Simple XOR encryption used by Kasa devices for discovery."""
    key = 171
    payload = bytearray()
    for i in string.encode('utf-8'):
        a = key ^ i
        key = a
        payload.append(a)
    return payload

def mask_key(key):
    """Mask API key for security."""
    if not key:
        return "Not Configured"
    if key.startswith("gsk_"):
        return f"gsk_...{key[-4:]}" if len(key) >= 8 else "gsk_..."
    if len(key) <= 8:
        return f"{key[:2]}...{key[-2:]}"
    return f"{key[:4]}...{key[-4:]}"

def validate_key(name, key):
    """Validate API key format without hitting the API."""
    if not key:
        return False, "Environment variable is not set"
    if any(c.isspace() for c in key):
        return False, "API key must not contain whitespace"
    if len(key) < 10:
        return False, "API key is too short"
    
    if name == "groq-bridge":
        if not key.startswith("gsk_"):
            return False, "GROQ_API_KEY must start with 'gsk_'"
        if len(key) < 15:
            return False, "GROQ_API_KEY must be at least 15 characters long"
            
    return True, "Valid format"

def check_registration():
    """Detect which bridges are registered in Claude Code or Claude Desktop config files."""
    registered = {
        "gpt-bridge": False,
        "groq-bridge": False,
        "gemini-bridge": False,
        "hf-bridge": False,
        "openrouter-bridge": False,
        "kasa-bridge": False,
        "cerebras-bridge": False
    }
    
    paths = []
    # 1. Project-level .mcp.json
    paths.append(Path(".") / ".mcp.json")
    
    # 2. User-level ~/.claude.json (Claude Code config)
    try:
        paths.append(Path.home() / ".claude.json")
    except Exception:
        pass
        
    # 3. Claude Desktop config
    try:
        if os.name == "nt":
            appdata = os.environ.get("APPDATA")
            if appdata:
                paths.append(Path(appdata) / "Claude" / "claude_desktop_config.json")
        else:
            paths.append(Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json")
            paths.append(Path.home() / ".config" / "Claude" / "claude_desktop_config.json")
    except Exception:
        pass

    for path in paths:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    mcp_servers = config.get("mcpServers", {})
                    for name in mcp_servers:
                        if name == "gpt-bridge":
                            registered["gpt-bridge"] = True
                        elif name == "groq-bridge":
                            registered["groq-bridge"] = True
                        elif name in ("gem-bridge", "gemini-bridge"):
                            registered["gemini-bridge"] = True
                        elif name == "hf-bridge":
                            registered["hf-bridge"] = True
                        elif name == "openrouter-bridge":
                            registered["openrouter-bridge"] = True
                        elif name == "kasa-bridge":
                            registered["kasa-bridge"] = True
                        elif name == "cerebras-bridge":
                            registered["cerebras-bridge"] = True
            except Exception:
                pass
    return registered

def ping_api(url, headers=None, key_present=True):
    """Ping an API endpoint and measure connection latency."""
    if not key_present:
        return False, None, "API key is not configured"
        
    start = time.time()
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=3.0) as response:
            latency = int((time.time() - start) * 1000)
            return True, latency, "Successfully connected."
    except urllib.error.HTTPError as e:
        latency = int((time.time() - start) * 1000)
        if e.code in (401, 403):
            return False, latency, f"Authentication failed (HTTP {e.code}). Invalid API key."
        elif e.code == 429:
            return False, latency, "Rate limit hit (HTTP 429)."
        else:
            return False, latency, f"HTTP Error {e.code} received."
    except urllib.error.URLError as e:
        return False, None, f"Connection failed (Host unreachable)."
    except Exception as e:
        return False, None, f"Unexpected error: {str(e)}"

def check_ollama():
    """Verify if local Ollama is running and list any active local models."""
    url = "http://localhost:11434/api/tags"
    start = time.time()
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=1.0) as response:
            latency = int((time.time() - start) * 1000)
            data = json.loads(response.read().decode('utf-8'))
            models = [m['name'] for m in data.get('models', [])]
            if models:
                models_str = ", ".join(models[:3]) + (f" (+{len(models)-3} more)" if len(models) > 3 else "")
                return True, latency, f"Ollama is running. Models: {models_str}"
            return True, latency, "Ollama is running, but no models are downloaded."
    except Exception:
        return False, None, "Ollama is offline (Connection refused)."

def check_kasa():
    """Perform a lightweight UDP broadcast discovery for TP-Link Kasa devices."""
    start = time.time()
    try:
        payload = encrypt_kasa('{"system":{"get_sysinfo":{}}}')
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.settimeout(0.5)
        sock.sendto(payload, ('255.255.255.255', 9999))
        
        devices = []
        for _ in range(5):
            try:
                data, addr = sock.recvfrom(2048)
                devices.append(addr[0])
            except socket.timeout:
                break
        
        latency = int((time.time() - start) * 1000)
        if devices:
            unique_devices = sorted(list(set(devices)))
            return True, latency, f"Online. Discovered {len(unique_devices)} device(s): {', '.join(unique_devices)}"
        return True, latency, "Online. No Kasa devices responded on local network."
    except Exception as e:
        return False, None, f"Local discovery skipped or failed: {str(e)}"

def run_bridge_diagnostic(bridge_id, reg_status):
    """Run diagnostic checks for a specific bridge."""
    env_vars = {
        "gpt-bridge": "OPENAI_API_KEY",
        "groq-bridge": "GROQ_API_KEY",
        "gemini-bridge": "GEMINI_API_KEY",
        "openrouter-bridge": "OPENROUTER_API_KEY",
        "cerebras-bridge": "CEREBRAS_API_KEY",
        "hf-bridge": "HF_TOKEN",
        "kasa-bridge": None
    }
    
    env_var = env_vars.get(bridge_id)
    key = os.environ.get(env_var) if env_var else None
    
    # Check key configuration
    key_valid = True
    key_details = "N/A"
    if env_var:
        key_valid, key_details = validate_key(bridge_id, key)
        
    masked = mask_key(key) if env_var else "N/A"
    
    # Availability & Latency
    status = "🔴 Offline"
    latency_str = "—"
    details = "Unknown status"
    
    if env_var and not key and bridge_id != "hf-bridge":
        status = "🔴 Key Missing"
        details = f"Required environment variable '{env_var}' is not set."
    elif env_var and not key_valid and bridge_id != "hf-bridge":
        status = "🔴 Config Error"
        details = f"Invalid API Key format: {key_details}"
    else:
        # Perform network check
        success = False
        latency = None
        msg = ""
        if bridge_id == "gpt-bridge":
            success, latency, msg = ping_api(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {key}"}
            )
        elif bridge_id == "groq-bridge":
            success, latency, msg = ping_api(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {key}"}
            )
        elif bridge_id == "gemini-bridge":
            success, latency, msg = ping_api(
                f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
            )
        elif bridge_id == "openrouter-bridge":
            success, latency, msg = ping_api(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {key}"}
            )
        elif bridge_id == "cerebras-bridge":
            success, latency, msg = ping_api(
                "https://api.cerebras.ai/v1/models",
                headers={"Authorization": f"Bearer {key}"}
            )
        elif bridge_id == "hf-bridge":
            # hf-bridge is a dual check: check local Ollama and Hugging Face Cloud
            ollama_success, ollama_lat, ollama_msg = check_ollama()
            hf_success, hf_lat, hf_msg = ping_api(
                "https://router.huggingface.co/v1/models",
                headers={"Authorization": f"Bearer {key}"} if key else None,
                key_present=bool(key)
            )
            
            success = ollama_success or hf_success
            latency = ollama_lat if ollama_success else hf_lat
            
            if ollama_success and hf_success:
                status = "🟢 Online"
                details = f"Local Ollama online. Cloud Hugging Face online. ({hf_msg})"
            elif ollama_success:
                status = "🟡 Local Only"
                details = f"Local Ollama online. Cloud Hugging Face unavailable (Key missing/invalid)."
            elif hf_success:
                status = "🟢 Cloud Only"
                details = f"Cloud Hugging Face online. Local Ollama is offline."
            else:
                status = "🔴 Offline"
                details = f"Local Ollama offline. Cloud Hugging Face unavailable: {hf_msg}"
                
            latency_str = f"{latency} ms" if latency is not None else "—"
            return {
                "id": bridge_id,
                "status": status,
                "latency": latency_str,
                "key_configured": "Yes" if key else "No",
                "key_masked": masked,
                "registered": reg_status,
                "details": details
            }
        elif bridge_id == "kasa-bridge":
            success, latency, msg = check_kasa()
            
        if success:
            status = "🟢 Online"
            details = msg
            latency_str = f"{latency} ms"
        else:
            status = status if status.startswith("🔴") else "🔴 Offline"
            details = msg
            latency_str = f"{latency} ms" if latency is not None else "—"

    return {
        "id": bridge_id,
        "status": status,
        "latency": latency_str,
        "key_configured": "Yes" if key else "No" if env_var else "N/A",
        "key_masked": masked,
        "registered": reg_status,
        "details": details
    }

def check_bridges() -> str:
    """Check availability, latency, keys, and registration of all AI bridges, returning a Markdown report."""
    # First get registrations
    registrations = check_registration()
    
    bridges = [
        ("gpt-bridge", "OpenAI (GPT)"),
        ("groq-bridge", "Groq"),
        ("gemini-bridge", "Gemini"),
        ("openrouter-bridge", "OpenRouter"),
        ("cerebras-bridge", "Cerebras"),
        ("hf-bridge", "Hugging Face / Ollama"),
        ("kasa-bridge", "TP-Link Kasa")
    ]
    
    # Run diagnostic checks in parallel
    with ThreadPoolExecutor(max_workers=len(bridges)) as executor:
        futures = [
            executor.submit(run_bridge_diagnostic, b_id, registrations.get(b_id, False))
            for b_id, _ in bridges
        ]
        results = [f.result() for f in futures]

    # Write to status cache
    try:
        cache_data = {
            "timestamp": time.time(),
            "bridges": {res["id"]: res for res in results}
        }
        bridge_state.write_status_cache(cache_data)
    except Exception:
        pass
        
    # Build clean, high-aesthetic markdown report
    md = []
    md.append("# 🛠️ AI Bridges Unified Diagnostics")
    md.append("")
    md.append("This report displays the configuration, registration, connection state, and latency of each bridge in this repository.")
    md.append("")
    
    # Status Summary Block
    online_count = sum(1 for r in results if "Online" in r["status"] or "Local Only" in r["status"] or "Cloud Only" in r["status"])
    md.append(f"> [!NOTE]")
    md.append(f"> **System Health:** {online_count} / {len(results)} bridges available.")
    md.append(f"> Run `claude mcp add` to register unregistered bridges. See [README.md](file:///C:/Users/lock_/AppData/Local/AI-Hub/worktrees/ai-bridges-auto-20260616-063219/README.md) for environment variable setup.")
    md.append("")
    
    # Table headers
    md.append("| Bridge / Provider | Status | Latency | Key Configured | Key Masked | Registered in Claude | Details / Diagnostics |")
    md.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    
    bridge_names = dict(bridges)
    for res in results:
        b_name = bridge_names[res["id"]]
        reg_emoji = "🟢 Yes" if res["registered"] else "⚪ No"
        md.append(
            f"| **{b_name}** (`{res['id']}`) "
            f"| {res['status']} "
            f"| `{res['latency']}` "
            f"| {res['key_configured']} "
            f"| `{res['key_masked']}` "
            f"| {reg_emoji} "
            f"| {res['details']} |"
        )
        
    md.append("")
    
    # Additional system details
    md.append("### 💻 Local Environment Check")
    # Check local Ollama specifically
    ollama_ok, _, ollama_details = check_ollama()
    ollama_status = "🟢 Running" if ollama_ok else "🔴 Offline"
    md.append(f"- **Ollama service:** {ollama_status} ({ollama_details})")
    
    # Local network kasa ping
    kasa_ok, _, kasa_details = check_kasa()
    kasa_status = "🟢 Network scan OK" if kasa_ok else "⚪ Local scan skipped/none found"
    md.append(f"- **Smart Home (Kasa) scan:** {kasa_status} — {kasa_details}")
    
    return "\n".join(md)

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    print(check_bridges())
