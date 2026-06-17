import os
import sys
import json
import time
import datetime

# Check if we are running unit tests
IS_TEST = "unittest" in sys.modules or "pytest" in sys.modules or os.environ.get("BRIDGE_TESTING") == "1"

# Resolve file paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if IS_TEST:
    USAGE_FILE_PATH = os.path.join(ROOT_DIR, ".bridge_usage_test.json")
else:
    USAGE_FILE_PATH = os.path.join(ROOT_DIR, ".bridge_usage.json")

LOCK_FILE_PATH = USAGE_FILE_PATH + ".lock"

# Standard pricing dict (in USD per single token)
PRICES = {
    "gpt-4o-mini": {"input": 0.15 / 1000000.0, "output": 0.60 / 1000000.0},
    "gpt-4o": {"input": 2.50 / 1000000.0, "output": 10.00 / 1000000.0},
    "gpt-5.5": {"input": 5.00 / 1000000.0, "output": 15.00 / 1000000.0},
    "gemini-2.5-flash": {"input": 0.0, "output": 0.0},
    "gemini-2.5-pro": {"input": 1.25 / 1000000.0, "output": 5.00 / 1000000.0},
}

class SimpleFileLock:
    def __init__(self, lock_file_path, timeout=5.0):
        self.lock_file_path = lock_file_path
        self.timeout = timeout
        self.is_locked = False

    def __enter__(self):
        start_time = time.time()
        while time.time() - start_time < self.timeout:
            try:
                fd = os.open(self.lock_file_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                self.is_locked = True
                return self
            except FileExistsError:
                # Clean up stale locks older than 10 seconds
                try:
                    mtime = os.path.getmtime(self.lock_file_path)
                    if time.time() - mtime > 10.0:
                        try:
                            os.remove(self.lock_file_path)
                        except OSError:
                            pass
                except OSError:
                    pass
                time.sleep(0.05)
        raise TimeoutError(f"Could not acquire lock on {self.lock_file_path}")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.is_locked:
            try:
                os.remove(self.lock_file_path)
            except OSError:
                pass

def get_prices(model_name, provider=None):
    """Get input and output pricing per token."""
    if provider in ("groq-bridge", "cerebras-bridge", "kasa-bridge"):
        return 0.0, 0.0
    if provider == "hf-bridge" and "/" not in model_name:
        return 0.0, 0.0
    if ":free" in model_name:
        return 0.0, 0.0
        
    base_name = model_name.split("/")[-1].split(":")[0].lower()
    for key, price in PRICES.items():
        if key in base_name:
            return price["input"], price["output"]
            
    if provider == "hf-bridge":
        return 0.0, 0.0
        
    fallback_input = float(os.environ.get("FALLBACK_INPUT_PRICE_PER_1M", "2.50")) / 1000000.0
    fallback_output = float(os.environ.get("FALLBACK_OUTPUT_PRICE_PER_1M", "10.00")) / 1000000.0
    return fallback_input, fallback_output

def load_usage_db():
    """Load the current usage tracking database."""
    if not os.path.exists(USAGE_FILE_PATH):
        return {
            "config": {
                "daily_budget_cap": None,
                "monthly_budget_cap": None
            },
            "daily": {},
            "monthly": {},
            "providers": {},
            "models": {},
            "logs": []
        }
    try:
        with open(USAGE_FILE_PATH, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return {
                    "config": {
                        "daily_budget_cap": None,
                        "monthly_budget_cap": None
                    },
                    "daily": {},
                    "monthly": {},
                    "providers": {},
                    "models": {},
                    "logs": []
                }
            return json.loads(content)
    except Exception:
        return {
            "config": {
                "daily_budget_cap": None,
                "monthly_budget_cap": None
            },
            "daily": {},
            "monthly": {},
            "providers": {},
            "models": {},
            "logs": []
        }

def save_usage_db(db):
    """Save the usage database atomically using a temp file."""
    try:
        temp_path = USAGE_FILE_PATH + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(db, f, indent=2)
        if os.path.exists(USAGE_FILE_PATH):
            os.remove(USAGE_FILE_PATH)
        os.rename(temp_path, USAGE_FILE_PATH)
    except Exception:
        try:
            with open(USAGE_FILE_PATH, "w", encoding="utf-8") as f:
                json.dump(db, f, indent=2)
        except Exception:
            pass

def get_budget_caps():
    """Retrieve daily and monthly budget caps from environment or config file."""
    daily_cap_env = os.environ.get("DAILY_BUDGET_CAP")
    monthly_cap_env = os.environ.get("MONTHLY_BUDGET_CAP")
    
    daily_cap = float(daily_cap_env) if daily_cap_env is not None and daily_cap_env != "" else None
    monthly_cap = float(monthly_cap_env) if monthly_cap_env is not None and monthly_cap_env != "" else None
    
    db = load_usage_db()
    config = db.get("config", {})
    
    if daily_cap is None:
        daily_cap = config.get("daily_budget_cap")
    if monthly_cap is None:
        monthly_cap = config.get("monthly_budget_cap")
        
    return daily_cap, monthly_cap

def check_budget(provider, model):
    """Raise ValueError if budget caps are exceeded."""
    input_price, output_price = get_prices(model, provider)
    if input_price == 0.0 and output_price == 0.0:
        return
        
    daily_cap, monthly_cap = get_budget_caps()
    db = load_usage_db()
    
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    current_month = datetime.datetime.now().strftime("%Y-%m")
    
    daily_usage = db.get("daily", {}).get(current_date, {}).get("cost", 0.0)
    monthly_usage = db.get("monthly", {}).get(current_month, {}).get("cost", 0.0)
    
    if daily_cap is not None and daily_usage >= daily_cap:
        raise ValueError(
            f"Daily budget cap exceeded for paid calls. Limit: ${daily_cap:.4f}, Current daily cost: ${daily_usage:.4f}."
        )
        
    if monthly_cap is not None and monthly_usage >= monthly_cap:
        raise ValueError(
            f"Monthly budget cap exceeded for paid calls. Limit: ${monthly_cap:.4f}, Current monthly cost: ${monthly_usage:.4f}."
        )

def record_usage(provider, model, prompt_tokens, completion_tokens):
    """Log the token usage and cost incurred."""
    try:
        prompt_tokens = int(prompt_tokens)
    except (TypeError, ValueError):
        prompt_tokens = 0
        
    try:
        completion_tokens = int(completion_tokens)
    except (TypeError, ValueError):
        completion_tokens = 0

    input_price, output_price = get_prices(model, provider)
    cost = (prompt_tokens * input_price) + (completion_tokens * output_price)
    total_tokens = prompt_tokens + completion_tokens
    
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    current_month = datetime.datetime.now().strftime("%Y-%m")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    
    with SimpleFileLock(LOCK_FILE_PATH):
        db = load_usage_db()
        
        # 1. Update daily
        daily = db.setdefault("daily", {})
        day_data = daily.setdefault(current_date, {"cost": 0.0, "tokens": 0, "input_tokens": 0, "output_tokens": 0})
        day_data["cost"] += cost
        day_data["tokens"] += total_tokens
        day_data["input_tokens"] += prompt_tokens
        day_data["output_tokens"] += completion_tokens
        
        # 2. Update monthly
        monthly = db.setdefault("monthly", {})
        month_data = monthly.setdefault(current_month, {"cost": 0.0, "tokens": 0, "input_tokens": 0, "output_tokens": 0})
        month_data["cost"] += cost
        month_data["tokens"] += total_tokens
        month_data["input_tokens"] += prompt_tokens
        month_data["output_tokens"] += completion_tokens
        
        # 3. Update providers
        providers = db.setdefault("providers", {})
        provider_data = providers.setdefault(provider, {"cost": 0.0, "tokens": 0, "input_tokens": 0, "output_tokens": 0})
        provider_data["cost"] += cost
        provider_data["tokens"] += total_tokens
        provider_data["input_tokens"] += prompt_tokens
        provider_data["output_tokens"] += completion_tokens
        
        # 4. Update models
        models = db.setdefault("models", {})
        model_data = models.setdefault(model, {"cost": 0.0, "tokens": 0, "input_tokens": 0, "output_tokens": 0})
        model_data["cost"] += cost
        model_data["tokens"] += total_tokens
        model_data["input_tokens"] += prompt_tokens
        model_data["output_tokens"] += completion_tokens
        
        # 5. Append log entry
        logs = db.setdefault("logs", [])
        logs.append({
            "timestamp": timestamp,
            "provider": provider,
            "model": model,
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "cost": cost
        })
        if len(logs) > 1000:
            db["logs"] = logs[-1000:]
            
        save_usage_db(db)

def get_bridge_costs(timeframe: str = "all") -> str:
    """Get a summary report of the tokens used and costs incurred across all AI bridges."""
    db = load_usage_db()
    daily_cap, monthly_cap = get_budget_caps()
    
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    current_month = datetime.datetime.now().strftime("%Y-%m")
    
    today_cost = db.get("daily", {}).get(current_date, {}).get("cost", 0.0)
    today_tokens = db.get("daily", {}).get(current_date, {}).get("tokens", 0)
    month_cost = db.get("monthly", {}).get(current_month, {}).get("cost", 0.0)
    month_tokens = db.get("monthly", {}).get(current_month, {}).get("tokens", 0)
    
    lines = []
    lines.append("# 📊 AI Bridges Usage & Cost Report")
    lines.append("")
    
    lines.append("## 🛡️ Budget Limits & Status")
    daily_status = f"${today_cost:.4f} / ${daily_cap:.4f}" if daily_cap is not None else f"${today_cost:.4f} / No Limit"
    monthly_status = f"${month_cost:.4f} / ${monthly_cap:.4f}" if monthly_cap is not None else f"${month_cost:.4f} / No Limit"
    
    if daily_cap is not None and today_cost >= daily_cap:
        daily_status += " ⚠️ EXCEEDED"
    if monthly_cap is not None and month_cost >= monthly_cap:
        monthly_status += " ⚠️ EXCEEDED"
        
    lines.append(f"- **Daily Budget Cap Status:** {daily_status}")
    lines.append(f"- **Monthly Budget Cap Status:** {monthly_status}")
    lines.append("")
    
    lines.append("## 📈 Cost and Usage Summary")
    lines.append("| Timeframe | Cost (USD) | Total Tokens |")
    lines.append("| :--- | :--- | :--- |")
    lines.append(f"| **Today ({current_date})** | ${today_cost:.4f} | {today_tokens:,} |")
    lines.append(f"| **This Month ({current_month})** | ${month_cost:.4f} | {month_tokens:,} |")
    
    lines.append("")
    lines.append("## 🔌 Usage by Provider")
    lines.append("| Provider | Cost (USD) | Input Tokens | Output Tokens | Total Tokens |")
    lines.append("| :--- | :--- | :--- | :--- | :--- |")
    
    providers_data = db.get("providers", {})
    if not providers_data:
        lines.append("| No data tracked yet | - | - | - | - |")
    else:
        for provider, data in sorted(providers_data.items(), key=lambda x: x[1].get("cost", 0.0), reverse=True):
            cost = data.get("cost", 0.0)
            tokens = data.get("tokens", 0)
            in_t = data.get("input_tokens", 0)
            out_t = data.get("output_tokens", 0)
            lines.append(f"| **{provider}** | ${cost:.4f} | {in_t:,} | {out_t:,} | {tokens:,} |")
            
    lines.append("")
    lines.append("## 🤖 Usage by Model")
    lines.append("| Model | Cost (USD) | Input Tokens | Output Tokens | Total Tokens |")
    lines.append("| :--- | :--- | :--- | :--- | :--- |")
    
    models_data = db.get("models", {})
    if not models_data:
        lines.append("| No data tracked yet | - | - | - | - |")
    else:
        for model, data in sorted(models_data.items(), key=lambda x: x[1].get("cost", 0.0), reverse=True):
            cost = data.get("cost", 0.0)
            tokens = data.get("tokens", 0)
            in_t = data.get("input_tokens", 0)
            out_t = data.get("output_tokens", 0)
            lines.append(f"| `{model}` | ${cost:.4f} | {in_t:,} | {out_t:,} | {tokens:,} |")
            
    return "\n".join(lines)
