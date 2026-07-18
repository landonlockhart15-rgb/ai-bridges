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
        import uuid
        self.owner_id = uuid.uuid4().hex

    def __enter__(self):
        start_time = time.time()
        import uuid
        while time.time() - start_time < self.timeout:
            try:
                fd = os.open(self.lock_file_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                try:
                    os.write(fd, self.owner_id.encode("utf-8"))
                finally:
                    os.close(fd)
                self.is_locked = True
                return self
            except FileExistsError:
                # Clean up stale locks older than 10 seconds
                try:
                    mtime = os.path.getmtime(self.lock_file_path)
                    if time.time() - mtime > 10.0:
                        stale_path = self.lock_file_path + f".stale.{uuid.uuid4().hex}"
                        try:
                            os.rename(self.lock_file_path, stale_path)
                            try:
                                os.remove(stale_path)
                            except OSError:
                                pass
                        except OSError:
                            pass
                except OSError:
                    pass
                time.sleep(0.05)
        raise TimeoutError(f"Could not acquire lock on {self.lock_file_path}")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.is_locked:
            try:
                try:
                    with open(self.lock_file_path, "r", encoding="utf-8") as f:
                        owner = f.read().strip()
                except Exception:
                    owner = None
                
                if owner == self.owner_id:
                    release_path = self.lock_file_path + f".release.{self.owner_id}"
                    try:
                        os.rename(self.lock_file_path, release_path)
                        try:
                            with open(release_path, "r", encoding="utf-8") as f:
                                verified_owner = f.read().strip()
                        except Exception:
                            verified_owner = None
                        
                        if verified_owner == self.owner_id:
                            os.remove(release_path)
                        else:
                            try:
                                os.rename(release_path, self.lock_file_path)
                            except OSError:
                                pass
                    except OSError:
                        pass
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

def get_provider_env_var(provider, suffix):
    """Retrieve provider-specific environment variables in various formats."""
    normalized = provider.replace("-", "_").upper()
    val = os.environ.get(f"PROVIDER_{suffix}_{normalized}")
    if val is not None:
        return val
    val = os.environ.get(f"provider_{suffix.lower()}_{provider.replace('-', '_').lower()}")
    if val is not None:
        return val
    val = os.environ.get(f"PROVIDER_{suffix}_{provider.upper()}")
    if val is not None:
        return val
    val = os.environ.get(f"provider_{suffix.lower()}_{provider.lower()}")
    return val


def get_provider_budget_caps(provider):
    """Retrieve daily and monthly budget caps for a specific provider/bridge."""
    daily_cap_env = get_provider_env_var(provider, "DAILY_BUDGET")
    monthly_cap_env = get_provider_env_var(provider, "MONTHLY_BUDGET")
    daily_token_env = get_provider_env_var(provider, "DAILY_TOKEN_BUDGET")
    monthly_token_env = get_provider_env_var(provider, "MONTHLY_TOKEN_BUDGET")
    soft_cap_ratio_env = get_provider_env_var(provider, "SOFT_CAP_RATIO")
    if soft_cap_ratio_env is None:
        soft_cap_ratio_env = os.environ.get("PROVIDER_SOFT_CAP_RATIO")

    daily_cap = float(daily_cap_env) if daily_cap_env is not None and daily_cap_env != "" else None
    monthly_cap = float(monthly_cap_env) if monthly_cap_env is not None and monthly_cap_env != "" else None
    daily_token_cap = int(daily_token_env) if daily_token_env is not None and daily_token_env != "" else None
    monthly_token_cap = int(monthly_token_env) if monthly_token_env is not None and monthly_token_env != "" else None
    soft_cap_ratio = float(soft_cap_ratio_env) if soft_cap_ratio_env is not None and soft_cap_ratio_env != "" else 0.8

    db = load_usage_db()
    provider_config = db.get("config", {}).get("providers", {}).get(provider, {})

    if daily_cap is None:
        daily_cap = provider_config.get("daily_budget_cap")
    if monthly_cap is None:
        monthly_cap = provider_config.get("monthly_budget_cap")
    if daily_token_cap is None:
        daily_token_cap = provider_config.get("daily_token_budget_cap")
    if monthly_token_cap is None:
        monthly_token_cap = provider_config.get("monthly_token_budget_cap")
    if soft_cap_ratio is None or soft_cap_ratio == 0.8:
        db_soft_cap_ratio = provider_config.get("soft_cap_ratio")
        if db_soft_cap_ratio is not None:
            soft_cap_ratio = db_soft_cap_ratio

    return {
        "daily_budget_cap": daily_cap,
        "monthly_budget_cap": monthly_cap,
        "daily_token_budget_cap": daily_token_cap,
        "monthly_token_budget_cap": monthly_token_cap,
        "soft_cap_ratio": soft_cap_ratio
    }


def get_provider_usage(provider):
    """Get the current daily and monthly cost and token usage for a provider."""
    db = load_usage_db()
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    current_month = datetime.datetime.now().strftime("%Y-%m")

    daily_data = db.get("daily", {}).get(current_date, {}).get("providers", {}).get(provider, {})
    monthly_data = db.get("monthly", {}).get(current_month, {}).get("providers", {}).get(provider, {})

    return {
        "daily_cost": daily_data.get("cost", 0.0),
        "daily_tokens": daily_data.get("tokens", 0),
        "monthly_cost": monthly_data.get("cost", 0.0),
        "monthly_tokens": monthly_data.get("tokens", 0),
    }


def check_provider_budget(provider):
    """
    Check provider daily/monthly budget caps (both cost and tokens).
    Returns a dict with:
      - is_exceeded (bool): True if any hard cap is exceeded.
      - is_soft_capped (bool): True if any cap is approaching its soft cap.
      - message (str): Explanation if exceeded or soft capped.
    """
    caps = get_provider_budget_caps(provider)
    usage = get_provider_usage(provider)

    daily_cost = usage["daily_cost"]
    daily_tokens = usage["daily_tokens"]
    monthly_cost = usage["monthly_cost"]
    monthly_tokens = usage["monthly_tokens"]

    daily_cost_cap = caps["daily_budget_cap"]
    monthly_cost_cap = caps["monthly_budget_cap"]
    daily_token_cap = caps["daily_token_budget_cap"]
    monthly_token_cap = caps["monthly_token_budget_cap"]
    soft_cap_ratio = caps["soft_cap_ratio"]

    is_exceeded = False
    is_soft_capped = False
    msg_parts = []

    # Check Daily Cost
    if daily_cost_cap is not None:
        if daily_cost >= daily_cost_cap:
            is_exceeded = True
            msg_parts.append(f"daily cost limit exceeded (${daily_cost:.4f}/${daily_cost_cap:.4f})")
        elif daily_cost >= daily_cost_cap * soft_cap_ratio:
            is_soft_capped = True
            msg_parts.append(f"approaching daily cost limit (${daily_cost:.4f}/${daily_cost_cap:.4f})")

    # Check Monthly Cost
    if monthly_cost_cap is not None:
        if monthly_cost >= monthly_cost_cap:
            is_exceeded = True
            msg_parts.append(f"monthly cost limit exceeded (${monthly_cost:.4f}/${monthly_cost_cap:.4f})")
        elif monthly_cost >= monthly_cost_cap * soft_cap_ratio:
            is_soft_capped = True
            msg_parts.append(f"approaching monthly cost limit (${monthly_cost:.4f}/${monthly_cost_cap:.4f})")

    # Check Daily Tokens
    if daily_token_cap is not None:
        if daily_tokens >= daily_token_cap:
            is_exceeded = True
            msg_parts.append(f"daily token limit exceeded ({daily_tokens}/{daily_token_cap})")
        elif daily_tokens >= daily_token_cap * soft_cap_ratio:
            is_soft_capped = True
            msg_parts.append(f"approaching daily token limit ({daily_tokens}/{daily_token_cap})")

    # Check Monthly Tokens
    if monthly_token_cap is not None:
        if monthly_tokens >= monthly_token_cap:
            is_exceeded = True
            msg_parts.append(f"monthly token limit exceeded ({monthly_tokens}/{monthly_token_cap})")
        elif monthly_tokens >= monthly_token_cap * soft_cap_ratio:
            is_soft_capped = True
            msg_parts.append(f"approaching monthly token limit ({monthly_tokens}/{monthly_token_cap})")

    message = ", ".join(msg_parts) if msg_parts else "within budget caps"
    return {
        "is_exceeded": is_exceeded,
        "is_soft_capped": is_soft_capped,
        "message": message
    }


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
        
        day_providers = day_data.setdefault("providers", {})
        p_day_data = day_providers.setdefault(provider, {"cost": 0.0, "tokens": 0, "input_tokens": 0, "output_tokens": 0})
        p_day_data["cost"] += cost
        p_day_data["tokens"] += total_tokens
        p_day_data["input_tokens"] += prompt_tokens
        p_day_data["output_tokens"] += completion_tokens
        
        # 2. Update monthly
        monthly = db.setdefault("monthly", {})
        month_data = monthly.setdefault(current_month, {"cost": 0.0, "tokens": 0, "input_tokens": 0, "output_tokens": 0})
        month_data["cost"] += cost
        month_data["tokens"] += total_tokens
        month_data["input_tokens"] += prompt_tokens
        month_data["output_tokens"] += completion_tokens
        
        month_providers = month_data.setdefault("providers", {})
        p_month_data = month_providers.setdefault(provider, {"cost": 0.0, "tokens": 0, "input_tokens": 0, "output_tokens": 0})
        p_month_data["cost"] += cost
        p_month_data["tokens"] += total_tokens
        p_month_data["input_tokens"] += prompt_tokens
        p_month_data["output_tokens"] += completion_tokens
        
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

    # Provider budget limits reporting
    has_provider_limits = False
    provider_limit_lines = []
    known_providers = ["groq-bridge", "gemini-bridge", "cerebras-bridge", "openrouter-bridge", "hf-bridge", "gpt-bridge"]
    for p in known_providers:
        caps = get_provider_budget_caps(p)
        if any(caps.get(k) is not None for k in ["daily_budget_cap", "monthly_budget_cap", "daily_token_budget_cap", "monthly_token_budget_cap"]):
            has_provider_limits = True
            status = check_provider_budget(p)
            status_text = "OK"
            if status["is_exceeded"]:
                status_text = "⚠️ EXCEEDED"
            elif status["is_soft_capped"]:
                status_text = "⚠️ SOFT-CAP REACHED"
            
            limit_desc = []
            usage = get_provider_usage(p)
            if caps["daily_budget_cap"] is not None:
                limit_desc.append(f"Daily Cost: ${usage['daily_cost']:.4f}/${caps['daily_budget_cap']:.4f}")
            if caps["monthly_budget_cap"] is not None:
                limit_desc.append(f"Monthly Cost: ${usage['monthly_cost']:.4f}/${caps['monthly_budget_cap']:.4f}")
            if caps["daily_token_budget_cap"] is not None:
                limit_desc.append(f"Daily Tokens: {usage['daily_tokens']:,}/{caps['daily_token_budget_cap']:,}")
            if caps["monthly_token_budget_cap"] is not None:
                limit_desc.append(f"Monthly Tokens: {usage['monthly_tokens']:,}/{caps['monthly_token_budget_cap']:,}")
            
            provider_limit_lines.append(f"- **{p}:** {status_text} ({', '.join(limit_desc)})")

    if has_provider_limits:
        lines.append("### 🔌 Per-Provider Limits")
        lines.extend(provider_limit_lines)
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
