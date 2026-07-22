"""
Bridge State Management and Circuit Breaker Controls.

This module tracks the health, latency, rate limits, and error rates of various LLM providers and models.
It implements two types of circuit breakers:

1. Error-based Circuit Breaker:
   - Triggered by consecutive failures (such as 429 rate limits or 5xx server errors).
   - Configured via the `BRIDGE_FAILURE_THRESHOLD` environment variable (default: 3, or 1 in tests).
   - Once tripped, the provider/model status is set to "open" and it cools off for a specified duration.

2. Latency-based Circuit Breaker:
   - Triggered by sustained slow calls.
   - When a successful request exceeds a specific latency threshold, it counts as a slow call.
   - Configured via:
     * `SMART_ROUTER_LATENCY_THRESHOLD` (env var) or `DEFAULT_LATENCY_THRESHOLD_SECONDS` (default: 5.0s):
       The maximum acceptable latency for a request to not be considered "slow".
     * `BRIDGE_SLOW_CALL_THRESHOLD` (env var) or `DEFAULT_SLOW_CALL_THRESHOLD` (default: 3):
       The number of consecutive slow calls allowed before the circuit breaker trips.
   - Once tripped (consecutive slow calls reaches the threshold):
     * The provider status is set to "open".
     * The reason is recorded as "high_latency: <count> consecutive calls exceeded <threshold>s".
     * The provider cools off for `DEFAULT_COOLDOWN_SECONDS` (default: 300s).
   - Recovery:
     * The circuit breaker recovers when a successful request is recorded with a latency within the
       threshold (`latency <= latency_threshold`). This resets `consecutive_slow_calls` to 0 and,
       if the provider status was "open" due to a "high_latency" reason, restores status to "ok".
     * Note: A failed request (`success=False`) also resets `consecutive_slow_calls` to 0, which
       prevents blending of slow-call counts across error bounds, but errors are handled separately
       by the error-based circuit breaker.

Distinction between Circuit Breakers and Degradation:
- **Circuit Breaker ("open" status)**: Trips when `consecutive_slow_calls >= slow_call_threshold`. Completely disables the provider/model (`is_available()` returns `False`). Recovers on the first successful fast call.
- **Degradation ("provider_is_degraded" flag)**: Triggered when average provider latency over a sliding window exceeds threshold, or rate limits occur frequently. Used for routing priority (smart router prefers non-degraded bridges), but does not disable the provider. Recovers gradually as old slow/failed samples roll off the 10-sample sliding window.
"""

import json
import os
import re
import sys
import time


IS_TEST = "unittest" in sys.modules or "pytest" in sys.modules or os.environ.get("BRIDGE_TESTING") == "1"
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE_PATH = os.path.join(ROOT_DIR, ".bridge_state_test.json" if IS_TEST else ".bridge_state.json")
LOCK_FILE_PATH = STATE_FILE_PATH + ".lock"
STATUS_CACHE_FILE = os.path.join(ROOT_DIR, ".bridge_status_test.json" if IS_TEST else ".bridge_status.json")
DEFAULT_COOLDOWN_SECONDS = int(os.environ.get("BRIDGE_STATE_COOLDOWN_SECONDS", "300"))
DEFAULT_LATENCY_THRESHOLD_SECONDS = 5.0
DEFAULT_SLOW_CALL_THRESHOLD = 3
DEFAULT_PROVIDER_RPM = {
    "groq-bridge": 30.0,
    "cerebras-bridge": 30.0,
    "openrouter-bridge": 20.0,
    "gemini-bridge": 15.0,
    "hf-bridge": 60.0,
    "gpt-bridge": 600.0,
}
DEFAULT_FALLBACK_RPM = 60.0



class ProviderUnavailableError(RuntimeError):
    pass


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


def _empty_state():
    return {"version": 1, "providers": {}}


def _now():
    return time.time()


def _iso_timestamp(epoch_seconds):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch_seconds))


class TokenBucket:
    """
    Token Bucket rate limiter for managing concurrent requests per provider.

    Attributes:
        rpm (float): Requests per minute limit.
        capacity (float): Maximum token capacity (burst size). Defaults to rpm.
        tokens (float): Currently available tokens in the bucket.
        last_refill (float): Timestamp of last token replenishment.
    """

    def __init__(self, rpm: float = 60.0, capacity: float | None = None, tokens: float | None = None, last_refill: float | None = None):
        self.rpm = max(1.0, float(rpm))
        self.capacity = max(1.0, float(capacity)) if capacity is not None else self.rpm
        self.last_refill = float(last_refill) if last_refill is not None else _now()
        self.tokens = min(self.capacity, max(0.0, float(tokens))) if tokens is not None else self.capacity

    def refill(self, now: float | None = None) -> float:
        current_time = _now() if now is None else now
        elapsed = max(0.0, current_time - self.last_refill)
        refill_rate = self.rpm / 60.0
        added = elapsed * refill_rate
        self.tokens = min(self.capacity, self.tokens + added)
        self.last_refill = current_time
        return self.tokens

    def try_consume(self, cost: float = 1.0, now: float | None = None) -> tuple[bool, float]:
        """
        Attempt to consume `cost` tokens from the bucket.

        Returns:
            (success, wait_seconds): If success is True, wait_seconds is 0.0.
            If success is False, wait_seconds is the estimated delay until sufficient tokens refill.
        """
        self.refill(now=now)
        if self.tokens >= cost:
            self.tokens -= cost
            return True, 0.0
        needed = cost - self.tokens
        refill_rate = self.rpm / 60.0
        wait_seconds = needed / refill_rate if refill_rate > 0 else 60.0
        return False, wait_seconds

    def get_status(self, now: float | None = None) -> dict:
        self.refill(now=now)
        refill_rate = self.rpm / 60.0
        wait_seconds = 0.0
        if self.tokens < 1.0:
            wait_seconds = (1.0 - self.tokens) / refill_rate if refill_rate > 0 else 60.0
        return {
            "rpm": self.rpm,
            "capacity": self.capacity,
            "tokens": round(self.tokens, 2),
            "is_depleted": self.tokens < 1.0,
            "wait_seconds": round(wait_seconds, 2),
        }

    def to_dict(self) -> dict:
        return {
            "rpm": self.rpm,
            "capacity": self.capacity,
            "tokens": self.tokens,
            "last_refill": self.last_refill,
        }

    @classmethod
    def from_dict(cls, d: dict, default_rpm: float = 60.0) -> "TokenBucket":
        if not isinstance(d, dict):
            return cls(rpm=default_rpm)
        return cls(
            rpm=d.get("rpm", default_rpm),
            capacity=d.get("capacity"),
            tokens=d.get("tokens"),
            last_refill=d.get("last_refill"),
        )


def get_provider_rpm(provider: str) -> float:
    """Retrieve configured RPM for a provider from environment or default table."""
    try:
        import usage_tracker
        env_val = usage_tracker.get_provider_env_var(provider, "RPM")
        if env_val is not None:
            return float(env_val)
    except Exception:
        pass

    env_direct = os.environ.get(f"PROVIDER_RPM_{provider.replace('-', '_').upper()}")
    if env_direct is not None:
        try:
            return float(env_direct)
        except ValueError:
            pass

    return DEFAULT_PROVIDER_RPM.get(provider, DEFAULT_FALLBACK_RPM)


def get_token_bucket(provider: str, state: dict | None = None) -> TokenBucket:
    """Get the current TokenBucket instance for a provider."""
    rpm = get_provider_rpm(provider)
    if state is None:
        state = load_state()
    pdata = state.get("providers", {}).get(provider, {})
    tb_data = pdata.get("token_bucket")
    if tb_data:
        bucket = TokenBucket.from_dict(tb_data, default_rpm=rpm)
    else:
        bucket = TokenBucket(rpm=rpm)
    bucket.refill()
    return bucket


def consume_provider_token(provider: str, cost: float = 1.0) -> tuple[bool, float]:
    """
    Attempt to consume a token from the provider's token bucket.
    Persists updated bucket state into the shared state file.
    Returns (success, wait_seconds).
    """
    rpm = get_provider_rpm(provider)
    with SimpleFileLock(LOCK_FILE_PATH):
        state = load_state()
        providers = state.setdefault("providers", {})
        pdata = providers.setdefault(provider, {"status": "ok", "models": {}})
        tb_data = pdata.get("token_bucket")
        if tb_data:
            bucket = TokenBucket.from_dict(tb_data, default_rpm=rpm)
        else:
            bucket = TokenBucket(rpm=rpm)

        success, wait_seconds = bucket.try_consume(cost=cost)
        pdata["token_bucket"] = bucket.to_dict()
        save_state(state)
        return success, wait_seconds


def is_provider_throttled(provider: str) -> bool:
    """Return True if the provider's token bucket has fewer than 1 token available."""
    bucket = get_token_bucket(provider)
    return bucket.tokens < 1.0



def parse_rate_limit_reset(value, now=None):
    """Return seconds until a provider's rate-limit reset, or ``None``.

    Providers use all of relative durations (``2m59.56s``), Unix seconds, and
    Unix milliseconds for these headers.  Keep this parser here so every bridge
    writes the same cooldown representation to the shared state file.
    """
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    now = _now() if now is None else now

    try:
        number = float(text)
    except (TypeError, ValueError):
        number = None
    if number is not None:
        # Epoch milliseconds (as returned by OpenRouter) are otherwise treated
        # as a multi-millennia relative cooldown.
        if number >= 100_000_000_000:
            return max(0.0, number / 1000.0 - now)
        # Contemporary Unix seconds are unambiguously timestamps; smaller
        # values are relative durations.
        if number >= 1_000_000_000:
            return max(0.0, number - now)
        return max(0.0, number)

    total = 0.0
    matched = False
    for amount, unit in re.findall(r"(\d+(?:\.\d+)?)\s*(d|h|m|s)", text):
        matched = True
        total += float(amount) * {"d": 86400, "h": 3600, "m": 60, "s": 1}[unit]
    return total if matched else None


def record_rate_limit_headers(provider, model, headers):
    """Record an explicit exhausted quota from OpenAI-compatible response headers.

    Returns ``True`` only when a zero remaining quota and a future reset are
    both present.  Header names intentionally accept the standard, Groq /
    SambaNova, and Cerebras resource-qualified variants.
    """
    if not headers:
        return False
    try:
        values = {str(key).lower().replace("_", "-"): value for key, value in headers.items()}
    except AttributeError:
        return False

    resets = []
    for name, value in values.items():
        if "ratelimit-remaining" not in name:
            continue
        try:
            if float(value) <= 0:
                reset_name = name.replace("ratelimit-remaining", "ratelimit-reset", 1)
                reset_in = parse_rate_limit_reset(values.get(reset_name))
                if reset_in is not None:
                    resets.append(reset_in)
        except (TypeError, ValueError):
            continue
    if not resets:
        return False

    # More than one resource can be empty (e.g. requests and tokens).  Do not
    # retry until the last exhausted resource has reset.
    cooldown_seconds = max(resets)
    if cooldown_seconds <= 0:
        return False
    # A reported zero quota is definitive, so it must not wait for the normal
    # consecutive-failure threshold before opening the route.
    mark_unavailable(
        provider,
        "rate_limit_headers",
        model=model,
        cooldown_seconds=cooldown_seconds,
        is_429_or_5xx=False,
    )
    return True


def load_state():
    if not os.path.exists(STATE_FILE_PATH):
        return _empty_state()
    try:
        with open(STATE_FILE_PATH, "r", encoding="utf-8") as f:
            content = f.read().strip()
            return json.loads(content) if content else _empty_state()
    except Exception:
        return _empty_state()


def load_status_cache():
    if not os.path.exists(STATUS_CACHE_FILE):
        return None
    try:
        with open(STATUS_CACHE_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
            return json.loads(content) if content else None
    except Exception:
        return None


def write_status_cache(cache_data):
    try:
        temp_path = STATUS_CACHE_FILE + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2, sort_keys=True)
        if os.path.exists(STATUS_CACHE_FILE):
            os.remove(STATUS_CACHE_FILE)
        os.rename(temp_path, STATUS_CACHE_FILE)
    except Exception:
        try:
            with open(STATUS_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2, sort_keys=True)
        except Exception:
            pass


def save_state(state):
    try:
        temp_path = STATE_FILE_PATH + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, sort_keys=True)
        if os.path.exists(STATE_FILE_PATH):
            os.remove(STATE_FILE_PATH)
        os.rename(temp_path, STATE_FILE_PATH)
    except Exception:
        try:
            with open(STATE_FILE_PATH, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, sort_keys=True)
        except Exception:
            pass


def _entry_is_available(entry, now=None):
    if not entry:
        return True
    now = _now() if now is None else now
    return float(entry.get("cooldown_until", 0) or 0) <= now


def mark_unavailable(provider, reason, model=None, cooldown_seconds=None, is_429_or_5xx=False,
                     failure_class=None, failure_category=None):
    """Record a provider failure.

    ``failure_class='fatal'`` records an actionable configuration diagnostic
    without opening a circuit.  Callers that do not classify a failure retain
    the legacy behavior for compatibility with the individual bridges.
    """
    cooldown_seconds = DEFAULT_COOLDOWN_SECONDS if cooldown_seconds is None else cooldown_seconds
    threshold = int(os.environ.get("BRIDGE_FAILURE_THRESHOLD", "1" if IS_TEST else "3"))
    now = _now()
    cooldown_until = now + cooldown_seconds

    with SimpleFileLock(LOCK_FILE_PATH):
        state = load_state()
        providers = state.setdefault("providers", {})
        provider_state = providers.setdefault(provider, {"status": "ok", "models": {}})

        if failure_class == "fatal":
            entry = provider_state if model is None else provider_state.setdefault("models", {}).setdefault(model, {"status": "ok"})
            entry.update({
                "status": "fatal",
                "reason": reason,
                "failure_class": "fatal",
                "failure_category": failure_category or "configuration",
                "last_error_at": _iso_timestamp(now),
                "cooldown_until": 0,
            })
            save_state(state)
            return
        
        if model is None:
            failures = provider_state.get("consecutive_failures", 0)
            if is_429_or_5xx:
                failures += 1
                provider_state["consecutive_failures"] = failures
                if failures >= threshold:
                    provider_state.update({
                        "status": "open",
                        "reason": reason,
                        "last_error_at": _iso_timestamp(now),
                        "cooldown_until": cooldown_until,
                        "failure_class": "transient",
                    })
                else:
                    provider_state.update({
                        "last_error_at": _iso_timestamp(now),
                        "reason": f"consecutive_failures_count: {failures} | {reason}"
                    })
            else:
                provider_state["consecutive_failures"] = failures + 1
                provider_state.update({
                    "status": "open",
                    "reason": reason,
                    "last_error_at": _iso_timestamp(now),
                    "cooldown_until": cooldown_until,
                })
        else:
            model_state = provider_state.setdefault("models", {}).setdefault(model, {"status": "ok"})
            failures = model_state.get("consecutive_failures", 0)
            if is_429_or_5xx:
                failures += 1
                model_state["consecutive_failures"] = failures
                if failures >= threshold:
                    model_state.update({
                        "status": "open",
                        "reason": reason,
                        "last_error_at": _iso_timestamp(now),
                        "cooldown_until": cooldown_until,
                        "failure_class": "transient",
                    })
                else:
                    model_state.update({
                        "last_error_at": _iso_timestamp(now),
                        "reason": f"consecutive_failures_count: {failures} | {reason}"
                    })
            else:
                model_state["consecutive_failures"] = failures + 1
                model_state.update({
                    "status": "open",
                    "reason": reason,
                    "last_error_at": _iso_timestamp(now),
                    "cooldown_until": cooldown_until,
                })
        save_state(state)


def mark_available(provider, model=None):
    with SimpleFileLock(LOCK_FILE_PATH):
        state = load_state()
        provider_state = state.setdefault("providers", {}).setdefault(provider, {"status": "ok", "models": {}})
        if model is None:
            provider_state.update({
                "status": "ok",
                "reason": None,
                "cooldown_until": 0,
                "consecutive_failures": 0,
                "failure_class": None,
                "failure_category": None,
            })
        else:
            model_state = provider_state.setdefault("models", {}).setdefault(model, {"status": "ok"})
            model_state.update({
                "status": "ok",
                "reason": None,
                "cooldown_until": 0,
                "consecutive_failures": 0,
                "failure_class": None,
                "failure_category": None,
            })
        save_state(state)


def is_available(provider, model=None):
    try:
        import usage_tracker
        if usage_tracker.check_provider_budget(provider)["is_exceeded"]:
            return False
    except Exception:
        pass

    cache = load_status_cache()
    if cache and "bridges" in cache:
        bridge_info = cache["bridges"].get(provider)
        if bridge_info:
            status = bridge_info.get("status", "")
            if status.startswith("🔴"):
                return False

    state = load_state()
    provider_state = state.get("providers", {}).get(provider, {})
    if not _entry_is_available(provider_state):
        return False
    if model is None:
        return True
    return _entry_is_available(provider_state.get("models", {}).get(model, {}))


def filter_available_models(provider, models):
    return [model for model in models if is_available(provider, model)]


def raise_if_unavailable(provider, model=None):
    if is_available(provider, model):
        return
    state = load_state()
    provider_state = state.get("providers", {}).get(provider, {})
    entry = provider_state
    if model is not None:
        entry = provider_state.get("models", {}).get(model, provider_state)
    until = _iso_timestamp(float(entry.get("cooldown_until", 0) or 0))
    target = f"{provider}/{model}" if model is not None else provider
    raise ProviderUnavailableError(f"{target} is cooling off until {until}: {entry.get('reason')}")


def record_metric(provider, model, latency, success, is_rate_limit=False):
    """
    Record route health and latency metrics, updating circuit breaker states.
    
    This function tracks recent latency and success metrics for both model and provider levels,
    maintaining a sliding window of the last 10 samples. It also manages the latency-based
    circuit breaker:
    
    - If `success` is True and `latency` exceeds the configured `latency_threshold`, the
      `consecutive_slow_calls` count for the provider is incremented.
    - If `consecutive_slow_calls` reaches the `slow_call_threshold`, the provider's circuit is
      tripped ("open"), forcing a cooldown period.
    - If `success` is True and `latency` is within the threshold, `consecutive_slow_calls` is
      reset to 0. If the circuit was previously opened due to high latency, it is automatically
      recovered back to "ok".
    - If `success` is False, `consecutive_slow_calls` is reset to 0 (errors are tracked separately
      by the error-based circuit breaker).
    
    Parameters:
        provider (str): The identifier of the LLM provider (e.g., 'groq-bridge').
        model (str): The model name (e.g., 'llama-3.3-70b-versatile').
        latency (float): The request latency in seconds.
        success (bool): Whether the request succeeded.
        is_rate_limit (bool): Whether the failure was specifically a 429 rate limit.
    """
    with SimpleFileLock(LOCK_FILE_PATH):
        state = load_state()
        providers = state.setdefault("providers", {})
        provider_state = providers.setdefault(provider, {"status": "ok", "models": {}})
        model_state = provider_state.setdefault("models", {}).setdefault(model, {"status": "ok"})
        
        latency_history = model_state.setdefault("latency_history", [])
        success_history = model_state.setdefault("success_history", [])
        
        latency_history.append(latency)
        success_history.append(1 if success else 0)
        
        if len(latency_history) > 10:
            latency_history[:] = latency_history[-10:]
        if len(success_history) > 10:
            success_history[:] = success_history[-10:]

        # Track at provider level
        p_latency_history = provider_state.setdefault("latency_history", [])
        p_success_history = provider_state.setdefault("success_history", [])
        p_rate_limit_history = provider_state.setdefault("rate_limit_history", [])
        
        p_latency_history.append(latency)
        p_success_history.append(1 if success else 0)
        p_rate_limit_history.append(1 if is_rate_limit else 0)
        
        if len(p_latency_history) > 10:
            p_latency_history[:] = p_latency_history[-10:]
        if len(p_success_history) > 10:
            p_success_history[:] = p_success_history[-10:]
        if len(p_rate_limit_history) > 10:
            p_rate_limit_history[:] = p_rate_limit_history[-10:]

        latency_threshold = float(os.environ.get(
            "SMART_ROUTER_LATENCY_THRESHOLD",
            str(DEFAULT_LATENCY_THRESHOLD_SECONDS),
        ))
        slow_call_threshold = int(os.environ.get(
            "BRIDGE_SLOW_CALL_THRESHOLD",
            str(DEFAULT_SLOW_CALL_THRESHOLD),
        ))
        if success and latency > latency_threshold:
            consecutive_slow_calls = provider_state.get("consecutive_slow_calls", 0) + 1
            provider_state["consecutive_slow_calls"] = consecutive_slow_calls
            if consecutive_slow_calls >= slow_call_threshold:
                now = _now()
                provider_state.update({
                    "status": "open",
                    "reason": (
                        f"high_latency: {consecutive_slow_calls} consecutive calls "
                        f"exceeded {latency_threshold:.3f}s"
                    ),
                    "last_error_at": _iso_timestamp(now),
                    "cooldown_until": now + DEFAULT_COOLDOWN_SECONDS,
                })
        else:
            provider_state["consecutive_slow_calls"] = 0
            if success and str(provider_state.get("reason", "")).startswith("high_latency:"):
                provider_state.update({
                    "status": "ok",
                    "reason": None,
                    "cooldown_until": 0,
                })
            
        save_state(state)


def get_metrics(provider, model):
    state = load_state()
    provider_state = state.get("providers", {}).get(provider, {})
    model_state = provider_state.get("models", {}).get(model, {})
    
    latency_history = model_state.get("latency_history", [])
    success_history = model_state.get("success_history", [])
    
    avg_latency = sum(latency_history) / len(latency_history) if latency_history else 9999.0
    success_rate = sum(success_history) / len(success_history) if success_history else 1.0
    
    return {
        "avg_latency": avg_latency,
        "success_rate": success_rate,
        "latency_history": latency_history,
        "success_history": success_history,
    }


def get_provider_metrics(provider):
    state = load_state()
    provider_state = state.get("providers", {}).get(provider, {})
    
    latency_history = provider_state.get("latency_history", [])
    success_history = provider_state.get("success_history", [])
    rate_limit_history = provider_state.get("rate_limit_history", [])
    
    avg_latency = sum(latency_history) / len(latency_history) if latency_history else 9999.0
    success_rate = sum(success_history) / len(success_history) if success_history else 1.0
    
    return {
        "avg_latency": avg_latency,
        "success_rate": success_rate,
        "latency_history": latency_history,
        "success_history": success_history,
        "rate_limit_history": rate_limit_history,
    }


def get_route_health(provider, model):
    """
    Get real-time health metrics for a route (provider + model).
    Returns a dict with:
      - is_available (bool)
      - consecutive_failures (int)
      - success_rate (float)
      - avg_latency (float)
      - provider_success_rate (float)
      - provider_avg_latency (float)
      - provider_is_degraded (bool)
      - consecutive_slow_calls (int)
      - is_soft_capped (bool)
    """
    state = load_state()
    provider_state = state.get("providers", {}).get(provider, {})
    model_state = provider_state.get("models", {}).get(model, {})

    # Check availability
    cache = load_status_cache()
    cache_offline = False
    if cache and "bridges" in cache:
        bridge_info = cache["bridges"].get(provider)
        if bridge_info and bridge_info.get("status", "").startswith("🔴"):
            cache_offline = True

    provider_avail = _entry_is_available(provider_state) and not cache_offline
    model_avail = _entry_is_available(model_state) and not cache_offline
    is_avail = provider_avail and model_avail

    # Check provider budget caps
    is_soft_capped = False
    try:
        import usage_tracker
        budget_status = usage_tracker.check_provider_budget(provider)
        if budget_status["is_exceeded"]:
            is_avail = False
        is_soft_capped = budget_status["is_soft_capped"]
    except Exception:
        pass

    # Get consecutive failures
    provider_failures = provider_state.get("consecutive_failures", 0)
    model_failures = model_state.get("consecutive_failures", 0)
    consecutive_failures = max(provider_failures, model_failures)

    # Get success rate and latency
    latency_history = model_state.get("latency_history", [])
    success_history = model_state.get("success_history", [])
    avg_latency = sum(latency_history) / len(latency_history) if latency_history else 9999.0
    success_rate = sum(success_history) / len(success_history) if success_history else 1.0

    # Provider level metrics
    p_latency_history = provider_state.get("latency_history", [])
    p_success_history = provider_state.get("success_history", [])
    p_rate_limit_history = provider_state.get("rate_limit_history", [])
    
    provider_avg_latency = sum(p_latency_history) / len(p_latency_history) if p_latency_history else 9999.0
    provider_success_rate = sum(p_success_history) / len(p_success_history) if p_success_history else 1.0
    provider_rate_limit_count = sum(p_rate_limit_history) if p_rate_limit_history else 0
    
    latency_threshold = float(os.environ.get("SMART_ROUTER_LATENCY_THRESHOLD", "5.0"))
    rate_limit_threshold = int(os.environ.get("SMART_ROUTER_RATE_LIMIT_THRESHOLD", "2"))
    
    provider_is_high_latency = len(p_latency_history) >= 3 and provider_avg_latency > latency_threshold
    provider_is_frequent_429s = provider_rate_limit_count >= rate_limit_threshold
    provider_is_degraded = provider_is_high_latency or provider_is_frequent_429s

    # Token Bucket rate limit status
    bucket = get_token_bucket(provider, state=state)
    tb_status = bucket.get_status()

    return {
        "is_available": is_avail,
        "consecutive_failures": consecutive_failures,
        "success_rate": success_rate,
        "avg_latency": avg_latency,
        "provider_success_rate": provider_success_rate,
        "provider_avg_latency": provider_avg_latency,
        "provider_is_degraded": provider_is_degraded,
        "consecutive_slow_calls": provider_state.get("consecutive_slow_calls", 0),
        "is_soft_capped": is_soft_capped,
        "token_bucket_available": not tb_status["is_depleted"],
        "token_bucket_tokens": tb_status["tokens"],
        "token_bucket_wait_seconds": tb_status["wait_seconds"],
    }



def get_latency_heatmap():
    """
    Generate a predictive latency and reliability heatmap for all recorded providers and models.

    Returns a dictionary mapping provider -> model -> heatmap status dict:
      - avg_latency (float)
      - success_rate (float)
      - latency_tier ("fast" | "moderate" | "slow" | "high_latency")
      - sample_count (int)
      - status ("ok" | "open")
    """
    state = load_state()
    heatmap = {}
    providers = state.get("providers", {})
    for provider, pdata in providers.items():
        p_heatmap = {}
        p_latency_history = pdata.get("latency_history", [])
        p_success_history = pdata.get("success_history", [])
        p_avg_latency = sum(p_latency_history) / len(p_latency_history) if p_latency_history else 9999.0
        p_success_rate = sum(p_success_history) / len(p_success_history) if p_success_history else 1.0

        models = pdata.get("models", {})
        for model, mdata in models.items():
            l_hist = mdata.get("latency_history", [])
            s_hist = mdata.get("success_history", [])
            avg_lat = sum(l_hist) / len(l_hist) if l_hist else p_avg_latency
            succ_rate = sum(s_hist) / len(s_hist) if s_hist else p_success_rate

            if avg_lat < 1.0:
                tier = "fast"
            elif avg_lat <= 3.0:
                tier = "moderate"
            elif avg_lat <= 5.0:
                tier = "slow"
            else:
                tier = "high_latency"

            p_heatmap[model] = {
                "avg_latency": round(avg_lat, 4),
                "success_rate": round(succ_rate, 4),
                "latency_tier": tier,
                "sample_count": len(l_hist),
                "status": mdata.get("status", "ok"),
            }

        if not p_heatmap:
            if p_avg_latency < 1.0:
                p_tier = "fast"
            elif p_avg_latency <= 3.0:
                p_tier = "moderate"
            elif p_avg_latency <= 5.0:
                p_tier = "slow"
            else:
                p_tier = "high_latency"
            p_heatmap["_provider_overall"] = {
                "avg_latency": round(p_avg_latency, 4),
                "success_rate": round(p_success_rate, 4),
                "latency_tier": p_tier,
                "sample_count": len(p_latency_history),
                "status": pdata.get("status", "ok"),
            }

        heatmap[provider] = p_heatmap
    return heatmap




