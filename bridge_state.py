import json
import os
import sys
import time


IS_TEST = "unittest" in sys.modules or "pytest" in sys.modules or os.environ.get("BRIDGE_TESTING") == "1"
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE_PATH = os.path.join(ROOT_DIR, ".bridge_state_test.json" if IS_TEST else ".bridge_state.json")
LOCK_FILE_PATH = STATE_FILE_PATH + ".lock"
STATUS_CACHE_FILE = os.path.join(ROOT_DIR, ".bridge_status_test.json" if IS_TEST else ".bridge_status.json")
DEFAULT_COOLDOWN_SECONDS = int(os.environ.get("BRIDGE_STATE_COOLDOWN_SECONDS", "300"))


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


def mark_unavailable(provider, reason, model=None, cooldown_seconds=None, is_429_or_5xx=False):
    cooldown_seconds = DEFAULT_COOLDOWN_SECONDS if cooldown_seconds is None else cooldown_seconds
    threshold = int(os.environ.get("BRIDGE_FAILURE_THRESHOLD", "1" if IS_TEST else "3"))
    now = _now()
    cooldown_until = now + cooldown_seconds

    with SimpleFileLock(LOCK_FILE_PATH):
        state = load_state()
        providers = state.setdefault("providers", {})
        provider_state = providers.setdefault(provider, {"status": "ok", "models": {}})
        
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
                "consecutive_failures": 0
            })
        else:
            model_state = provider_state.setdefault("models", {}).setdefault(model, {"status": "ok"})
            model_state.update({
                "status": "ok",
                "reason": None,
                "cooldown_until": 0,
                "consecutive_failures": 0
            })
        save_state(state)


def is_available(provider, model=None):
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

    return {
        "is_available": is_avail,
        "consecutive_failures": consecutive_failures,
        "success_rate": success_rate,
        "avg_latency": avg_latency,
        "provider_success_rate": provider_success_rate,
        "provider_avg_latency": provider_avg_latency,
        "provider_is_degraded": provider_is_degraded,
    }



