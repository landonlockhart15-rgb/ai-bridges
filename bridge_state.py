import json
import os
import sys
import time


IS_TEST = "unittest" in sys.modules or "pytest" in sys.modules or os.environ.get("BRIDGE_TESTING") == "1"
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE_PATH = os.path.join(ROOT_DIR, ".bridge_state_test.json" if IS_TEST else ".bridge_state.json")
LOCK_FILE_PATH = STATE_FILE_PATH + ".lock"
DEFAULT_COOLDOWN_SECONDS = int(os.environ.get("BRIDGE_STATE_COOLDOWN_SECONDS", "300"))


class ProviderUnavailableError(RuntimeError):
    pass


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


def mark_unavailable(provider, reason, model=None, cooldown_seconds=None):
    cooldown_seconds = DEFAULT_COOLDOWN_SECONDS if cooldown_seconds is None else cooldown_seconds
    now = _now()
    cooldown_until = now + cooldown_seconds
    update = {
        "status": "cooldown",
        "reason": reason,
        "last_error_at": _iso_timestamp(now),
        "cooldown_until": cooldown_until,
    }
    if model is not None:
        update["model"] = model

    with SimpleFileLock(LOCK_FILE_PATH):
        state = load_state()
        providers = state.setdefault("providers", {})
        provider_state = providers.setdefault(provider, {"status": "ok", "models": {}})
        if model is None:
            provider_state.update(update)
        else:
            provider_state.setdefault("models", {})[model] = update
        save_state(state)


def mark_available(provider, model=None):
    with SimpleFileLock(LOCK_FILE_PATH):
        state = load_state()
        provider_state = state.setdefault("providers", {}).setdefault(provider, {"status": "ok", "models": {}})
        if model is None:
            provider_state.update({"status": "ok", "reason": None, "cooldown_until": 0})
        else:
            provider_state.setdefault("models", {})[model] = {"status": "ok", "reason": None, "cooldown_until": 0}
        save_state(state)


def is_available(provider, model=None):
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


def record_metric(provider, model, latency, success):
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
            latency_history.pop(0)
        if len(success_history) > 10:
            success_history.pop(0)
            
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


