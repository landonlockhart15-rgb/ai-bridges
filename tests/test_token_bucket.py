import importlib.util
import os
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bridge_state
import usage_tracker


class TestTokenBucket(unittest.TestCase):
    def setUp(self):
        for path in (
            bridge_state.STATE_FILE_PATH,
            bridge_state.LOCK_FILE_PATH,
            bridge_state.STATUS_CACHE_FILE,
        ):
            if Path(path).exists():
                try:
                    Path(path).unlink()
                except OSError:
                    pass

    def test_token_bucket_initialization(self):
        bucket = bridge_state.TokenBucket(rpm=30.0)
        self.assertEqual(bucket.rpm, 30.0)
        self.assertEqual(bucket.capacity, 30.0)
        self.assertEqual(bucket.tokens, 30.0)

    def test_try_consume_and_refill(self):
        now = 1000.0
        bucket = bridge_state.TokenBucket(rpm=60.0, capacity=10.0, tokens=10.0, last_refill=now)

        # Consume 10 tokens
        for _ in range(10):
            success, wait = bucket.try_consume(cost=1.0, now=now)
            self.assertTrue(success)
            self.assertEqual(wait, 0.0)

        # 11th token should fail
        success, wait = bucket.try_consume(cost=1.0, now=now)
        self.assertFalse(success)
        self.assertAlmostEqual(wait, 1.0, delta=0.01)

        # Advance time by 0.5s -> should refill 0.5 tokens (refill rate = 1 token/sec)
        now += 0.5
        bucket.refill(now=now)
        self.assertAlmostEqual(bucket.tokens, 0.5, delta=0.01)

        # Advance another 0.5s -> should have 1.0 token
        now += 0.5
        success, wait = bucket.try_consume(cost=1.0, now=now)
        self.assertTrue(success)
        self.assertEqual(wait, 0.0)

    def test_token_bucket_serialization(self):
        bucket = bridge_state.TokenBucket(rpm=30.0, capacity=15.0, tokens=5.0, last_refill=12345.0)
        data = bucket.to_dict()
        self.assertEqual(data["rpm"], 30.0)
        self.assertEqual(data["capacity"], 15.0)
        self.assertEqual(data["tokens"], 5.0)

        restored = bridge_state.TokenBucket.from_dict(data)
        self.assertEqual(restored.rpm, 30.0)
        self.assertEqual(restored.capacity, 15.0)
        self.assertEqual(restored.tokens, 5.0)

    def test_get_provider_rpm(self):
        self.assertEqual(bridge_state.get_provider_rpm("groq-bridge"), 30.0)
        self.assertEqual(bridge_state.get_provider_rpm("cerebras-bridge"), 30.0)
        self.assertEqual(bridge_state.get_provider_rpm("gemini-bridge"), 15.0)
        self.assertEqual(bridge_state.get_provider_rpm("unknown-provider"), 60.0)

        with patch.dict("os.environ", {"PROVIDER_RPM_GROQ_BRIDGE": "120.0"}):
            self.assertEqual(bridge_state.get_provider_rpm("groq-bridge"), 120.0)

    def test_consume_provider_token_persistence(self):
        provider = "groq-bridge"
        # Initial call consumes 1 token from groq-bridge (RPM=30, starting tokens=30)
        success, wait = bridge_state.consume_provider_token(provider)
        self.assertTrue(success)
        self.assertEqual(wait, 0.0)

        # Check state file
        state = bridge_state.load_state()
        pdata = state.get("providers", {}).get(provider, {})
        tb_data = pdata.get("token_bucket", {})
        self.assertAlmostEqual(tb_data.get("tokens", 0.0), 29.0, delta=0.1)

    def test_get_route_health_includes_token_bucket(self):
        health = bridge_state.get_route_health("groq-bridge", "llama-3.3-70b-versatile")
        self.assertIn("token_bucket_available", health)
        self.assertIn("token_bucket_tokens", health)
        self.assertIn("token_bucket_wait_seconds", health)
        self.assertTrue(health["token_bucket_available"])


if __name__ == "__main__":
    unittest.main()
