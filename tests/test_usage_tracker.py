import os
import sys
import unittest
from pathlib import Path
import json
from unittest.mock import patch

# Add repository root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import usage_tracker

class TestUsageTracker(unittest.TestCase):
    def setUp(self):
        # Ensure clean state for test database
        if os.path.exists(usage_tracker.USAGE_FILE_PATH):
            try:
                os.remove(usage_tracker.USAGE_FILE_PATH)
            except OSError:
                pass
        if os.path.exists(usage_tracker.LOCK_FILE_PATH):
            try:
                os.remove(usage_tracker.LOCK_FILE_PATH)
            except OSError:
                pass

    def tearDown(self):
        self.setUp()

    def test_get_prices(self):
        # 1. Paid models
        in_p, out_p = usage_tracker.get_prices("gpt-4o-mini", "gpt-bridge")
        self.assertAlmostEqual(in_p, 0.15 / 1000000.0)
        self.assertAlmostEqual(out_p, 0.60 / 1000000.0)

        in_p, out_p = usage_tracker.get_prices("openai/gpt-4o", "openrouter-bridge")
        self.assertAlmostEqual(in_p, 2.50 / 1000000.0)
        self.assertAlmostEqual(out_p, 10.00 / 1000000.0)

        # 2. Free models / providers
        in_p, out_p = usage_tracker.get_prices("nvidia/nemotron-3-super-120b-a12b:free", "openrouter-bridge")
        self.assertEqual(in_p, 0.0)
        self.assertEqual(out_p, 0.0)

        in_p, out_p = usage_tracker.get_prices("llama-3.3-70b-versatile", "groq-bridge")
        self.assertEqual(in_p, 0.0)
        self.assertEqual(out_p, 0.0)

        # Local Ollama
        in_p, out_p = usage_tracker.get_prices("gemma4:latest", "hf-bridge")
        self.assertEqual(in_p, 0.0)
        self.assertEqual(out_p, 0.0)

        # 3. Fallback price
        in_p, out_p = usage_tracker.get_prices("unknown-expensive-model", "openrouter-bridge")
        self.assertAlmostEqual(in_p, 2.50 / 1000000.0)
        self.assertAlmostEqual(out_p, 10.00 / 1000000.0)

    def test_record_usage_and_load(self):
        # Record usage first time
        usage_tracker.record_usage("gpt-bridge", "gpt-4o-mini", 1000, 500)
        
        # Load and verify
        db = usage_tracker.load_usage_db()
        
        # Expected cost = 1000 * 0.15/1e6 + 500 * 0.60/1e6 = 0.00015 + 0.00030 = 0.00045
        expected_cost = 0.00045
        
        # Check providers
        self.assertIn("gpt-bridge", db["providers"])
        self.assertAlmostEqual(db["providers"]["gpt-bridge"]["cost"], expected_cost)
        self.assertEqual(db["providers"]["gpt-bridge"]["tokens"], 1500)

        # Check models
        self.assertIn("gpt-4o-mini", db["models"])
        self.assertAlmostEqual(db["models"]["gpt-4o-mini"]["cost"], expected_cost)

        # Check daily / monthly
        import datetime
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        self.assertIn(today, db["daily"])
        self.assertAlmostEqual(db["daily"][today]["cost"], expected_cost)

    def test_budget_caps_enforced(self):
        # Configure small daily limit in DB
        db = usage_tracker.load_usage_db()
        db["config"]["daily_budget_cap"] = 0.001
        usage_tracker.save_usage_db(db)

        # Free models should not be blocked
        try:
            usage_tracker.check_budget("openrouter-bridge", "some-model:free")
        except ValueError:
            self.fail("Free model was unexpectedly blocked by budget cap")

        # Paid call below budget
        try:
            usage_tracker.check_budget("gpt-bridge", "gpt-4o-mini")
        except ValueError:
            self.fail("Paid model below budget cap was unexpectedly blocked")

        # Exceed budget by recording usage
        # 10,000 input tokens of gpt-4o = 10,000 * 2.5/1e6 = 0.025 (which is > 0.001 limit)
        usage_tracker.record_usage("gpt-bridge", "gpt-4o", 10000, 0)

        # Paid model should now be blocked
        with self.assertRaises(ValueError) as ctx:
            usage_tracker.check_budget("gpt-bridge", "gpt-4o-mini")
        self.assertIn("Daily budget cap exceeded", str(ctx.exception))

        # Free model should still NOT be blocked even when cap is exceeded
        try:
            usage_tracker.check_budget("openrouter-bridge", "some-model:free")
        except ValueError:
            self.fail("Free model was unexpectedly blocked after budget cap was exceeded")

    def test_get_bridge_costs_report(self):
        # Record some usage
        usage_tracker.record_usage("gpt-bridge", "gpt-4o-mini", 2000, 1000)
        
        report = usage_tracker.get_bridge_costs()
        self.assertIn("AI Bridges Usage & Cost Report", report)
        self.assertIn("gpt-bridge", report)
        self.assertIn("gpt-4o-mini", report)

        # Set budget limit and exceed it to test warning
        db = usage_tracker.load_usage_db()
        db["config"]["daily_budget_cap"] = 0.0001
        usage_tracker.save_usage_db(db)

        report = usage_tracker.get_bridge_costs()
        self.assertIn("⚠️ EXCEEDED", report)

    @patch.dict("os.environ", {
        "PROVIDER_DAILY_BUDGET_OPENROUTER_BRIDGE": "0.005",
        "PROVIDER_MONTHLY_BUDGET_OPENROUTER_BRIDGE": "0.10",
        "PROVIDER_DAILY_TOKEN_BUDGET_OPENROUTER_BRIDGE": "10000",
        "PROVIDER_SOFT_CAP_RATIO_OPENROUTER_BRIDGE": "0.5"
    })
    def test_provider_budgets(self):
        # 1. Check configs are read properly from env
        caps = usage_tracker.get_provider_budget_caps("openrouter-bridge")
        self.assertEqual(caps["daily_budget_cap"], 0.005)
        self.assertEqual(caps["monthly_budget_cap"], 0.10)
        self.assertEqual(caps["daily_token_budget_cap"], 10000)
        self.assertEqual(caps["soft_cap_ratio"], 0.5)

        # 2. Check no usage first
        usage = usage_tracker.get_provider_usage("openrouter-bridge")
        self.assertEqual(usage["daily_cost"], 0.0)
        self.assertEqual(usage["daily_tokens"], 0)

        # 3. record_usage and verify provider stats accumulate daily/monthly
        usage_tracker.record_usage("openrouter-bridge", "gpt-4o", 1000, 0)
        usage = usage_tracker.get_provider_usage("openrouter-bridge")
        expected_cost = 1000 * (2.50 / 1e6) # 0.0025
        self.assertAlmostEqual(usage["daily_cost"], expected_cost)
        self.assertEqual(usage["daily_tokens"], 1000)

        # 4. Check budget status - should be soft-capped (cost >= 0.005 * 0.5 = 0.0025)
        status = usage_tracker.check_provider_budget("openrouter-bridge")
        self.assertFalse(status["is_exceeded"])
        self.assertTrue(status["is_soft_capped"])

        # 5. Add more usage to exceed daily cost limit
        usage_tracker.record_usage("openrouter-bridge", "gpt-4o", 2000, 0) # total 3000 tokens, cost 0.0075 (> 0.005 cap)
        status = usage_tracker.check_provider_budget("openrouter-bridge")
        self.assertTrue(status["is_exceeded"])

        # 6. Verify report contains per-provider limits
        report = usage_tracker.get_bridge_costs()
        self.assertIn("Per-Provider Limits", report)
        self.assertIn("openrouter-bridge", report)
        self.assertIn("EXCEEDED", report)

    def test_provider_budget_caps_falls_back_to_db_on_malformed_env(self):
        db = usage_tracker.load_usage_db()
        db.setdefault("config", {}).setdefault("providers", {})["openrouter-bridge"] = {
            "daily_budget_cap": 0.0125,
            "monthly_budget_cap": 0.25,
            "daily_token_budget_cap": 25000,
            "monthly_token_budget_cap": 50000,
            "soft_cap_ratio": 0.6,
        }
        usage_tracker.save_usage_db(db)

        with patch.dict("os.environ", {
            "PROVIDER_DAILY_BUDGET_OPENROUTER_BRIDGE": "not-a-number",
            "PROVIDER_MONTHLY_BUDGET_OPENROUTER_BRIDGE": "still-not-a-number",
            "PROVIDER_DAILY_TOKEN_BUDGET_OPENROUTER_BRIDGE": "bad",
            "PROVIDER_MONTHLY_TOKEN_BUDGET_OPENROUTER_BRIDGE": "worse",
            "PROVIDER_SOFT_CAP_RATIO_OPENROUTER_BRIDGE": "nan?",
        }, clear=True):
            try:
                caps = usage_tracker.get_provider_budget_caps("openrouter-bridge")
            except Exception as exc:
                self.fail(f"Malformed provider budget env should fall back to DB config, not raise: {exc}")

        self.assertEqual(caps["daily_budget_cap"], 0.0125)
        self.assertEqual(caps["monthly_budget_cap"], 0.25)
        self.assertEqual(caps["daily_token_budget_cap"], 25000)
        self.assertEqual(caps["monthly_token_budget_cap"], 50000)
        self.assertEqual(caps["soft_cap_ratio"], 0.6)


class TestSimpleFileLock(unittest.TestCase):
    def setUp(self):
        self.lock_path = ROOT / ".test_simple_file_lock.lock"
        if self.lock_path.exists():
            try:
                self.lock_path.unlink()
            except OSError:
                pass

    def tearDown(self):
        if self.lock_path.exists():
            try:
                self.lock_path.unlink()
            except OSError:
                pass

    def test_acquire_and_release(self):
        lock = usage_tracker.SimpleFileLock(str(self.lock_path))
        self.assertFalse(lock.is_locked)
        with lock:
            self.assertTrue(lock.is_locked)
            self.assertTrue(self.lock_path.exists())
            with open(self.lock_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            self.assertEqual(content, lock.owner_id)
        self.assertFalse(self.lock_path.exists())

    def test_timeout_on_existing_lock(self):
        lock1 = usage_tracker.SimpleFileLock(str(self.lock_path))
        lock2 = usage_tracker.SimpleFileLock(str(self.lock_path), timeout=0.1)
        
        with lock1:
            with self.assertRaises(TimeoutError):
                with lock2:
                    pass

    def test_break_stale_lock(self):
        import time
        lock1 = usage_tracker.SimpleFileLock(str(self.lock_path))
        with lock1:
            # Manually make the lock file look stale by setting mtime to 20 seconds ago
            stale_time = time.time() - 20.0
            os.utime(self.lock_path, (stale_time, stale_time))
            
            # Now lock2 should be able to break it and acquire lock
            lock2 = usage_tracker.SimpleFileLock(str(self.lock_path), timeout=0.5)
            with lock2:
                self.assertTrue(lock2.is_locked)
                with open(self.lock_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                self.assertEqual(content, lock2.owner_id)
                self.assertNotEqual(content, lock1.owner_id)
            
            self.assertFalse(self.lock_path.exists())

    def test_release_stolen_lock_does_not_delete_new_lock(self):
        lock1 = usage_tracker.SimpleFileLock(str(self.lock_path))
        lock2 = usage_tracker.SimpleFileLock(str(self.lock_path))
        
        # Simulate lock1 acquiring lock
        lock1.__enter__()
        self.assertTrue(self.lock_path.exists())
        
        # Simulate lock1 being broken/stolen by lock2
        with open(self.lock_path, "w", encoding="utf-8") as f:
            f.write(lock2.owner_id)
            
        # When lock1 exits, it should NOT delete the lock file because its owner ID no longer matches
        lock1.__exit__(None, None, None)
        
        self.assertTrue(self.lock_path.exists())
        with open(self.lock_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        self.assertEqual(content, lock2.owner_id)

    def test_release_race_preserves_reacquired_lock(self):
        lock1 = usage_tracker.SimpleFileLock(str(self.lock_path))
        lock2 = usage_tracker.SimpleFileLock(str(self.lock_path))

        lock1.__enter__()
        self.assertTrue(self.lock_path.exists())

        original_rename = os.rename

        def raced_rename(src, dst):
            if src == str(self.lock_path) and dst == f"{self.lock_path}.release.{lock1.owner_id}":
                with open(self.lock_path, "w", encoding="utf-8") as f:
                    f.write(lock2.owner_id)
            return original_rename(src, dst)

        with patch.object(usage_tracker.os, "rename", side_effect=raced_rename):
            lock1.__exit__(None, None, None)

        self.assertTrue(self.lock_path.exists())
        with open(self.lock_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        self.assertEqual(content, lock2.owner_id)


if __name__ == "__main__":
    unittest.main()
