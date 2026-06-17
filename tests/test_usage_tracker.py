import os
import sys
import unittest
from pathlib import Path
import json

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

if __name__ == "__main__":
    unittest.main()
