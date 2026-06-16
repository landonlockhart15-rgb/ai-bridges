import os
import json
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Import diagnostics functions
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import check_bridges

class TestDiagnostics(unittest.TestCase):
    def test_mask_key(self):
        # Test key masking formats
        self.assertEqual(check_bridges.mask_key(None), "Not Configured")
        self.assertEqual(check_bridges.mask_key(""), "Not Configured")
        self.assertEqual(check_bridges.mask_key("gsk_123456789"), "gsk_...6789") # Just checking parts
        self.assertEqual(check_bridges.mask_key("gsk_somekeyval"), "gsk_...yval")
        self.assertEqual(check_bridges.mask_key("sk-proj-someopenai1234key"), "sk-p...4key")
        self.assertEqual(check_bridges.mask_key("abc"), "ab...bc")
        
    def test_validate_key(self):
        # Groq key validation
        valid, msg = check_bridges.validate_key("groq-bridge", "gsk_12345678")
        self.assertFalse(valid)
        self.assertIn("at least 15 characters", msg)
        
        valid, msg = check_bridges.validate_key("groq-bridge", "gsk_12 4567890123")
        self.assertFalse(valid)
        self.assertIn("whitespace", msg)
        
        valid, msg = check_bridges.validate_key("groq-bridge", "notgsk_1234567890")
        self.assertFalse(valid)
        self.assertIn("must start with 'gsk_'", msg)
        
        valid, msg = check_bridges.validate_key("groq-bridge", "gsk_123456789012345")
        self.assertTrue(valid)
        
        # General key validation
        valid, msg = check_bridges.validate_key("gpt-bridge", "sk-proj-key123")
        self.assertTrue(valid)
        
        valid, msg = check_bridges.validate_key("gpt-bridge", "sk-proj key")
        self.assertFalse(valid)
        
        valid, msg = check_bridges.validate_key("gpt-bridge", "short")
        self.assertFalse(valid)
        
    @patch("check_bridges.Path.exists")
    @patch("builtins.open")
    def test_check_registration(self, mock_open, mock_exists):
        # Setup mock file reading for registration detection
        mock_exists.return_value = True
        
        mock_file = MagicMock()
        mock_file.read.return_value = json.dumps({
            "mcpServers": {
                "groq-bridge": {},
                "hf-bridge": {},
                "gem-bridge": {},
                "cerebras-bridge": {}
            }
        })
        mock_open.return_value.__enter__.return_value = mock_file
        
        registered = check_bridges.check_registration()
        self.assertTrue(registered["groq-bridge"])
        self.assertTrue(registered["hf-bridge"])
        self.assertTrue(registered["gemini-bridge"])
        self.assertTrue(registered["cerebras-bridge"])
        self.assertFalse(registered["gpt-bridge"])
        
    @patch("urllib.request.urlopen")
    def test_ping_api_success(self, mock_urlopen):
        # Mock successful urlopen
        mock_response = MagicMock()
        mock_response.status = 200
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        success, latency, msg = check_bridges.ping_api("http://test-url", key_present=True)
        self.assertTrue(success)
        self.assertIsNotNone(latency)
        self.assertEqual(msg, "Successfully connected.")
        
    @patch("urllib.request.urlopen")
    def test_ping_api_auth_failure(self, mock_urlopen):
        # Mock authentication failure (HTTP 401)
        from urllib.error import HTTPError
        fp = MagicMock()
        mock_urlopen.side_effect = HTTPError("http://test-url", 401, "Unauthorized", {}, fp)
        
        success, latency, msg = check_bridges.ping_api("http://test-url", key_present=True)
        self.assertFalse(success)
        self.assertIsNotNone(latency)
        self.assertIn("Authentication failed", msg)
        
    @patch("urllib.request.urlopen")
    def test_check_ollama_success(self, mock_urlopen):
        # Mock successful Ollama response
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "models": [
                {"name": "llama3:latest"},
                {"name": "gemma2:latest"}
            ]
        }).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        success, latency, msg = check_bridges.check_ollama()
        self.assertTrue(success)
        self.assertIn("Ollama is running", msg)
        self.assertIn("llama3:latest", msg)
        
    @patch("socket.socket")
    def test_check_kasa_no_devices(self, mock_socket):
        # Mock Kasa broadcast socket timing out (no devices found)
        import socket
        mock_sock_inst = MagicMock()
        mock_sock_inst.recvfrom.side_effect = socket.timeout
        mock_socket.return_value = mock_sock_inst
        
        success, latency, msg = check_bridges.check_kasa()
        self.assertTrue(success)
        self.assertIn("No Kasa devices responded", msg)

    @patch("check_bridges.check_registration")
    @patch("check_bridges.run_bridge_diagnostic")
    @patch("check_bridges.check_ollama")
    @patch("check_bridges.check_kasa")
    def test_check_bridges_report_generation(self, mock_kasa, mock_ollama, mock_run, mock_reg):
        # Mock components to check report generation formatting
        mock_reg.return_value = {
            "gpt-bridge": False,
            "groq-bridge": True,
            "gemini-bridge": False,
            "hf-bridge": True,
            "openrouter-bridge": True,
            "cerebras-bridge": True,
            "kasa-bridge": True
        }
        
        mock_run.side_effect = lambda b_id, reg: {
            "id": b_id,
            "status": "🟢 Online" if b_id != "gpt-bridge" else "🔴 Key Missing",
            "latency": "120 ms" if b_id != "gpt-bridge" else "—",
            "key_configured": "Yes" if b_id != "gpt-bridge" else "No",
            "key_masked": "sk-p...32ef" if b_id != "gpt-bridge" else "Not Configured",
            "registered": reg,
            "details": "Success" if b_id != "gpt-bridge" else "Missing env"
        }
        
        mock_ollama.return_value = (True, 5, "Ollama mock running")
        mock_kasa.return_value = (True, 2, "Kasa mock running")
        
        report = check_bridges.check_bridges()
        self.assertIn("# 🛠️ AI Bridges Unified Diagnostics", report)
        self.assertIn("groq-bridge", report)
        self.assertIn("cerebras-bridge", report)
        self.assertIn("🟢 Yes", report) # Registered groq
        self.assertIn("⚪ No", report) # Unregistered gpt
        self.assertIn("Ollama mock running", report)

if __name__ == "__main__":
    unittest.main()
