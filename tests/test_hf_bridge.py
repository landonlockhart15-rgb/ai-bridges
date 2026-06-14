import sys
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch
import unittest

# Define MockFastMCP before importing server
class MockFastMCP:
    def __init__(self, name):
        self.name = name
        self.tools = {}

    def tool(self, *args, **kwargs):
        def decorator(func):
            self.tools[func.__name__] = func
            return func
        return decorator

    def run(self, *args, **kwargs):
        pass

# Setup mock modules
mock_fastmcp_mod = MagicMock()
mock_fastmcp_mod.FastMCP = MockFastMCP
sys.modules['mcp'] = MagicMock()
sys.modules['mcp.server'] = MagicMock()
sys.modules['mcp.server.fastmcp'] = mock_fastmcp_mod

# Mock OpenAI client
class MockOpenAI:
    def __init__(self, api_key=None, base_url=None):
        self.api_key = api_key
        self.base_url = base_url
        self.chat = MagicMock()
        self.chat.completions = MagicMock()
        self.chat.completions.create = MagicMock(side_effect=self._mock_create)

    def _mock_create(self, model, messages, max_tokens=None, **kwargs):
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        
        parts = []
        for msg in messages:
            if msg['role'] == 'system':
                parts.append(f"System: {msg['content']}")
            elif msg['role'] == 'user':
                parts.append(f"User: {msg['content']}")
        
        mock_message.content = f"HF Response from {model} (max_tokens={max_tokens}) for " + " | ".join(parts)
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        return mock_response

mock_openai_mod = MagicMock()
mock_openai_mod.OpenAI = MockOpenAI
sys.modules['openai'] = mock_openai_mod

# Set environment variables before module import evaluates them at module level
import os
os.environ['HF_TOKEN'] = 'my-hf-token'

# Import hf-bridge server using importlib
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("hf_server", ROOT / "hf-bridge" / "server.py")
hf_server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hf_server)


class TestHFBridge(unittest.TestCase):
    @patch.dict('os.environ', {'HF_TOKEN': 'my-hf-token'})
    def test_get_client_routing_local(self):
        # Local model (no slash) should route to local Ollama
        client = hf_server.get_client("gemma4:latest")
        self.assertEqual(client.api_key, "ollama")
        self.assertEqual(client.base_url, "http://localhost:11434/v1")

    @patch.dict('os.environ', {'HF_TOKEN': 'my-hf-token'})
    def test_get_client_routing_cloud(self):
        # Cloud model (with slash) should route to HuggingFace
        client = hf_server.get_client("google/gemma-4-31B-it")
        self.assertEqual(client.api_key, "my-hf-token")
        self.assertEqual(client.base_url, "https://router.huggingface.co/v1")

    @patch.dict('os.environ', {'HF_TOKEN': 'my-hf-token'})
    def test_ask_hf_local(self):
        result = hf_server.ask_hf(prompt="Hello local")
        self.assertEqual(
            result, 
            "HF Response from gemma4:latest (max_tokens=4096) for User: Hello local"
        )

    @patch.dict('os.environ', {'HF_TOKEN': 'my-hf-token'})
    def test_ask_hf_cloud(self):
        result = hf_server.ask_hf(prompt="Hello cloud", model="google/gemma-4-31B-it")
        self.assertEqual(
            result, 
            "HF Response from google/gemma-4-31B-it (max_tokens=4096) for User: Hello cloud"
        )

    @patch.dict('os.environ', {'HF_TOKEN': 'my-hf-token'})
    def test_ask_hf_with_context(self):
        result = hf_server.ask_hf_with_context(
            system="Be concise", 
            prompt="Hello there", 
            model="google/gemma-4-31B-it"
        )
        self.assertEqual(
            result, 
            "HF Response from google/gemma-4-31B-it (max_tokens=4096) for System: Be concise | User: Hello there"
        )


if __name__ == "__main__":
    unittest.main()
