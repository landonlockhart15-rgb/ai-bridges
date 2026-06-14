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
    def __init__(self, api_key=None, base_url=None, default_headers=None):
        self.api_key = api_key
        self.base_url = base_url
        self.default_headers = default_headers
        self.chat = MagicMock()
        self.chat.completions = MagicMock()
        self.chat.completions.create = MagicMock(side_effect=self._mock_create)

    def _mock_create(self, model, messages, **kwargs):
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        
        # Format a response that includes system instruction if present
        parts = []
        for msg in messages:
            if msg['role'] == 'system':
                parts.append(f"System: {msg['content']}")
            elif msg['role'] == 'user':
                parts.append(f"User: {msg['content']}")
        
        mock_message.content = f"Response from {model} for " + " | ".join(parts)
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        return mock_response

mock_openai_mod = MagicMock()
mock_openai_mod.OpenAI = MockOpenAI
sys.modules['openai'] = mock_openai_mod

# Import openrouter-bridge server using importlib
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("openrouter_server", ROOT / "openrouter-bridge" / "server.py")
openrouter_server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(openrouter_server)


class TestOpenRouterBridge(unittest.TestCase):
    @patch.dict('os.environ', {
        'OPENROUTER_API_KEY': 'test-key',
        'OPENROUTER_REFERER': 'test-referer',
        'OPENROUTER_TITLE': 'test-title'
    })
    def test_client_initialization(self):
        # Retrieve client instance via private method to inspect initialization
        client = openrouter_server._client()
        self.assertEqual(client.api_key, 'test-key')
        self.assertEqual(client.base_url, 'https://openrouter.ai/api/v1')
        self.assertEqual(client.default_headers, {
            "HTTP-Referer": "test-referer",
            "X-Title": "test-title",
        })

    @patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test-key'})
    def test_ask_openrouter_default_model(self):
        result = openrouter_server.ask_openrouter(prompt="Hello OpenRouter")
        self.assertEqual(
            result, 
            "Response from nvidia/nemotron-3-super-120b-a12b:free for User: Hello OpenRouter"
        )

    @patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test-key'})
    def test_ask_openrouter_custom_model(self):
        result = openrouter_server.ask_openrouter(
            prompt="Hello OpenRouter", 
            model="google/gemma-4-31b-it:free"
        )
        self.assertEqual(
            result, 
            "Response from google/gemma-4-31b-it:free for User: Hello OpenRouter"
        )

    @patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test-key'})
    def test_ask_openrouter_with_context(self):
        result = openrouter_server.ask_openrouter_with_context(
            system="You are a helpful assistant", 
            prompt="Hello"
        )
        self.assertEqual(
            result, 
            "Response from nvidia/nemotron-3-super-120b-a12b:free for System: You are a helpful assistant | User: Hello"
        )


if __name__ == "__main__":
    unittest.main()
