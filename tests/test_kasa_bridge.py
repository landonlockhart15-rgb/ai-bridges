import sys
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock
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
sys.modules['fastmcp'] = mock_fastmcp_mod

mock_kasa_mod = MagicMock()
sys.modules['kasa'] = mock_kasa_mod

# Import kasa-bridge server using importlib
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("kasa_server", ROOT / "kasa-bridge" / "server.py")
kasa_server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kasa_server)


class MockFeature:
    def __init__(self, value):
        self.value = value


class MockDevice:
    def __init__(self, alias, host, is_on=False, features=None):
        self.alias = alias
        self.host = host
        self.is_on = is_on
        self.features = features or {}
        self.update = AsyncMock()
        self.turn_on = AsyncMock(side_effect=self._turn_on)
        self.turn_off = AsyncMock(side_effect=self._turn_off)
        self.set_brightness = AsyncMock(side_effect=self._set_brightness)

    def _turn_on(self):
        self.is_on = True

    def _turn_off(self):
        self.is_on = False

    def _set_brightness(self, val):
        if "brightness" in self.features:
            self.features["brightness"].value = val

    def has_feature(self, feature):
        return feature in self.features

    def get_feature(self, feature):
        return self.features.get(feature)


class TestKasaBridge(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Reset the device cache in the server
        kasa_server._device_cache.clear()
        
        # Prepare mock devices
        self.bulb = MockDevice(
            alias="Living Room Bulb", 
            host="192.168.1.10", 
            is_on=False, 
            features={"brightness": MockFeature(50)}
        )
        self.plug = MockDevice(
            alias="Kitchen Plug", 
            host="192.168.1.11", 
            is_on=True
        )
        
        # Mock Discover.discover to return our mock devices
        self.mock_devices = {
            "192.168.1.10": self.bulb,
            "192.168.1.11": self.plug
        }
        kasa_server.Discover.discover = AsyncMock(return_value=self.mock_devices)
        
        # Pre-populate device cache so tools don't need to auto-discover
        kasa_server._device_cache = {
            "living room bulb": self.bulb,
            "kitchen plug": self.plug
        }

    async def test_discover_devices(self):
        # Clear cache first to test actual discovery
        kasa_server._device_cache.clear()
        result = await kasa_server.discover_devices()
        self.assertIn("Living Room Bulb (192.168.1.10) — off", result)
        self.assertIn("Kitchen Plug (192.168.1.11) — on", result)
        self.bulb.update.assert_awaited_once()
        self.plug.update.assert_awaited_once()

    async def test_discover_devices_none_found(self):
        kasa_server._device_cache.clear()
        kasa_server.Discover.discover = AsyncMock(return_value={})
        result = await kasa_server.discover_devices()
        self.assertEqual(result, "No Kasa devices found on the network.")

    async def test_turn_on_device(self):
        result = await kasa_server.turn_on("living room bulb")
        self.assertEqual(result, "Turned on: Living Room Bulb")
        self.assertTrue(self.bulb.is_on)
        self.bulb.turn_on.assert_awaited_once()

    async def test_turn_on_device_not_found(self):
        result = await kasa_server.turn_on("nonexistent")
        self.assertEqual(result, "Device 'nonexistent' not found. Run discover_devices() first.")

    async def test_turn_off_device(self):
        result = await kasa_server.turn_off("kitchen plug")
        self.assertEqual(result, "Turned off: Kitchen Plug")
        self.assertFalse(self.plug.is_on)
        self.plug.turn_off.assert_awaited_once()

    async def test_set_brightness_success(self):
        # Bulb starts off
        result = await kasa_server.set_brightness("living room bulb", 80)
        self.assertEqual(result, "Set Living Room Bulb brightness to 80%")
        self.assertTrue(self.bulb.is_on)
        self.bulb.turn_on.assert_awaited_once()
        self.bulb.set_brightness.assert_awaited_once_with(80)
        self.assertEqual(self.bulb.get_feature("brightness").value, 80)

    async def test_set_brightness_not_supported(self):
        # Plug does not support brightness
        result = await kasa_server.set_brightness("kitchen plug", 80)
        self.assertEqual(result, "Kitchen Plug does not support brightness control.")

    async def test_get_status_bulb(self):
        result = await kasa_server.get_status("living room bulb")
        self.assertEqual(result, "Living Room Bulb is off, brightness: 50%")
        self.bulb.update.assert_awaited_once()

    async def test_get_status_plug(self):
        result = await kasa_server.get_status("kitchen plug")
        self.assertEqual(result, "Kitchen Plug is on")
        self.plug.update.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
