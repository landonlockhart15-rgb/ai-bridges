import asyncio
from fastmcp import FastMCP
from kasa import Discover, Module

mcp = FastMCP("kasa-bridge")

_device_cache: dict = {}


async def _discover():
    global _device_cache
    devices = await Discover.discover()
    for ip, dev in devices.items():
        await dev.update()
        _device_cache[dev.alias.lower()] = dev
    return _device_cache


def _find(name: str):
    key = name.lower()
    for alias, dev in _device_cache.items():
        if key in alias:
            return dev
    return None


@mcp.tool()
async def discover_devices() -> str:
    """Discover all TP-Link Kasa devices on the network."""
    devices = await _discover()
    if not devices:
        return "No Kasa devices found on the network."
    lines = []
    for alias, dev in devices.items():
        state = "on" if dev.is_on else "off"
        lines.append(f"- {dev.alias} ({dev.host}) — {state}")
    return "\n".join(lines)


@mcp.tool()
async def turn_on(device_name: str) -> str:
    """Turn on a Kasa device by name."""
    if not _device_cache:
        await _discover()
    dev = _find(device_name)
    if not dev:
        return f"Device '{device_name}' not found. Run discover_devices() first."
    await dev.turn_on()
    return f"Turned on: {dev.alias}"


@mcp.tool()
async def turn_off(device_name: str) -> str:
    """Turn off a Kasa device by name."""
    if not _device_cache:
        await _discover()
    dev = _find(device_name)
    if not dev:
        return f"Device '{device_name}' not found. Run discover_devices() first."
    await dev.turn_off()
    return f"Turned off: {dev.alias}"


@mcp.tool()
async def set_brightness(device_name: str, brightness: int) -> str:
    """Set brightness (0-100) on a Kasa smart bulb."""
    if not _device_cache:
        await _discover()
    dev = _find(device_name)
    if not dev:
        return f"Device '{device_name}' not found. Run discover_devices() first."
    if not dev.is_on:
        await dev.turn_on()
    if dev.has_feature("brightness"):
        await dev.set_brightness(brightness)
        return f"Set {dev.alias} brightness to {brightness}%"
    return f"{dev.alias} does not support brightness control."


@mcp.tool()
async def get_status(device_name: str) -> str:
    """Get the current status of a Kasa device."""
    if not _device_cache:
        await _discover()
    dev = _find(device_name)
    if not dev:
        return f"Device '{device_name}' not found. Run discover_devices() first."
    await dev.update()
    state = "on" if dev.is_on else "off"
    info = f"{dev.alias} is {state}"
    if dev.has_feature("brightness"):
        info += f", brightness: {dev.get_feature('brightness').value}%"
    return info


if __name__ == "__main__":
    mcp.run()
