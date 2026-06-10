"""Integrity tests for the MCP bridge servers.

Bridges are FastMCP servers registered with Claude Code by absolute path, so
the failure modes that matter are structural: a bridge dir losing its server
or requirements, a syntax error from a bad edit, a server that no longer
declares any MCP tools. No network calls and no bridge-venv imports — these
run under the engine's Python.
"""
import ast
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BRIDGE_DIRS = sorted(p for p in ROOT.iterdir()
                     if p.is_dir() and p.name.endswith("-bridge"))
BASH = shutil.which("bash")
PWSH = shutil.which("pwsh") or shutil.which("powershell")


def bridge_sources(bridge: Path):
    return [p for p in bridge.rglob("*.py")
            if "__pycache__" not in p.parts
            and not any(part.startswith((".venv", "venv")) for part in p.parts)]


class BridgeLayout(unittest.TestCase):
    def test_bridges_discovered(self):
        self.assertGreaterEqual(len(BRIDGE_DIRS), 5,
                                f"expected the bridge family, found {[p.name for p in BRIDGE_DIRS]}")

    def test_every_bridge_has_server_and_requirements(self):
        for bridge in BRIDGE_DIRS:
            self.assertTrue((bridge / "server.py").is_file(), f"{bridge.name}: server.py missing")
            self.assertTrue((bridge / "requirements.txt").is_file(),
                            f"{bridge.name}: requirements.txt missing")


class BridgeSources(unittest.TestCase):
    def test_python_sources_parse(self):
        for bridge in BRIDGE_DIRS:
            for src in bridge_sources(bridge):
                try:
                    ast.parse(src.read_text(encoding="utf-8"), filename=str(src))
                except SyntaxError as e:
                    self.fail(f"{src.relative_to(ROOT)}: {e}")

    def test_every_server_declares_mcp_tools(self):
        for bridge in BRIDGE_DIRS:
            text = (bridge / "server.py").read_text(encoding="utf-8")
            self.assertIn("FastMCP", text, f"{bridge.name}: server.py is not a FastMCP server")
            self.assertIn("@mcp.tool", text, f"{bridge.name}: server.py declares no MCP tools")

    def test_no_hardcoded_api_keys(self):
        # Keys come from the environment; a literal assignment is a leak.
        for bridge in BRIDGE_DIRS:
            for src in bridge_sources(bridge):
                tree = ast.parse(src.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    if (isinstance(node, ast.keyword) and node.arg == "api_key"
                            and isinstance(node.value, ast.Constant)
                            and isinstance(node.value.value, str)
                            and len(node.value.value) > 12):
                        self.fail(f"{src.relative_to(ROOT)}: hardcoded api_key literal")


class SetupScripts(unittest.TestCase):
    @unittest.skipUnless(BASH, "bash not on PATH")
    def test_setup_sh_syntax(self):
        # Pass the script as a -c string: the PATH bash may be WSL's, which
        # can't read C:\ paths (and its /dev/stdin is unreliable under interop).
        r = subprocess.run(
            [BASH, "-nc", (ROOT / "setup.sh").read_text(encoding="utf-8")],
            capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 0, r.stderr[-400:])

    @unittest.skipUnless(PWSH, "no PowerShell on PATH")
    def test_setup_ps1_syntax(self):
        probe = (
            "$t=$null;$e=$null;"
            "[System.Management.Automation.PSParser]::Tokenize("
            "(Get-Content -Raw '" + str(ROOT / "setup.ps1") + "'),[ref]$e)|Out-Null;"
            "if($e.Count){exit 1}"
        )
        r = subprocess.run([PWSH, "-NoProfile", "-NonInteractive", "-Command", probe],
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, "setup.ps1 has parse errors")


if __name__ == "__main__":
    unittest.main(verbosity=1)
