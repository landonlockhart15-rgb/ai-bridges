#!/bin/bash
# setup.sh — Mac/Linux setup script for AI Bridges
# Run from the repo root: bash setup.sh

set -euo pipefail

failure_handler() {
    local exit_code=$?
    if [ "$exit_code" -ne 0 ]; then
        echo " FAILED"
        echo "Error: An unexpected error occurred on line $1 (exit code $exit_code)." >&2
    fi
}
trap 'failure_handler $LINENO' ERR

BRIDGES=("gpt-bridge" "groq-bridge" "gemini-bridge" "hf-bridge" "openrouter-bridge" "kasa-bridge" "cerebras-bridge")
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "AI Bridges Setup"
echo "================"
echo ""

if ! command -v python3 &>/dev/null; then
    echo "Error: python3 is not installed or not in PATH." >&2
    exit 1
fi

for bridge in "${BRIDGES[@]}"; do
    path="$ROOT/$bridge"
    printf "Setting up %s..." "$bridge"
    
    if [ ! -d "$path" ]; then
        echo " FAILED"
        echo "Error: Directory '$bridge' does not exist." >&2
        exit 1
    fi
    
    if [ ! -f "$path/requirements.txt" ]; then
        echo " FAILED"
        echo "Error: requirements.txt not found in '$bridge'." >&2
        exit 1
    fi
    
    if ! python3 -m venv "$path/venv" 2>/dev/null; then
        echo " FAILED"
        echo "Error: Failed to create virtual environment for '$bridge'." >&2
        exit 1
    fi
    
    if ! "$path/venv/bin/pip" install -r "$path/requirements.txt" --quiet --upgrade; then
        echo " FAILED"
        echo "Error: pip install failed for '$bridge'." >&2
        exit 1
    fi
    echo " done"
done

echo ""
echo "All bridges ready."
echo ""
echo "Next: register each bridge with Claude Code."
echo "These commands use the actual path to this folder:"
echo ""

declare -A NAMES=(
    ["gpt-bridge"]="gpt-bridge"
    ["groq-bridge"]="groq-bridge"
    ["gemini-bridge"]="gem-bridge"
    ["hf-bridge"]="hf-bridge"
    ["openrouter-bridge"]="openrouter-bridge"
    ["kasa-bridge"]="kasa-bridge"
    ["cerebras-bridge"]="cerebras-bridge"
)

for folder in "${BRIDGES[@]}"; do
    name="${NAMES[$folder]}"
    echo "claude mcp add $name -- \"$ROOT/$folder/venv/bin/python\" \"$ROOT/$folder/server.py\""
done

echo ""
echo "See README.md for API key setup."
