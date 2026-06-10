#!/bin/bash
# setup.sh — Mac/Linux setup script for AI Bridges
# Run from the repo root: bash setup.sh

BRIDGES=("gpt-bridge" "groq-bridge" "gemini-bridge" "hf-bridge" "openrouter-bridge" "kasa-bridge")
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "AI Bridges Setup"
echo "================"
echo ""

echo "Setting up all bridges in parallel (this may take a moment)..."
for bridge in "${BRIDGES[@]}"; do
    (
        path="$ROOT/$bridge"
        python3 -m venv "$path/venv" 2>/dev/null
        "$path/venv/bin/pip" install -r "$path/requirements.txt" --quiet --upgrade 2>/dev/null
        echo "  $bridge: done"
    ) &
done
wait

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
)

for folder in "${BRIDGES[@]}"; do
    name="${NAMES[$folder]}"
    echo "claude mcp add $name -- \"$ROOT/$folder/venv/bin/python\" \"$ROOT/$folder/server.py\""
done

echo ""
echo "See README.md for API key setup."
