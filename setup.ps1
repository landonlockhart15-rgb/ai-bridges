# setup.ps1 — Windows setup script for AI Bridges
# Run from the repo root: .\setup.ps1

$bridges = @("gpt-bridge", "groq-bridge", "gemini-bridge", "hf-bridge", "openrouter-bridge", "kasa-bridge")
$root = $PSScriptRoot

Write-Host ""
Write-Host "AI Bridges Setup" -ForegroundColor Cyan
Write-Host "================" -ForegroundColor Cyan
Write-Host ""

foreach ($bridge in $bridges) {
    $path = Join-Path $root $bridge
    Write-Host "Setting up $bridge..." -NoNewline
    python -m venv "$path\venv" 2>$null
    & "$path\venv\Scripts\pip.exe" install -r "$path\requirements.txt" --quiet --upgrade
    Write-Host " done" -ForegroundColor Green
}

Write-Host ""
Write-Host "All bridges ready." -ForegroundColor Green
Write-Host ""
Write-Host "Next: register each bridge with Claude Code." -ForegroundColor Yellow
Write-Host "Replace C:\ai-bridges with the actual path to this folder:" -ForegroundColor Yellow
Write-Host ""

$here = $root.Replace("\", "/")
$pairs = @(
    @("gpt-bridge",        "gpt-bridge"),
    @("groq-bridge",       "groq-bridge"),
    @("gem-bridge",        "gemini-bridge"),
    @("hf-bridge",         "hf-bridge"),
    @("openrouter-bridge", "openrouter-bridge"),
    @("kasa-bridge",       "kasa-bridge")
)
foreach ($p in $pairs) {
    $name = $p[0]; $folder = $p[1]
    Write-Host "claude mcp add $name -- `"$here/$folder/venv/Scripts/python.exe`" `"$here/$folder/server.py`""
}

Write-Host ""
Write-Host "See README.md for API key setup." -ForegroundColor Cyan
