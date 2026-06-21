# setup.ps1 — Windows setup script for AI Bridges
# Run from the repo root: .\setup.ps1

$bridges = @("gpt-bridge", "groq-bridge", "gemini-bridge", "hf-bridge", "openrouter-bridge", "kasa-bridge", "cerebras-bridge", "smart-router-bridge")
$root = $PSScriptRoot

Write-Host ""
Write-Host "AI Bridges Setup" -ForegroundColor Cyan
Write-Host "================" -ForegroundColor Cyan
Write-Host ""

if (-not (Get-Command "python" -ErrorAction SilentlyContinue)) {
    Write-Host "Error: python is not installed or not in PATH." -ForegroundColor Red
    exit 1
}

$ErrorActionPreference = "Stop"

foreach ($bridge in $bridges) {
    $path = Join-Path $root $bridge
    Write-Host "Setting up $bridge..." -NoNewline
    
    try {
        if (-not (Test-Path $path -PathType Container)) {
            throw "Directory '$bridge' does not exist."
        }
        
        $reqPath = Join-Path $path "requirements.txt"
        if (-not (Test-Path $reqPath -PathType Leaf)) {
            throw "requirements.txt not found in '$bridge'."
        }
        
        python -m venv "$path\venv" 2>$null
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path "$path\venv\Scripts\pip.exe" -PathType Leaf)) {
            throw "Failed to create virtual environment for '$bridge'."
        }
        
        & "$path\venv\Scripts\pip.exe" install -r $reqPath --quiet --upgrade
        if ($LASTEXITCODE -ne 0) {
            throw "pip install failed for '$bridge'."
        }
        
        Write-Host " done" -ForegroundColor Green
    }
    catch {
        Write-Host " FAILED" -ForegroundColor Red
        Write-Host "Error: $_" -ForegroundColor Red
        exit 1
    }
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
    @("kasa-bridge",       "kasa-bridge"),
    @("cerebras-bridge",   "cerebras-bridge"),
    @("smart-router-bridge", "smart-router-bridge")
)
foreach ($p in $pairs) {
    $name = $p[0]; $folder = $p[1]
    Write-Host "claude mcp add $name -- `"$here/$folder/venv/Scripts/python.exe`" `"$here/$folder/server.py`""
}

Write-Host ""
Write-Host "See README.md for API key setup." -ForegroundColor Cyan
