# setup.ps1 — Windows setup script for AI Bridges
# Run from the repo root: .\setup.ps1

$bridges = @("gpt-bridge", "groq-bridge", "gemini-bridge", "hf-bridge", "openrouter-bridge", "kasa-bridge")
$root = $PSScriptRoot

Write-Host ""
Write-Host "AI Bridges Setup" -ForegroundColor Cyan
Write-Host "================" -ForegroundColor Cyan
Write-Host ""

if (-not (Get-Command "python" -ErrorAction SilentlyContinue)) {
    Write-Host "Error: python is not installed or not in PATH." -ForegroundColor Red
    exit 1
}

foreach ($bridge in $bridges) {
    $path = Join-Path $root $bridge
    Write-Host "Setting up $bridge..." -NoNewline
    
    if (-not (Test-Path $path -PathType Container)) {
        Write-Host " FAILED" -ForegroundColor Red
        Write-Host "Error: Directory '$bridge' does not exist." -ForegroundColor Red
        exit 1
    }
    
    $reqPath = Join-Path $path "requirements.txt"
    if (-not (Test-Path $reqPath -PathType Leaf)) {
        Write-Host " FAILED" -ForegroundColor Red
        Write-Host "Error: requirements.txt not found in '$bridge'." -ForegroundColor Red
        exit 1
    }
    
    python -m venv "$path\venv" 2>$null
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path "$path\venv\Scripts\pip.exe" -PathType Leaf)) {
        Write-Host " FAILED" -ForegroundColor Red
        Write-Host "Error: Failed to create virtual environment for '$bridge'." -ForegroundColor Red
        exit 1
    }
    
    & "$path\venv\Scripts\pip.exe" install -r $reqPath --quiet --upgrade
    if ($LASTEXITCODE -ne 0) {
        Write-Host " FAILED" -ForegroundColor Red
        Write-Host "Error: pip install failed for '$bridge'." -ForegroundColor Red
        exit 1
    }
    
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
