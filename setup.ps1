# setup.ps1 — Windows setup script for AI Bridges
# Run from the repo root: .\setup.ps1

$bridges = @("gpt-bridge", "groq-bridge", "gemini-bridge", "hf-bridge", "openrouter-bridge", "kasa-bridge")
$root = $PSScriptRoot

Write-Host ""
Write-Host "AI Bridges Setup" -ForegroundColor Cyan
Write-Host "================" -ForegroundColor Cyan
Write-Host ""

$jobs = @()
foreach ($bridge in $bridges) {
    $path = Join-Path $root $bridge
    $job = Start-Job -ScriptBlock {
        param($path, $bridge)
        python -m venv "$path\venv" 2>$null
        & "$path\venv\Scripts\pip.exe" install -r "$path\requirements.txt" --quiet --upgrade 2>$null
        return $bridge
    } -ArgumentList $path, $bridge
    $jobs += $job
}

Write-Host "Setting up all bridges in parallel (this may take a moment)..." -ForegroundColor Cyan

$remainingJobs = @($jobs)
while ($remainingJobs.Count -gt 0) {
    $completedJob = Wait-Job -Job $remainingJobs -Any
    $bridgeName = Receive-Job -Job $completedJob
    if ($bridgeName) {
        Write-Host "  $($bridgeName): done" -ForegroundColor Green
    }
    $remainingJobs = @($remainingJobs | Where-Object { $_.Id -ne $completedJob.Id })
    Remove-Job -Job $completedJob
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
