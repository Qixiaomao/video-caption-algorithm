param(
    [string]$BackendHost = "127.0.0.1",
    [int]$BackendPort = 8001,
    [string]$FrontendHost = "127.0.0.1",
    [int]$FrontendPort = 8000
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$CheckScript = Join-Path $PSScriptRoot "check_project_env.ps1"
& powershell -ExecutionPolicy Bypass -File $CheckScript

$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
$LogsDir = Join-Path $Root "logs"
if (!(Test-Path $LogsDir)) {
    New-Item -ItemType Directory -Path $LogsDir | Out-Null
}

$Timestamp = Get-Date -Format yyyyMMdd_HHmmss
$BackendLog = Join-Path $LogsDir "backend_${Timestamp}.log"
$FrontendLog = Join-Path $LogsDir "frontend_${Timestamp}.log"

$BackendCmd = "Set-Location '$Root'; `$env:PYTHONPATH='$Root'; & '$VenvPython' -m uvicorn server.app:app --host $BackendHost --port $BackendPort --reload *>&1 | Tee-Object -FilePath '$BackendLog'"
$FrontendCmd = "Set-Location '$Root'; `$env:PYTHONPATH='$Root'; & '$VenvPython' -m chainlit run frontend/chainlit_app.py -w --host $FrontendHost --port $FrontendPort *>&1 | Tee-Object -FilePath '$FrontendLog'"

Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $BackendCmd | Out-Null
Start-Sleep -Seconds 2
Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $FrontendCmd | Out-Null

Write-Host "[DONE] Backend launching at http://${BackendHost}:${BackendPort}"
Write-Host "[DONE] Frontend launching at http://${FrontendHost}:${FrontendPort}"
Write-Host "[LOG]  Backend:  $BackendLog"
Write-Host "[LOG]  Frontend: $FrontendLog"
