#!/usr/bin/env pwsh
#MISE description="Start Open WebUI on :3000 with logging"
#MISE alias="sw"
#MISE depends=["ensure-logs-dir"]

param()

$root = Split-Path -Parent $PSScriptRoot
$root = Split-Path -Parent $root
$logDir = Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$existing = Get-NetTCPConnection -LocalPort 3000 -State Listen -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Open WebUI already running on :3000 (PID $($existing.OwningProcess))"
    exit 0
}

$webuiDir = $env:WEBUI_DIR ?? "F:/ProjectGorgon/mywebui"
$exe = 'uv'
$serverArgs = @('run','--directory',$webuiDir,'open-webui','serve','--port','3000')
$outLog = Join-Path $logDir 'webui.log'
$errLog = Join-Path $logDir 'webui-error.log'

$batPath = [IO.Path]::GetTempFileName() + '.bat'
$batContent = "@echo off`n`"$exe`" $($serverArgs -join ' ') > `"$outLog`" 2> `"$errLog`""
[IO.File]::WriteAllText($batPath, $batContent)

Start-Process -FilePath 'cmd.exe' -ArgumentList "/c `"$batPath`"" -WindowStyle Hidden

Write-Host "Started Open WebUI (:3000) - logs in logs/webui.log"