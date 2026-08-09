#!/usr/bin/env pwsh
#MISE description="Start LLM server (gemma-4-26B on :8080) with logging"
#MISE alias="sl"
#MISE depends=["ensure-logs-dir"]

param()

$root = Split-Path -Parent $PSScriptRoot
$root = Split-Path -Parent $root
$logDir = Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$existing = Get-NetTCPConnection -LocalPort 8080 -State Listen -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "LLM server already running on :8080 (PID $($existing.OwningProcess))"
    exit 0
}

$exe = 'llama-server'
$logFile = Join-Path $logDir 'llm.log'
$serverArgs = @('-hf','unsloth/gemma-4-26B-A4B-it-qat-GGUF:UD-Q4_K_XL','--spec-type','draft-mtp','--spec-draft-n-max','4','-ngl','999','-fa','on','-c','16384','-np','1','--reasoning-budget','1024','--host','0.0.0.0','--port','8080',"--log-file","$logFile")

Start-Process -FilePath $exe -ArgumentList $serverArgs -WindowStyle Hidden

Write-Host "Started LLM server (:8080) - log: $logFile"